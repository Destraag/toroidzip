// Command toroidzip is the CLI for the ToroidZip ratio-first float codec.
//
// Usage:
//
//	toroidzip encode  [flags] <input.f64> <output.tzrz>
//	toroidzip decode  <input.tzrz> <output.f64>
//	toroidzip analyze [flags] <input.f64>
//
// Input/output files are raw IEEE 754 little-endian float64 sequences.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/Destraag/toroidzip/codec"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	var err error
	switch os.Args[1] {
	case "encode":
		err = runEncode(os.Args[2:])
	case "decode":
		err = runDecode(os.Args[2:])
	case "analyze":
		err = runAnalyze(os.Args[2:])
	default:
		usage()
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func runEncode(args []string) error {
	fs := flag.NewFlagSet("encode", flag.ContinueOnError)
	reanchorInterval := fs.Int("reanchor-interval", 0,
		"safety cap on verbatim anchor cadence; 0 = derive from --sig-figs or use default (256)")
	driftModeStr := fs.String("drift-mode", "A",
		"error-management strategy: A=reanchor (default), B=compensate, C=quantize")
	entropyModeStr := fs.String("entropy-mode", "quantized",
		"entropy mode: quantized (default) or adaptive")
	sigFigs := fs.Int("sig-figs", 0,
		"guarantee N significant figures end-to-end in reconstructed values (1-9);\n"+
			"\timplies --entropy-mode adaptive with adaptive reanchoring")
	fs.IntVar(sigFigs, "n", 0, "shorthand for --sig-figs")
	bytesFlag := fs.Int("bytes", 0,
		"storage tier by byte width: 1=u8 (~2 sf), 2=u16 (~4 sf), 4=u32 (~9 sf);\n"+
			"\timplies --entropy-mode adaptive; cannot combine with --sig-figs")
	precBits := fs.Int("precision", 0,
		"precision bits (1-30 for adaptive or quantized); cannot combine with --sig-figs\n"+
			"\t23 bits matches the float32 mantissa (~7 sig figs); use for float32-origin data")
	tolerance := fs.Float64("tolerance", 0.0,
		"max relative quantization error for adaptive mode (e.g. 1e-4 for ~4 sig figs)")
	auto := fs.Bool("auto", false,
		"auto-select encoding parameters from data analysis")
	fs.BoolVar(auto, "a", false, "shorthand for --auto")
	lossy := fs.Bool("lossy", false,
		"with --auto: use adaptive encoding at recommended precision and tolerance 1e-4")
	parallel := fs.Int("parallel", 1,
		"number of parallel encoding goroutines (default 1 = single-threaded, identical to v1.0.0;\n"+
			"\t0 = use all CPUs; N > 1 = use N goroutines; requires --drift-mode A/reanchor)")
	dynamicOffset := fs.Bool("dynamic-offset", false,
		"enable per-segment k_center optimisation (v9 stream); reduces payload for drifting/bimodal data;\n"+
			"\tonly applies to --entropy-mode adaptive; cannot combine with --parallel > 1")
	if err := fs.Parse(args); err != nil {
		return err
	}

	// Track which flags were explicitly set so derived defaults don't clobber them.
	reanchorExplicit := false
	entropyExplicit := false
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "reanchor-interval":
			reanchorExplicit = true
		case "entropy-mode":
			entropyExplicit = true
		}
	})

	// Validation.
	if *lossy && !*auto {
		return fmt.Errorf("--lossy requires --auto")
	}
	if *bytesFlag != 0 && *bytesFlag != 1 && *bytesFlag != 2 && *bytesFlag != 3 && *bytesFlag != 4 {
		return fmt.Errorf("--bytes must be 1, 2, 3, or 4 (got %d)", *bytesFlag)
	}
	if *bytesFlag > 0 && *sigFigs > 0 {
		return fmt.Errorf("--bytes and --sig-figs cannot both be set")
	}
	if *bytesFlag > 0 && *precBits > 0 {
		return fmt.Errorf("--bytes and --precision cannot both be set")
	}
	if *bytesFlag > 0 && *auto {
		return fmt.Errorf("--bytes and --auto cannot both be set")
	}
	if *sigFigs > 0 && *precBits > 0 {
		return fmt.Errorf("--sig-figs and --precision cannot both be set")
	}
	if *sigFigs > 0 && *auto {
		return fmt.Errorf("--sig-figs and --auto cannot both be set")
	}
	if *sigFigs > 0 && entropyExplicit && strings.ToLower(*entropyModeStr) != "adaptive" {
		return fmt.Errorf("--sig-figs forces --entropy-mode adaptive; remove --entropy-mode or set it to adaptive")
	}
	if *tolerance != 0 && strings.ToLower(*entropyModeStr) != "adaptive" && *sigFigs == 0 {
		return fmt.Errorf("--tolerance only applies to --entropy-mode adaptive")
	}

	if fs.NArg() < 2 {
		return fmt.Errorf("encode requires <input.f64> <output.tzrz>")
	}

	values, err := readFloat64File(fs.Arg(0))
	if err != nil {
		return fmt.Errorf("reading input: %w", err)
	}

	var driftMode codec.DriftMode
	switch strings.ToUpper(*driftModeStr) {
	case "A":
		driftMode = codec.DriftReanchor
	case "B":
		driftMode = codec.DriftCompensate
	case "C":
		driftMode = codec.DriftQuantize
	default:
		return fmt.Errorf("--drift-mode must be A, B, or C (got %q)", *driftModeStr)
	}

	var entropyMode codec.EntropyMode
	switch strings.ToLower(*entropyModeStr) {
	case "quantized":
		entropyMode = codec.EntropyQuantized
	case "adaptive":
		entropyMode = codec.EntropyAdaptive
	default:
		return fmt.Errorf("--entropy-mode must be quantized or adaptive (got %q)", *entropyModeStr)
	}

	// Apply default reanchor interval now that we know it wasn't explicitly set.
	if *reanchorInterval <= 0 && !reanchorExplicit {
		*reanchorInterval = codec.DefaultReanchorInterval
	}

	// --sig-figs N / --bytes B: end-to-end reconstruction guarantee.
	// Forces adaptive mode; PrecisionBits=SigFigsToBits(N+2) drives tier accuracy.
	// Tolerance=MaxFloat64 lets the v7 fast path quantize all ratios via tier routing
	// (prevents ClassNormalExact eclipse). EndToEndTolerance drives adaptive reanchor.
	var endToEndTolerance float64
	adaptiveReanchor := false
	precisionBits := 0
	if *sigFigs > 0 {
		entropyMode = codec.EntropyAdaptive
		precisionBits = codec.SigFigsToBits(*sigFigs + 2) // tighter per-ratio precision
		if *tolerance == 0 {
			*tolerance = math.MaxFloat64 // fast path: B drives accuracy, not per-ratio gate
		}
		endToEndTolerance = codec.SigFigsToTolerance(*sigFigs) // T_end guarantee
		adaptiveReanchor = true
		if !reanchorExplicit {
			*reanchorInterval = codec.SigFigsToMaxK(*sigFigs) // circuit breaker = 10000
		}
		// Tier feedback: show which storage tier --sig-figs N selects.
		ceilBits := tierCeilingBits(precisionBits)
		fmt.Fprintf(os.Stderr, "-n %d: %d bits → %s tier (ceiling %d bits / %d sig figs; extra precision free; K_max=%d)\n",
			*sigFigs, precisionBits, tierName(ceilBits), ceilBits,
			codec.BitsToSigFigs(ceilBits), codec.SigFigsToMaxK(*sigFigs))
	} else if *bytesFlag > 0 {
		// --bytes B: select storage tier directly by byte width.
		// Accuracy is controlled by PrecisionBits; periodic reanchor (default 256)
		// handles accumulated drift. Adaptive reanchor is not enabled — the
		// endToEndTolerance check fires per-step, not per-series, and would flood
		// the stream with ClassReanchor events at tiers where ε_max(B) > tol.
		var ceilBits int
		switch *bytesFlag {
		case 1:
			ceilBits = 8
		case 2:
			ceilBits = 16
		case 3:
			ceilBits = 24
		default: // 4
			ceilBits = 30
		}
		sf := codec.BitsToSigFigs(ceilBits)
		entropyMode = codec.EntropyAdaptive
		precisionBits = ceilBits
		if *tolerance == 0 {
			*tolerance = math.MaxFloat64 // fast path: B drives accuracy, not per-ratio gate
		}
		fmt.Fprintf(os.Stderr, "--bytes %d: %s tier (%d bits / %d sig figs; reanchor-interval=%d)\n",
			*bytesFlag, tierName(ceilBits), ceilBits, sf, *reanchorInterval)
	} else if *precBits > 0 {
		if entropyMode != codec.EntropyAdaptive {
			entropyMode = codec.EntropyQuantized
		}
		precisionBits = *precBits
	}

	// --auto: run analysis and override settings.
	// NOTE: the tolerance set inside this block intentionally bypasses the
	// flag-validation guard above (which rejects --tolerance without
	// --entropy-mode adaptive). Here we know the mode is being forced to
	// adaptive internally, so it is safe to set tolerance directly.
	if *auto {
		intervals := []int{64, 128, 256, 512}
		driftRpt := codec.AnalyzeDrift(values, intervals)
		driftMode = driftRpt.RecommendedMode
		if !reanchorExplicit && *sigFigs == 0 {
			*reanchorInterval = driftRpt.RecommendedInterval
		}
		if *lossy {
			precRpt := codec.AnalyzePrecision(codec.ExtractNormalRatios(values, driftMode, *reanchorInterval))
			entropyMode = codec.EntropyAdaptive
			precisionBits = precRpt.RecommendedBits
			if precisionBits > 16 {
				precisionBits = 16
			}
			if *tolerance == 0 {
				*tolerance = 1e-4 // sensible default for --auto --lossy
			}
		} else {
			entropyMode = codec.EntropyQuantized
		}
		fmt.Printf("auto: drift-mode=%v reanchor-interval=%d entropy-mode=%v precision-bits=%d tolerance=%g\n",
			driftMode, *reanchorInterval, entropyMode, precisionBits, *tolerance)
	}

	out, err := os.Create(fs.Arg(1))
	if err != nil {
		return fmt.Errorf("creating output: %w", err)
	}
	defer out.Close()

	opts := codec.EncodeOptions{
		ReanchorInterval:  *reanchorInterval,
		DriftMode:         driftMode,
		EntropyMode:       entropyMode,
		PrecisionBits:     precisionBits,
		Tolerance:         *tolerance,
		AdaptiveReanchor:  adaptiveReanchor,
		EndToEndTolerance: endToEndTolerance,
		Parallelism:       *parallel,
		DynamicOffset:     *dynamicOffset,
	}
	if err := codec.Encode(values, out, opts); err != nil {
		return fmt.Errorf("encoding: %w", err)
	}
	fmt.Printf("encoded %d values -> %s\n", len(values), fs.Arg(1))
	return nil
}

func runAnalyze(args []string) error {
	fs := flag.NewFlagSet("analyze", flag.ContinueOnError)
	sigFigs := fs.Int("sig-figs", 0,
		"show adaptive tier preview for this many significant figures (1-9)")
	fs.IntVar(sigFigs, "n", 0, "shorthand for --sig-figs")
	bytesFlag := fs.Int("bytes", 0,
		"show adaptive tier preview for this byte width: 1=u8, 2=u16, 4=u32")
	tolerance := fs.Float64("tolerance", 0.0,
		"show adaptive tier preview at this relative tolerance (e.g. 1e-4)")
	precBits := fs.Int("precision", 0,
		"u16 precision bits for tier preview (default: derived from --sig-figs or recommended bits)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *bytesFlag != 0 && *bytesFlag != 1 && *bytesFlag != 2 && *bytesFlag != 3 && *bytesFlag != 4 {
		return fmt.Errorf("--bytes must be 1, 2, 3, or 4 (got %d)", *bytesFlag)
	}
	if *sigFigs > 0 && *tolerance != 0 {
		return fmt.Errorf("--sig-figs and --tolerance cannot both be set")
	}
	if *sigFigs > 0 && *bytesFlag > 0 {
		return fmt.Errorf("--sig-figs and --bytes cannot both be set")
	}
	if *bytesFlag > 0 && *tolerance != 0 {
		return fmt.Errorf("--bytes and --tolerance cannot both be set")
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("analyze requires <input.f64>")
	}
	values, err := readFloat64File(fs.Arg(0))
	if err != nil {
		return fmt.Errorf("reading input: %w", err)
	}

	// Precision analysis: AnalyzePrecision expects ClassNormal ratios, not raw
	// values. Extract them first using defaults (user hasn't chosen mode yet).
	ratios := codec.ExtractNormalRatios(values, codec.DriftReanchor, codec.DefaultReanchorInterval)
	precRpt := codec.AnalyzePrecision(ratios)
	fmt.Println("=== Precision Analysis ===")
	fmt.Printf("  values      : %d\n", len(values))
	fmt.Printf("  coverage    : %.2f%%\n", precRpt.Coverage*100)
	fmt.Printf("  recommended : %d bits (%d sig figs)\n",
		precRpt.RecommendedBits, precRpt.RecommendedSigFigs)
	fmt.Printf("  note        : bits rounded to tier ceiling (u8=8, u16=16, u24=24, u32=30);\n")
	fmt.Printf("                within a tier higher precision is free\n")
	fmt.Println("  bits  entropy(bits/sym)")
	for _, b := range []int{4, 8, 12, 16, 20, 24, 28, 30} {
		fmt.Printf("  %4d  %.4f\n", b, precRpt.Entropy[b])
	}

	// Identity fraction info.
	fmt.Printf("\n  identity fraction : %.2f%% of ratios within IdentityEpsilon (%.0e)\n",
		precRpt.IdentityFraction*100, codec.IdentityEpsilon)

	// Drift analysis.
	intervals := []int{64, 128, 256, 512}
	driftRpt := codec.AnalyzeDrift(values, intervals)
	fmt.Println("\n=== Drift Analysis ===")
	fmt.Printf("  %-8s  %-4s  %-12s  %-12s  %s\n",
		"interval", "mode", "max_rel_err", "mean_rel_err", "anchor_overhead")
	for _, row := range driftRpt.Rows {
		modeStr := "A"
		if row.Mode == codec.DriftCompensate {
			modeStr = "B"
		}
		fmt.Printf("  %-8d  %-4s  %-12.4e  %-12.4e  %.2f%%\n",
			row.Interval, modeStr, row.MaxRelErr, row.MeanRelErr, row.AnchorOverhead*100)
	}
	recommendedModeStr := "A (Reanchor)"
	if driftRpt.RecommendedMode == codec.DriftCompensate {
		recommendedModeStr = "B (Compensate)"
	}
	fmt.Printf("  recommended : mode %s, interval %d\n",
		recommendedModeStr, driftRpt.RecommendedInterval)
	fmt.Printf("  speed note  : mode A (Reanchor) encodes ~3x faster than mode B (Compensate)\n")
	fmt.Printf("                at identical output size. Use A for throughput-sensitive workloads.\n")

	// Adaptive tier preview — printed only when --tolerance, --sig-figs, or --bytes is given.
	// --bytes B: translate to the same sig-figs path via BitsToSigFigs(ceilBits).
	if *bytesFlag > 0 {
		var ceilBits int
		switch *bytesFlag {
		case 1:
			ceilBits = 8
		case 2:
			ceilBits = 16
		case 3:
			ceilBits = 24
		default: // 4
			ceilBits = 30
		}
		*sigFigs = codec.BitsToSigFigs(ceilBits)
		*precBits = ceilBits
	}
	tol := *tolerance
	bits := *precBits
	var endToEndTol float64
	if *sigFigs > 0 {
		// Use the same per-ratio ε the encoder uses: SigFigsToTolerance(N+2).
		// The end-to-end guarantee (N sig figs) is enforced by the adaptive drift
		// check in gatherRans7; K_max = SigFigsToMaxK(N) = 10000 is a circuit breaker.
		tol = codec.SigFigsToTolerance(*sigFigs + 2)
		endToEndTol = codec.SigFigsToTolerance(*sigFigs)
		if bits == 0 {
			bits = codec.SigFigsToBits(*sigFigs + 2)
		}
	}
	if tol > 0 {
		if bits == 0 {
			bits = precRpt.RecommendedBits
		}
		row := codec.AnalyzeTiersV8(ratios, bits, tol)
		fmt.Println("\n=== Adaptive Tier Preview (v8) ===")
		if endToEndTol > 0 {
			fmt.Printf("  sig-figs guarantee : %d sig figs end-to-end (T_end = %.2e)\n",
				*sigFigs, endToEndTol)
			fmt.Printf("  per-ratio epsilon  : %.2e  (K_max = %d circuit breaker)\n",
				tol, codec.SigFigsToMaxK(*sigFigs))
		} else {
			fmt.Printf("  epsilon     : %.2e\n", tol)
		}
		fmt.Printf("  precision bits : %d\n", bits)
		fmt.Printf("  normal ratios  : %d\n", row.Total)
		if row.Total > 0 {
			pU8 := float64(row.U8) / float64(row.Total) * 100
			pU16 := float64(row.U16) / float64(row.Total) * 100
			pU24 := float64(row.U24) / float64(row.Total) * 100
			pU32 := float64(row.U32) / float64(row.Total) * 100
			pF64 := float64(row.F64) / float64(row.Total) * 100
			fmt.Printf("  u8   (1 B)  : %6d  (%5.1f%%)\n", row.U8, pU8)
			fmt.Printf("  u16  (2 B)  : %6d  (%5.1f%%)\n", row.U16, pU16)
			fmt.Printf("  u24  (3 B)  : %6d  (%5.1f%%)\n", row.U24, pU24)
			fmt.Printf("  u32  (4 B)  : %6d  (%5.1f%%)\n", row.U32, pU32)
			fmt.Printf("  f64  (8 B)  : %6d  (%5.1f%%)\n", row.F64, pF64)
			fmt.Printf("  eff bytes/ratio: %.2f  (lossless=8.00)\n", row.EffectiveBytesPerRatio())
		}

		// Dynamic offset savings preview.
		dynOpts := codec.EncodeOptions{
			EntropyMode:       codec.EntropyAdaptive,
			PrecisionBits:     bits,
			Tolerance:         tol,
			ReanchorInterval:  codec.DefaultReanchorInterval,
			AdaptiveReanchor:  endToEndTol > 0,
			EndToEndTolerance: endToEndTol,
		}
		dynRpt := codec.AnalyzeDynamicOffset(values, dynOpts)
		if dynRpt.TotalSegments > 0 {
			fmt.Printf("\n=== Dynamic Offset Preview (--dynamic-offset) ===\n")
			fmt.Printf("  segments analysed  : %d\n", dynRpt.TotalSegments)
			fmt.Printf("  segments benefiting: %d  (%.0f%%)\n",
				dynRpt.BenefitSegments, dynRpt.BenefitFraction()*100)
			fmt.Printf("  default payload    : %d B\n", dynRpt.DefaultPayloadBytes)
			fmt.Printf("  optimal payload    : %d B\n", dynRpt.OptimalPayloadBytes)
			if dynRpt.SavedBytes() > 0 {
				fmt.Printf("  estimated saving   : %d B  (%.1f%% of normal-ratio payload)\n",
					dynRpt.SavedBytes(), dynRpt.PayloadReduction()*100)
			} else {
				fmt.Printf("  estimated saving   : none (dynamic offset would not help this data)\n")
			}
		}
	}

	return nil
}

func runDecode(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("decode requires <input.tzrz> <output.f64>")
	}
	in, err := os.Open(args[0])
	if err != nil {
		return fmt.Errorf("opening input: %w", err)
	}
	defer in.Close()

	values, err := codec.Decode(in)
	if err != nil {
		return fmt.Errorf("decoding: %w", err)
	}

	if err := writeFloat64File(args[1], values); err != nil {
		return fmt.Errorf("writing output: %w", err)
	}
	fmt.Printf("decoded %d values -> %s\n", len(values), args[1])
	return nil
}

func readFloat64File(path string) ([]float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data)%8 != 0 {
		return nil, fmt.Errorf("file size %d is not a multiple of 8 (not a float64 sequence)", len(data))
	}
	values := make([]float64, len(data)/8)
	for i := range values {
		bits := binary.LittleEndian.Uint64(data[i*8 : i*8+8])
		values[i] = math.Float64frombits(bits)
	}
	return values, nil
}

func writeFloat64File(path string, values []float64) error {
	data := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(data[i*8:i*8+8], math.Float64bits(v))
	}
	return os.WriteFile(path, data, 0644)
}

// tierCeilingBits returns the bit-depth ceiling of the storage tier for bits b.
//
//	bits 1–8  → 8  (u8)
//	bits 9–16 → 16 (u16)
//	bits 17+  → 30 (u32)
func tierCeilingBits(b int) int {
	switch {
	case b <= 8:
		return 8
	case b <= 16:
		return 16
	default:
		return 30
	}
}

// tierName returns the Go type name for a tier ceiling bit depth.
func tierName(ceilBits int) string {
	switch ceilBits {
	case 8:
		return "u8"
	case 16:
		return "u16"
	default:
		return "u32"
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `toroidzip — ratio-first float64 codec

Usage:
  toroidzip encode  [flags] <input.f64> <output.tzrz>
  toroidzip decode  <input.tzrz> <output.f64>
  toroidzip analyze [flags] <input.f64>

Encode flags:
  --entropy-mode quantized|adaptive
                          entropy mode (default quantized)
  --drift-mode A|B|C      error-management strategy (default A)
                          A = reanchor: periodic verbatim anchors (default, ~3x faster)
                          B = compensate: Kahan log-space (identical output to A)
                          C = quantize: ratios rounded to float32
  --reanchor-interval N   verbatim anchor every N values (default 256)
  --sig-figs N / -n N     guarantee N significant figures end-to-end (1-9);
                          implies adaptive mode with adaptive reanchoring;
                          prints selected tier to stderr
  --bytes 1|2|3|4         select storage tier by byte width (1=u8/~2sf,
                          2=u16/~4sf, 3=u24/~7sf, 4=u32/~9sf); implies adaptive mode;
                          cannot combine with --sig-figs
  --precision B           precision bits; 1-30 for quantized, 1-16 for adaptive
  --tolerance T           max relative quantisation error for adaptive mode
                          (e.g. 1e-4 for ~4 sig figs)
                          only valid with --entropy-mode adaptive
  --auto / -a             auto-select parameters from data analysis
  --lossy                 with --auto: use adaptive encoding at recommended
                          precision and tolerance 1e-4

Analyze flags:
  --sig-figs N / -n N     show adaptive tier preview for N significant figures
  --bytes 1|2|4           show adaptive tier preview for this byte width
  --tolerance T           show adaptive tier preview at tolerance T
  --precision B           u16 precision bits for tier preview (default: derived
                          from --sig-figs / --bytes or recommended bits)`)
}
