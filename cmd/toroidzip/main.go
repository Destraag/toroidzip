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
	reanchorInterval := fs.Int("reanchor-interval", codec.DefaultReanchorInterval,
		"write a verbatim anchor every N values (default 256)")
	driftModeStr := fs.String("drift-mode", "A",
		"error-management strategy: A=reanchor (default), B=compensate, C=quantize")
	entropyModeStr := fs.String("entropy-mode", "lossless",
		"entropy mode: raw, lossless (default), quantized, or adaptive")
	sigFigs := fs.Int("sig-figs", 0,
		"significant figures to preserve (1-9); implies --entropy-mode quantized")
	precBits := fs.Int("precision", 0,
		"precision bits (1-16 for adaptive, 1-30 for quantized); cannot combine with --sig-figs")
	tolerance := fs.Float64("tolerance", 0.0,
		"max relative quantization error for adaptive mode (0=lossless-equivalent, e.g. 1e-4)")
	auto := fs.Bool("auto", false,
		"auto-select encoding parameters from data analysis")
	lossy := fs.Bool("lossy", false,
		"with --auto: use adaptive encoding at recommended precision and tolerance 1e-4")
	if err := fs.Parse(args); err != nil {
		return err
	}

	// Validation.
	if *lossy && !*auto {
		return fmt.Errorf("--lossy requires --auto")
	}
	if *sigFigs > 0 && *precBits > 0 {
		return fmt.Errorf("--sig-figs and --precision cannot both be set")
	}
	if *tolerance != 0 && strings.ToLower(*entropyModeStr) != "adaptive" {
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
	case "raw":
		entropyMode = codec.EntropyRaw
	case "lossless":
		entropyMode = codec.EntropyLossless
	case "quantized":
		entropyMode = codec.EntropyQuantized
	case "adaptive":
		entropyMode = codec.EntropyAdaptive
	default:
		return fmt.Errorf("--entropy-mode must be raw, lossless, quantized, or adaptive (got %q)", *entropyModeStr)
	}

	// --sig-figs / --precision imply quantized mode (or set precision for adaptive).
	precisionBits := 0
	if *sigFigs > 0 {
		if entropyMode == codec.EntropyAdaptive {
			precisionBits = codec.SigFigsToBits(*sigFigs)
			// Wire tolerance from sig-figs so the tiered encoder selects the right tier.
			// User may still override with an explicit --tolerance flag.
			if *tolerance == 0 {
				tol := codec.SigFigsToTolerance(*sigFigs)
				*tolerance = tol
			}
		} else {
			entropyMode = codec.EntropyQuantized
			precisionBits = codec.SigFigsToBits(*sigFigs)
		}
	} else if *precBits > 0 {
		if entropyMode != codec.EntropyAdaptive {
			entropyMode = codec.EntropyQuantized
		}
		precisionBits = *precBits
	}

	// --auto: run analysis and override settings.
	if *auto {
		intervals := []int{64, 128, 256, 512}
		driftRpt := codec.AnalyzeDrift(values, intervals)
		driftMode = driftRpt.RecommendedMode
		*reanchorInterval = driftRpt.RecommendedInterval
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
			entropyMode = codec.EntropyLossless
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
		ReanchorInterval: *reanchorInterval,
		DriftMode:        driftMode,
		EntropyMode:      entropyMode,
		PrecisionBits:    precisionBits,
		Tolerance:        *tolerance,
	}
	if err := codec.Encode(values, out, opts); err != nil {
		return fmt.Errorf("encoding: %w", err)
	}
	fmt.Printf("encoded %d values -> %s\n", len(values), fs.Arg(1))
	return nil
}

func runAnalyze(args []string) error {
	fs := flag.NewFlagSet("analyze", flag.ContinueOnError)
	if err := fs.Parse(args); err != nil {
		return err
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
	precRpt := codec.AnalyzePrecision(codec.ExtractNormalRatios(values, codec.DriftReanchor, codec.DefaultReanchorInterval))
	fmt.Println("=== Precision Analysis ===")
	fmt.Printf("  values      : %d\n", len(values))
	fmt.Printf("  coverage    : %.2f%%\n", precRpt.Coverage*100)
	fmt.Printf("  recommended : %d bits (%d sig figs)\n",
		precRpt.RecommendedBits, precRpt.RecommendedSigFigs)
	fmt.Printf("  note        : bits rounded to tier ceiling (u8=8, u16=16, u32=30);\n")
	fmt.Printf("                within a tier higher precision is free\n")
	fmt.Println("  bits  entropy(bits/sym)")
	for _, b := range []int{4, 8, 12, 16, 20, 24, 28, 30} {
		fmt.Printf("  %4d  %.4f\n", b, precRpt.Entropy[b])
	}

	// Lossless viability warning based on identity fraction.
	fmt.Printf("\n  identity fraction : %.2f%% of ratios within IdentityEpsilon (%.0e)\n",
		precRpt.IdentityFraction*100, codec.IdentityEpsilon)
	if precRpt.IdentityFraction < 0.05 {
		fmt.Printf("  WARNING: lossless mode will barely compress this data (<5%% identity events).\n")
		fmt.Printf("           Consider --entropy-mode quantized --sig-figs %d instead.\n",
			precRpt.RecommendedSigFigs)
	}

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

func usage() {
	fmt.Fprintln(os.Stderr, `toroidzip — ratio-first float64 codec

Usage:
  toroidzip encode  [flags] <input.f64> <output.tzrz>
  toroidzip decode  <input.tzrz> <output.f64>
  toroidzip analyze <input.f64>

Encode flags:
  --entropy-mode raw|lossless|quantized|adaptive
                          entropy mode (default lossless)
  --drift-mode A|B|C      error-management strategy (default A)
                          A = reanchor: periodic verbatim anchors (default, ~3x faster)
                          B = compensate: Kahan log-space (identical output to A)
                          C = quantize: ratios rounded to float32
  --reanchor-interval N   verbatim anchor every N values (default 256)
  --sig-figs N            significant figures 1-9; implies quantized (or sets
                          adaptive precision, capped at 16 bits)
  --precision B           precision bits; 1-30 for quantized, 1-16 for adaptive
  --tolerance T           max relative quantisation error for adaptive mode
                          (0 = lossless-equivalent; e.g. 1e-4 for ~4 sig figs)
                          only valid with --entropy-mode adaptive
  --auto                  auto-select parameters from data analysis
  --lossy                 with --auto: use adaptive encoding at recommended
                          precision and tolerance 1e-4`)
}
