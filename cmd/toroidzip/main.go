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
	driftModeStr := fs.String("drift-mode", "B",
		"error-management strategy: A=reanchor, B=compensate (default), C=quantize")
	entropyModeStr := fs.String("entropy-mode", "lossless",
		"entropy mode: raw, lossless (default), or quantized")
	sigFigs := fs.Int("sig-figs", 0,
		"significant figures to preserve (1-9); implies --entropy-mode quantized")
	precBits := fs.Int("precision", 0,
		"precision bits (1-30); implies --entropy-mode quantized; cannot combine with --sig-figs")
	auto := fs.Bool("auto", false,
		"auto-select encoding parameters from data analysis")
	lossy := fs.Bool("lossy", false,
		"with --auto: use quantized (lossy) encoding at recommended precision")
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
	default:
		return fmt.Errorf("--entropy-mode must be raw, lossless, or quantized (got %q)", *entropyModeStr)
	}

	// --sig-figs / --precision imply quantized mode.
	precisionBits := 0
	if *sigFigs > 0 {
		entropyMode = codec.EntropyQuantized
		precisionBits = codec.SigFigsToBits(*sigFigs)
	} else if *precBits > 0 {
		entropyMode = codec.EntropyQuantized
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
			entropyMode = codec.EntropyQuantized
			precisionBits = precRpt.RecommendedBits
		} else {
			entropyMode = codec.EntropyLossless
		}
		fmt.Printf("auto: drift-mode=%v reanchor-interval=%d entropy-mode=%v precision-bits=%d\n",
			driftMode, *reanchorInterval, entropyMode, precisionBits)
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

	// Precision analysis.
	precRpt := codec.AnalyzePrecision(values)
	fmt.Println("=== Precision Analysis ===")
	fmt.Printf("  values      : %d\n", len(values))
	fmt.Printf("  coverage    : %.2f%%\n", precRpt.Coverage*100)
	fmt.Printf("  recommended : %d bits (%d sig figs)\n",
		precRpt.RecommendedBits, precRpt.RecommendedSigFigs)
	fmt.Println("  bits  entropy(bits/sym)")
	for _, b := range []int{4, 8, 12, 16, 20, 24, 28, 30} {
		fmt.Printf("  %4d  %.4f\n", b, precRpt.Entropy[b])
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
	recommendedModeStr := "A"
	if driftRpt.RecommendedMode == codec.DriftCompensate {
		recommendedModeStr = "B"
	}
	fmt.Printf("  recommended : mode %s, interval %d\n",
		recommendedModeStr, driftRpt.RecommendedInterval)
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
  --entropy-mode raw|lossless|quantized
                          entropy mode (default lossless)
  --drift-mode A|B|C      error-management strategy (default B)
                          A = reanchor: periodic verbatim anchors
                          B = compensate: Kahan log-space (default)
                          C = quantize: ratios rounded to float32
  --reanchor-interval N   verbatim anchor every N values (default 256)
  --sig-figs N            significant figures 1-9; implies --entropy-mode quantized
  --precision B           precision bits 1-30; implies --entropy-mode quantized
  --auto                  auto-select parameters from data analysis
  --lossy                 with --auto: use quantized encoding at recommended precision`)
}
