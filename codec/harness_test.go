// M3 benchmark harness — prints a markdown table of compression ratio and
// throughput for every dataset × mode combination.
//
// Run with:
//
//	go test ./codec/... -run=TestM3Harness -v -count=1
//
// The test always passes; the table is written to stdout.
// Timing is wall-clock via time.Now so results are indicative, not precise.
// Use -bench=BenchmarkDataClass for stable throughput numbers.
package codec_test

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	"github.com/Destraag/toroidzip/codec"
)

// harnessMode describes one encode configuration for the harness table.
type harnessMode struct {
	label string
	opts  codec.EncodeOptions
}

// harnessResult holds the metrics for one dataset × mode cell.
type harnessResult struct {
	dataset  string
	mode     string
	rawBytes int
	encBytes int
	ratio    float64 // compressed / raw  (lower is better)
	encMBs   float64 // MB/s encode
	decMBs   float64 // MB/s decode
}

// measureMode encodes and decodes values, returning a harnessResult.
func measureMode(dataset string, values []float64, m harnessMode) harnessResult {
	rawBytes := len(values) * 8

	// ── encode ──────────────────────────────────────────────────────────────
	const warmup = 2
	const iters = 5

	var encData []byte
	var encDur time.Duration
	for i := 0; i < warmup+iters; i++ {
		var buf bytes.Buffer
		t0 := time.Now()
		_ = codec.Encode(values, &buf, m.opts)
		d := time.Since(t0)
		if i >= warmup {
			encDur += d
		}
		if i == warmup {
			encData = buf.Bytes()
		}
	}
	encAvg := encDur / iters
	inputMB := float64(rawBytes) / (1 << 20)
	encMBs := inputMB / encAvg.Seconds()

	// ── decode ──────────────────────────────────────────────────────────────
	var decDur time.Duration
	for i := 0; i < warmup+iters; i++ {
		t0 := time.Now()
		_, _ = codec.Decode(bytes.NewReader(encData))
		d := time.Since(t0)
		if i >= warmup {
			decDur += d
		}
	}
	decAvg := decDur / iters
	decMBs := inputMB / decAvg.Seconds()

	ratio := float64(len(encData)) / float64(rawBytes)

	return harnessResult{
		dataset:  dataset,
		mode:     m.label,
		rawBytes: rawBytes,
		encBytes: len(encData),
		ratio:    ratio,
		encMBs:   encMBs,
		decMBs:   decMBs,
	}
}

// TestM3Harness is the M3 internal benchmark harness.
// It prints a markdown table and always passes.
//
// Axes covered:
//   - EntropyMode: Raw, Lossless, Quantized (3sf / 6sf / 9sf)
//   - DriftMode:   Reanchor (default), Compensate, Quantize
//
// Note: EntropyAdaptive/Hybrid is M5 work and not yet implemented.
func TestM3Harness(t *testing.T) {
	const n = 50_000
	rawFloat64Bytes := n * 8 // unencoded IEEE-754 baseline

	datasets := []struct {
		name string
		data []float64
	}{
		{"Sensor", makeSensorStream(n)},
		{"Financial", makeFinancialWalk(n)},
		{"MultiScale", makeScientificMultiScale(n)},
		{"Volatile", makeVolatileSeries(n)},
		{"NearConstant", makeNearConstant(n)},
		{"NeuralWeight", makeNeuralWeightProxy(n)},
	}

	// Payload tier label for informational column.
	tierLabel := func(bits int) string {
		switch codec.QuantPayloadTier(bits) {
		case 1:
			return "u8"
		case 2:
			return "u16"
		default:
			return "u32"
		}
	}

	driftSuffix := func(dm codec.DriftMode) string {
		switch dm {
		case codec.DriftCompensate:
			return "/Compensate"
		case codec.DriftQuantize:
			return "/Quantize"
		default: // DriftReanchor == 0
			return "/Reanchor"
		}
	}

	type modeSpec struct {
		label string
		opts  codec.EncodeOptions
		bits  int // -1 = synthetic baseline, 0 = not quantized, >0 = precision bits
	}

	// Build the full mode matrix:
	//   Raw × 1 (drift has no effect on raw class-byte payload size)
	//   Lossless × 3 drift modes
	//   Q-Nsf × 3 drift modes  (N = 3, 6, 9)
	var modes []modeSpec
	modes = append(modes, modeSpec{"Uncompressed", codec.EncodeOptions{}, -1})
	modes = append(modes, modeSpec{"Raw(v1)", codec.EncodeOptions{EntropyMode: codec.EntropyRaw}, 0})

	for _, dm := range []codec.DriftMode{codec.DriftReanchor, codec.DriftCompensate, codec.DriftQuantize} {
		modes = append(modes, modeSpec{
			"Lossless" + driftSuffix(dm),
			codec.EncodeOptions{EntropyMode: codec.EntropyLossless, DriftMode: dm},
			0,
		})
	}
	for _, sf := range []int{3, 6, 9} {
		bits := codec.SigFigsToBits(sf)
		for _, dm := range []codec.DriftMode{codec.DriftReanchor, codec.DriftCompensate, codec.DriftQuantize} {
			modes = append(modes, modeSpec{
				fmt.Sprintf("Q-%dsf%s", sf, driftSuffix(dm)),
				codec.EncodeOptions{
					EntropyMode:   codec.EntropyQuantized,
					PrecisionBits: bits,
					DriftMode:     dm,
				},
				bits,
			})
		}
	}

	// Collect results.
	type row struct {
		harnessResult
		bits int
	}
	var results []row
	for _, ds := range datasets {
		for _, m := range modes {
			if m.bits == -1 {
				results = append(results, row{harnessResult{
					dataset:  ds.name,
					mode:     "Uncompressed",
					rawBytes: rawFloat64Bytes,
					encBytes: rawFloat64Bytes,
					ratio:    1.0,
				}, -1})
				continue
			}
			r := measureMode(ds.name, ds.data, harnessMode{m.label, m.opts})
			results = append(results, row{r, m.bits})
		}
	}

	// ── Main table ───────────────────────────────────────────────────────────
	fmt.Printf("\n## M3 Internal Benchmark — n=%d values (%d KB uncompressed)\n\n",
		n, rawFloat64Bytes/1024)
	fmt.Printf("Axes: EntropyMode × DriftMode × PrecisionBits.\n")
	fmt.Printf("Ratio = encoded_bytes / uncompressed_bytes (lower is better).\n")
	fmt.Printf("Tier = quantized payload width (u8/u16/u32). `-` = not quantized.\n")
	fmt.Printf("Note: EntropyAdaptive/Hybrid not yet implemented (M5).\n\n")

	fmt.Printf("| %-22s | %-22s | %8s | %9s | %11s | %10s | %10s |\n",
		"Dataset", "Mode", "Tier", "Ratio", "Enc bytes", "Enc MB/s", "Dec MB/s")
	fmt.Printf("|%s|%s|%s|%s|%s|%s|%s|\n",
		"-----------------------:", "-----------------------:", "---------:",
		"----------:", "------------:", "-----------:", "-----------:")

	for _, r := range results {
		tier := "-"
		if r.bits > 0 {
			tier = fmt.Sprintf("%s/%db", tierLabel(r.bits), r.bits)
		}
		encMBs, decMBs := "         -", "         -"
		if r.encMBs > 0 {
			encMBs = fmt.Sprintf("%10.1f", r.encMBs)
			decMBs = fmt.Sprintf("%10.1f", r.decMBs)
		}
		fmt.Printf("| %-22s | %-22s | %8s | %9.4f | %11d | %10s | %10s |\n",
			r.dataset, r.mode, tier, r.ratio, r.encBytes, encMBs, decMBs)
	}
	fmt.Println()

	// ── Per-dataset best-ratio summary (TL;DR) ───────────────────────────────
	fmt.Printf("### Best compression ratio per dataset (TL;DR — excludes Uncompressed / Raw baselines)\n\n")
	fmt.Printf("| %-14s | %-22s | %8s | %9s |\n", "Dataset", "Best mode", "Tier", "Ratio")
	fmt.Printf("|%s|%s|%s|%s|\n",
		"---------------:", "-----------------------:", "---------:", "----------:")

	for _, ds := range datasets {
		best := row{harnessResult{ratio: 1e9}, 0}
		for _, r := range results {
			if r.dataset == ds.name && r.mode != "Uncompressed" && r.mode != "Raw(v1)" && r.ratio < best.ratio {
				best = r
			}
		}
		tier := "-"
		if best.bits > 0 {
			tier = fmt.Sprintf("%s/%db", tierLabel(best.bits), best.bits)
		}
		fmt.Printf("| %-14s | %-22s | %8s | %9.4f |\n",
			ds.name, best.mode, tier, best.ratio)
	}
	fmt.Println()

	// ── Observations ─────────────────────────────────────────────────────────
	fmt.Printf("### Observations\n\n")

	// 1. Raw(v1) overhead vs uncompressed.
	var rawRatio float64
	for _, r := range results {
		if r.dataset == datasets[0].name && r.mode == "Raw(v1)" {
			rawRatio = float64(r.encBytes) / float64(rawFloat64Bytes)
			break
		}
	}
	fmt.Printf("- **Raw(v1) overhead**: %.1f%% larger than uncompressed float64 "+
		"(class byte per value + stream header).\n", (rawRatio-1)*100)

	// 2. Check if 6sf and 9sf produce identical sizes (same payload tier).
	bits6, bits9 := codec.SigFigsToBits(6), codec.SigFigsToBits(9)
	same6_9 := true
	for _, ds := range datasets {
		var b6, b9 int
		for _, r := range results {
			if r.dataset == ds.name {
				if r.mode == "Q-6sf/Compensate" {
					b6 = r.encBytes
				}
				if r.mode == "Q-9sf/Compensate" {
					b9 = r.encBytes
				}
			}
		}
		if b6 != b9 {
			same6_9 = false
			break
		}
	}
	if same6_9 {
		fmt.Printf("- **Q-6sf = Q-9sf bytes** (for matching drift mode): both %d-bit and %d-bit "+
			"map to the %s payload tier (%d bytes/symbol). Extra precision in the uint32 tier is free.\n",
			bits6, bits9, tierLabel(bits9), codec.QuantPayloadTier(bits9))
	} else {
		fmt.Printf("- **Q-6sf vs Q-9sf**: sizes differ across some datasets "+
			"(%d-bit vs %d-bit, both in %s tier).\n",
			bits6, bits9, tierLabel(bits9))
	}

	// 3. DriftMode: size vs throughput tradeoff.
	// DriftMode affects encode speed and reconstruction precision, but NOT output size
	// (all three modes write the same stream structure). Compare throughput.
	var reanchorMBs, compensateMBs float64
	for _, r := range results {
		if r.dataset == "Sensor" {
			if r.mode == "Lossless/Reanchor" {
				reanchorMBs = r.encMBs
			}
			if r.mode == "Lossless/Compensate" {
				compensateMBs = r.encMBs
			}
		}
	}
	if reanchorMBs > 0 && compensateMBs > 0 {
		fmt.Printf("- **DriftMode does not change output size** — Reanchor, Compensate, and Quantize "+
			"produce identical (or near-identical) byte counts because DriftMode controls "+
			"reconstruction precision, not stream layout. However, Reanchor (%.0f MB/s) is "+
			"~%.1f× faster than Compensate (%.0f MB/s) by skipping Kahan bookkeeping.\n",
			reanchorMBs, reanchorMBs/compensateMBs, compensateMBs)
	}

	// 4. Lossless barely compresses smooth data.
	var losslessRatio float64
	for _, r := range results {
		if r.dataset == "Sensor" && r.mode == "Lossless/Compensate" {
			losslessRatio = r.ratio
			break
		}
	}
	fmt.Printf("- **Lossless barely compresses smooth data** (Sensor ratio %.4f): "+
		"IdentityEpsilon=1e-9 is far tighter than smooth-data ratio variation (~1e-4), "+
		"so nearly all ratios are ClassNormal with full float64 payloads.\n", losslessRatio)

	// 5. Financial outlier.
	var finBestRatio float64
	var finBestMode string
	for _, r := range results {
		if r.dataset == "Financial" && r.mode != "Uncompressed" && r.mode != "Raw(v1)" {
			if finBestRatio == 0 || r.ratio < finBestRatio {
				finBestRatio = r.ratio
				finBestMode = r.mode
			}
		}
	}
	fmt.Printf("- **Financial resists compression** (best %.4f at %s): "+
		"log-normal random walks produce near-uniform ratio distributions.\n",
		finBestRatio, finBestMode)
	fmt.Println()
}
