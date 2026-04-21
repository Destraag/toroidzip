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
	"math"
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
	maxErr   float64 // max relative reconstruction error vs original (0 = lossless)
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
	var decoded []float64
	for i := 0; i < warmup+iters; i++ {
		t0 := time.Now()
		got, _ := codec.Decode(bytes.NewReader(encData))
		d := time.Since(t0)
		if i >= warmup {
			decDur += d
		}
		if i == warmup {
			decoded = got
		}
	}
	decAvg := decDur / iters
	decMBs := inputMB / decAvg.Seconds()

	ratio := float64(len(encData)) / float64(rawBytes)

	// ── max relative error ───────────────────────────────────────────────────
	var maxErr float64
	if len(decoded) == len(values) {
		for i, orig := range values {
			if orig != 0 {
				if e := math.Abs(decoded[i]-orig) / math.Abs(orig); e > maxErr {
					maxErr = e
				}
			}
		}
	}

	return harnessResult{
		dataset:  dataset,
		mode:     m.label,
		rawBytes: rawBytes,
		encBytes: len(encData),
		ratio:    ratio,
		encMBs:   encMBs,
		decMBs:   decMBs,
		maxErr:   maxErr,
	}
}

// TestM3Harness is the internal benchmark harness covering M3 and M5 modes.
// It prints a markdown table and always passes.
//
// Axes covered:
//   - EntropyMode: Raw, Lossless, Quantized (3sf / 6sf / 9sf), Adaptive (ε=1e-4 / ε=1e-3)
//   - DriftMode:   Reanchor (default), Compensate, Quantize
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
	// Adaptive (v5): three tolerance levels × Reanchor only (Compensate and
	// Quantize drift modes have identical stream size; Reanchor is the
	// representative for ratio comparison).
	// ε=1e-6 exercises the ClassNormal32 (u32) mid-tier introduced in v5.
	for _, tol := range []float64{1e-6, 1e-4, 1e-3} {
		tolLabel := fmt.Sprintf("%.0e", tol)
		modes = append(modes, modeSpec{
			fmt.Sprintf("Adaptive(ε=%s)/Reanchor", tolLabel),
			codec.EncodeOptions{
				EntropyMode:   codec.EntropyAdaptive,
				PrecisionBits: 16,
				Tolerance:     tol,
				DriftMode:     codec.DriftReanchor,
			},
			16,
		})
	}
	// Adaptive sig-figs (v5, end-to-end guarantee): N sig figs end-to-end
	// using adaptive reanchoring.  "G-Nsf" = guaranteed N sig figs.
	// Per-ratio ε = SigFigsToTolerance(N+2); K_max = 100; T_end = SigFigsToTolerance(N).
	for _, sf := range []int{4, 6} {
		sf := sf
		bits := codec.SigFigsToBits(sf + 2)
		modes = append(modes, modeSpec{
			fmt.Sprintf("G-%dsf/AdaptiveReanchor", sf),
			codec.EncodeOptions{
				EntropyMode:       codec.EntropyAdaptive,
				PrecisionBits:     bits,
				Tolerance:         codec.SigFigsToTolerance(sf + 2),
				ReanchorInterval:  codec.SigFigsToMaxK(sf),
				AdaptiveReanchor:  true,
				EndToEndTolerance: codec.SigFigsToTolerance(sf),
				DriftMode:         codec.DriftReanchor,
			},
			bits,
		})
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
	fmt.Printf("Tier = quantized payload width (u8/u16/u32). `-` = not quantized.\n\n")

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
	fmt.Printf("| %-14s | %-30s | %8s | %9s | %12s |\n", "Dataset", "Best mode", "Tier", "Ratio", "MaxErr")
	fmt.Printf("|%s|%s|%s|%s|%s|\n",
		"---------------:", "-------------------------------:", "---------:", "----------:", "-------------:")

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
		maxErrStr := "            0"
		if best.maxErr > 0 {
			maxErrStr = fmt.Sprintf("%12.3e", best.maxErr)
		}
		fmt.Printf("| %-14s | %-30s | %8s | %9.4f | %12s |\n",
			ds.name, best.mode, tier, best.ratio, maxErrStr)
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

	// 6. Adaptive vs Quantized comparison.
	var adaptTight, adaptFine, adaptCoarse, q3sf, q6sf float64
	for _, r := range results {
		if r.dataset == "Sensor" {
			switch r.mode {
			case "Adaptive(ε=1e-06)/Reanchor":
				adaptTight = r.ratio
			case "Adaptive(ε=1e-04)/Reanchor":
				adaptFine = r.ratio
			case "Adaptive(ε=1e-03)/Reanchor":
				adaptCoarse = r.ratio
			case "Q-3sf/Reanchor":
				q3sf = r.ratio
			case "Q-6sf/Reanchor":
				q6sf = r.ratio
			}
		}
	}
	if adaptTight > 0 {
		fmt.Printf("- **Adaptive ε=1e-6 (u32 mid-tier, Sensor)**: ratio %.4f. "+
			"At this tolerance most ratios exceed 16-bit precision and fall into the "+
			"ClassNormal32 (uint32) tier, giving ~4 bytes/ratio vs 2 for u16 or 8 for float64.\n",
			adaptTight)
	}
	if adaptFine > 0 && q6sf > 0 {
		fmt.Printf("- **Adaptive ε=1e-4 vs Q-6sf (Sensor)**: adaptive ratio %.4f vs quantized %.4f. "+
			"Adaptive routes exact-path ratios as float64 payloads, so it is larger than pure quantized "+
			"when most ratios fit within tolerance, but smaller than lossless.\n", adaptFine, q6sf)
	}
	if adaptCoarse > 0 && q3sf > 0 {
		fmt.Printf("- **Adaptive ε=1e-3 vs Q-3sf (Sensor)**: adaptive ratio %.4f vs Q-3sf %.4f.\n",
			adaptCoarse, q3sf)
	}
	fmt.Println()
}

// TestU24Savings measures what fraction of u32 payloads (ClassNormal32) would
// fit in 24 bits (quantized index < 2^24), to assess whether a 24-bit tier
// would provide meaningful savings. Runs across all 7 dataset classes and
// sig-figs values 5, 6, 7.
//
// A u32 payload uses index < 2^24 when the ratio is in the tightest ~6/8 of
// the quantization range — roughly the interior two-thirds of the log-space.
//
// Run with:
//
//	go test ./codec/... -run=TestU24Savings -v -count=1
func TestU24Savings(t *testing.T) {
	const n = 50_000

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
		{"Float32Smooth", makeFloat32PrecisionSmooth(n)},
	}

	// u24Limit is the maximum index that fits in 24 bits.
	const u24Limit = uint32(1 << 24)

	type result struct {
		dataset     string
		sf          int
		totalRatios int
		u16Count    int
		u32Count    int
		u32FitsIn24 int // u32 payloads where index < 2^24
		f64Count    int
		bytesNow    int // bytes with current u16/u32/f64 tiers
		bytesWith24 int // bytes if u32 payloads fitting in 24 bits used 3 bytes instead of 4
	}

	var results []result

	for _, ds := range datasets {
		for _, sf := range []int{5, 6, 7} {
			bits := codec.SigFigsToBits(sf + 2)
			if bits <= 0 || bits > 16 {
				bits = 16
			}
			tol := codec.SigFigsToTolerance(sf + 2)

			// Replicate tier-selection logic from gatherRans7.
			const bits30 = 30
			levels16 := uint32(1) << bits
			levels30 := uint32(1) << bits30
			delta16 := math.Pow(2, codec.QuantMaxLog2R/float64(levels16)) - 1
			delta30 := math.Pow(2, codec.QuantMaxLog2R/float64(levels30)) - 1
			fastPath := delta16 < tol

			values := ds.data
			prev := values[0]
			var r result
			r.dataset = ds.name
			r.sf = sf

			for i := 1; i < len(values); i++ {
				ratio, class := codec.ComputeRatioExported(values[i], prev)
				if class != codec.ClassNormal {
					prev = values[i]
					continue
				}
				r.totalRatios++

				if ratio == 0 {
					r.f64Count++
					r.bytesNow += 8
					r.bytesWith24 += 8
					prev = values[i]
					continue
				}

				sym16 := codec.QuantizeRatio(ratio, bits)
				dequant16 := codec.DequantizeRatio(sym16, bits)

				if fastPath || math.Abs(dequant16/ratio-1.0) < tol {
					// u16 tier
					r.u16Count++
					r.bytesNow += 2
					r.bytesWith24 += 2
				} else {
					sym30 := codec.QuantizeRatio(ratio, bits30)
					dequant30 := codec.DequantizeRatio(sym30, bits30)
					if delta30 < tol || math.Abs(dequant30/ratio-1.0) < tol {
						// u32 tier — measure how many fit in 24 bits
						r.u32Count++
						r.bytesNow += 4
						if sym30 < u24Limit {
							r.u32FitsIn24++
							r.bytesWith24 += 3
						} else {
							r.bytesWith24 += 4
						}
					} else {
						// float64 exact
						r.f64Count++
						r.bytesNow += 8
						r.bytesWith24 += 8
					}
				}
				prev = values[i]
			}
			results = append(results, r)
		}
	}

	// Print results table.
	fmt.Printf("\n## U24 Savings Analysis — n=%d values\n\n", n)
	fmt.Printf("Measures: of all ClassNormal ratios that fall into the u32 tier,\n")
	fmt.Printf("how many have index < 2^24 and would save 1 byte with a 24-bit tier?\n\n")
	fmt.Printf("| %-14s | %3s | %8s | %8s | %8s | %8s | %8s | %8s |\n",
		"Dataset", "sf", "normals", "u16", "u32", "u32<24b", "pct<24b", "saving%")
	fmt.Printf("|%s|%s|%s|%s|%s|%s|%s|%s|\n",
		"---------------:", "----:", "---------:", "---------:",
		"---------:", "---------:", "---------:", "---------:")

	for _, r := range results {
		pctFits := 0.0
		if r.u32Count > 0 {
			pctFits = float64(r.u32FitsIn24) / float64(r.u32Count) * 100
		}
		savingPct := 0.0
		if r.bytesNow > 0 {
			savingPct = float64(r.bytesNow-r.bytesWith24) / float64(r.bytesNow) * 100
		}
		fmt.Printf("| %-14s | %3d | %8d | %8d | %8d | %8d | %7.1f%% | %7.2f%% |\n",
			r.dataset, r.sf, r.totalRatios, r.u16Count, r.u32Count,
			r.u32FitsIn24, pctFits, savingPct)
	}
	fmt.Println()
}

// TestSignedOffsetSavings measures how the v6 signed-offset routing distributes
// ClassNormal ratios between the u16 (int16 offset, 2 bytes) and u32 (int32
// offset, 4 bytes) tiers across all dataset classes and tolerances.
//
// It also computes the theoretical savings vs a hypothetical v5-style encoding
// where all 30-bit ratios require 4 bytes (u32 absolute).
//
// Run with:
//
//	go test ./codec/... -run=TestSignedOffsetSavings -v -count=1
func TestSignedOffsetSavings(t *testing.T) {
	const n = 50_000
	const bits30 = 30
	const int16Max = (1 << 15) - 1

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
		{"Float32Smooth", makeFloat32PrecisionSmooth(n)},
	}

	levels30 := uint32(1) << bits30
	delta30 := math.Pow(2, codec.QuantMaxLog2R/float64(levels30)) - 1

	type result struct {
		dataset  string
		tol      float64
		u16Count int // ClassNormal: int16 offset (2 bytes)
		u32Count int // ClassNormal32: int32 offset (4 bytes)
		f64Count int // ClassNormalExact: float64 (8 bytes)
		bytesV6  int // v6 payload bytes
		bytesV5  int // hypothetical v5: all 30-bit go to u32 (4 bytes each)
		savingPct float64
	}

	tolerances := []float64{1e-3, 1e-6, 1e-9}
	var results []result

	for _, ds := range datasets {
		for _, tol := range tolerances {
			fastPath := tol > 0 && delta30 < tol
			values := ds.data
			prev := values[0]
			var r result
			r.dataset = ds.name
			r.tol = tol

			for i := 1; i < len(values); i++ {
				ratio, class := codec.ComputeRatioExported(values[i], prev)
				if class != codec.ClassNormal {
					prev = values[i]
					continue
				}
				if ratio == 0 {
					r.f64Count++
					r.bytesV6 += 8
					r.bytesV5 += 8
					prev = values[i]
					continue
				}

				off30 := codec.QuantizeRatioOffset(ratio, bits30)
				dequant30 := codec.DequantizeRatioOffset(off30, bits30)
				withinTol := fastPath || delta30 < tol || math.Abs(dequant30/ratio-1.0) < tol

				if !withinTol {
					r.f64Count++
					r.bytesV6 += 8
					r.bytesV5 += 8
				} else if off30 >= -(1<<15) && off30 <= int16Max {
					r.u16Count++
					r.bytesV6 += 2
					r.bytesV5 += 4 // would have been u32 in v5
				} else {
					r.u32Count++
					r.bytesV6 += 4
					r.bytesV5 += 4
				}
				prev = values[i]
			}
			if r.bytesV5 > 0 {
				r.savingPct = float64(r.bytesV5-r.bytesV6) / float64(r.bytesV5) * 100
			}
			results = append(results, r)
		}
	}

	fmt.Printf("\n## Signed-Offset Savings Analysis — n=%d values\n\n", n)
	fmt.Printf("Compares v6 payload bytes (u16=int16 offset / u32=int32 offset)\n")
	fmt.Printf("vs hypothetical v5 (all 30-bit ratios stored as u32, 4 bytes).\n\n")
	fmt.Printf("| %-14s | %6s | %8s | %8s | %8s | %8s | %8s | %9s |\n",
		"Dataset", "tol", "u16", "u32", "f64", "bytesV6", "bytesV5", "saving%")
	fmt.Printf("|%s|%s|%s|%s|%s|%s|%s|%s|\n",
		"---------------:", "-------:", "---------:", "---------:",
		"---------:", "---------:", "---------:", "----------:")

	for _, r := range results {
		fmt.Printf("| %-14s | %6.0e | %8d | %8d | %8d | %8d | %8d | %8.1f%% |\n",
			r.dataset, r.tol, r.u16Count, r.u32Count, r.f64Count,
			r.bytesV6, r.bytesV5, r.savingPct)
	}
	fmt.Println()
}
