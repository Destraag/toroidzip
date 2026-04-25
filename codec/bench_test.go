// Benchmark tests for the codec using synthetic datasets that approximate
// the four target data classes for Milestone 3.
//
// Run all benchmarks:
//
//	go test ./codec/... -bench=. -benchtime=3s -benchmem
//
// Run a specific class:
//
//	go test ./codec/... -bench=BenchmarkDataClass/Financial
//
// Dataset generators use a simple deterministic LCG so results are
// reproducible without importing math/rand or depending on a global seed.
package codec_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/Destraag/toroidzip/codec"
)

// ─── Deterministic dataset generators ───────────────────────────────────────
//
// All generators are deterministic and require no random seed argument.
// They are designed to approximate realistic distributions for each data
// class without importing a PRNG package.

// lcg is a simple 64-bit LCG (linear congruential generator) used to make
// benchmark datasets deterministic without importing math/rand.
type lcg struct{ state uint64 }

func newLCG(seed uint64) lcg { return lcg{seed | 1} }

// float returns a pseudo-random float64 in (-0.5, 0.5].
func (l *lcg) float() float64 {
	l.state = l.state*6364136223846793005 + 1442695040888963407
	return float64(int64(l.state>>11)) / float64(1<<53)
}

// makeSensorStream generates an IoT/sensor stream: slow sinusoidal drift with
// small Gaussian-like noise.  Values stay within [base*0.8, base*1.2].
// Models: temperature sensors, pressure gauges, energy consumption.
func makeSensorStream(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0xDEAD_BEEF)
	v := 100.0
	for i := range out {
		// Slow sine component (period ≈ n/3) + tiny per-step noise.
		drift := 0.2 * math.Sin(2*math.Pi*float64(i)/float64(n/3))
		noise := rng.float() * 0.002
		v *= math.Exp(drift/float64(n)*6 + noise)
		out[i] = v
	}
	return out
}

// makeFinancialWalk generates a log-normal random walk approximating equity
// tick data: σ ≈ 0.5%/step with occasional ±15% regime jumps (~1% of steps).
// Models: stock prices, FX rates, crypto tick data.
func makeFinancialWalk(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0xF17A4CE0)
	v := 100.0
	for i := range out {
		// Box-Muller: clamp u1 to (0,1) to avoid log(0).
		u1 := math.Abs(rng.float()) + 1e-9 // strictly positive
		if u1 > 1 {
			u1 = 1 - 1e-9
		}
		u2 := rng.float() + 0.5001
		normal := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		ret := normal * 0.005 // σ≈0.5% per step

		// Occasional regime jump (~1% probability): use low bits of state.
		jump := rng.float()
		if jump < -0.49 {
			ret += 0.15
		} else if jump > 0.49 {
			ret -= 0.15
		}
		v *= math.Exp(ret)
		if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			v = 100.0
		}
		out[i] = v
	}
	return out
}

// makeScientificMultiScale generates a multi-order-of-magnitude series
// approximating scientific measurement data: values oscillate between 1e-6
// and 1e6 on a smooth exponential trajectory.
// Models: astronomical luminosity, chemical concentrations, physics simulations.
func makeScientificMultiScale(n int) []float64 {
	out := make([]float64, n)
	v := 1e-6
	for i := range out {
		// Smooth sinusoidal exponent covers 12 decades over the series.
		phase := 2 * math.Pi * float64(i) / float64(n)
		logTarget := 6 * math.Sin(phase) // −6 to +6 in log10 space
		logCurrent := math.Log10(v)
		v *= math.Pow(10, (logTarget-logCurrent)*0.01) // lazy chase
		out[i] = v
	}
	return out
}

// makeVolatileSeries generates a spiky series: most ratios near 1.0 but
// ~2% of steps have a ×10 or ÷10 jump.  The majority of ratios are nearly
// ClassIdentity, making this a stress test for boundary detection.
// Models: event-driven sensor data, network packet inter-arrival times.
func makeVolatileSeries(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0xBEEF_CAFE)
	v := 1.0
	for i := range out {
		r := rng.float() + 0.5001 // uniform (0,1)
		if r < 0.01 {
			v *= 10 // large upward jump
		} else if r < 0.02 {
			v *= 0.1 // large downward jump
		} else {
			v *= 1.0 + (rng.float())*0.0002 // tiny change (~ClassIdentity boundary)
		}
		if v <= 0 {
			v = 1e-6
		}
		out[i] = v
	}
	return out
}

// makeNearConstant generates a near-flat sensor stream with only noise-floor
// variation (σ ≈ 0.01%).  Nearly all ratios should be ClassIdentity, giving
// maximum compression benefit from the entropy model.
// Models: stable temperature/humidity sensors, calibrated reference signals.
func makeNearConstant(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0xC0FFEE42)
	v := 273.15 // Kelvin
	for i := range out {
		v *= 1 + rng.float()*0.0001 // ±0.01% per step
		out[i] = v
	}
	return out
}

// makeNeuralWeightProxy generates a weight-tensor proxy: values are drawn from
// a mixture of narrow Gaussians at ±0.01 and ±0.1, slowly transitioning
// between "layers" (every n/4 steps).
// Models: MLP weight tensors, attention head projections.
func makeNeuralWeightProxy(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0x1337_C0DE)
	// Layer centres at different scales.
	centres := [4]float64{0.001, 0.05, -0.03, 0.12}
	for i := range out {
		layer := (i / (n / 4)) % 4
		// Box-Muller with safe u1.
		u1 := math.Abs(rng.float())*0.99 + 0.01 // (0.01, 1.0)
		u2 := rng.float() + 0.5001
		normal := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		v := centres[layer] + normal*0.01
		if v == 0 || math.IsNaN(v) {
			v = 1e-9
		}
		out[i] = v
	}
	return out
}

// makeFloat32PrecisionSmooth generates a smooth sensor-like stream where the
// signal has float32-class precision (~7 sig figs of real variation). Values
// are generated at float64, then rounded to float32 at each step to simulate
// storing float32 instrument readings in a float64 array.
// Models: geospatial coordinates, scientific instrument output, CFD fields,
// neural weight storage at float32 fidelity.
func makeFloat32PrecisionSmooth(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0xF32F32F3)
	v := 100.0
	for i := range out {
		// Slow sinusoidal drift + float32-class noise (7 sf of real signal).
		drift := 0.3 * math.Sin(2*math.Pi*float64(i)/float64(n/4))
		noise := rng.float() * 0.0001
		v *= math.Exp(drift/float64(n)*6 + noise)
		// Round to float32 precision: this is the key difference from Sensor.
		// The resulting float64 values differ from each other by ~1e-7 relative,
		// giving ~7 sf of genuine signal variation.
		out[i] = float64(float32(v))
	}
	return out
}

// ─── Correctness smoke tests for each generator ─────────────────────────────
// These are fast sanity checks that each generator produces finite, non-zero
// values and round-trips cleanly through the default encoder.

func TestDatasetSensorStream(t *testing.T) {
	testDatasetRoundTrip(t, "SensorStream", makeSensorStream(2000),
		codec.EncodeOptions{EntropyMode: codec.EntropyAdaptive, PrecisionBits: 16, Tolerance: 1e-9, DriftMode: codec.DriftCompensate},
		1e-6)
}

func TestDatasetFinancialWalk(t *testing.T) {
	testDatasetRoundTrip(t, "FinancialWalk", makeFinancialWalk(2000),
		codec.EncodeOptions{EntropyMode: codec.EntropyAdaptive, PrecisionBits: 16, Tolerance: 1e-9, DriftMode: codec.DriftCompensate},
		1e-6)
}

func TestDatasetScientificMultiScale(t *testing.T) {
	testDatasetRoundTrip(t, "ScientificMultiScale", makeScientificMultiScale(2000),
		codec.EncodeOptions{EntropyMode: codec.EntropyAdaptive, PrecisionBits: 16, Tolerance: 1e-9, DriftMode: codec.DriftCompensate},
		1e-6)
}

func TestDatasetVolatileSeries(t *testing.T) {
	testDatasetRoundTrip(t, "VolatileSeries", makeVolatileSeries(2000),
		codec.EncodeOptions{EntropyMode: codec.EntropyAdaptive, PrecisionBits: 16, Tolerance: 1e-9, DriftMode: codec.DriftCompensate},
		1e-6)
}

func TestDatasetNearConstant(t *testing.T) {
	testDatasetRoundTrip(t, "NearConstant", makeNearConstant(2000),
		codec.EncodeOptions{EntropyMode: codec.EntropyAdaptive, PrecisionBits: 16, Tolerance: 1e-9, DriftMode: codec.DriftCompensate},
		1e-6)
}

func TestDatasetNeuralWeightProxy(t *testing.T) {
	// Weights change sign → many boundary events; verify round-trip produces
	// finite values rather than checking relative error across zeros.
	values := makeNeuralWeightProxy(2000)
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{EntropyMode: codec.EntropyAdaptive, PrecisionBits: 16, Tolerance: 1e-9}); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(values))
	}
	// Weights are stored verbatim via boundary+reanchor events; spot-check a few.
	for _, i := range []int{0, 100, 500, 999} {
		if math.IsNaN(got[i]) || math.IsInf(got[i], 0) {
			t.Errorf("index %d: non-finite %v", i, got[i])
		}
	}
}

// testDatasetRoundTrip is a shared helper that encodes+decodes and checks
// relative error is below tolerance for all non-zero expected values.
func testDatasetRoundTrip(t *testing.T, name string, values []float64,
	opts codec.EncodeOptions, tolRel float64) {
	t.Helper()
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, opts); err != nil {
		t.Fatalf("%s Encode: %v", name, err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("%s Decode: %v", name, err)
	}
	if len(got) != len(values) {
		t.Fatalf("%s: length mismatch got=%d want=%d", name, len(got), len(values))
	}
	for i, want := range values {
		if math.IsNaN(got[i]) || math.IsInf(got[i], 0) {
			t.Errorf("%s index %d: non-finite %v", name, i, got[i])
			continue
		}
		if want == 0 {
			continue
		}
		if e := math.Abs(got[i]-want) / math.Abs(want); e > tolRel {
			t.Errorf("%s index %d: rel_err=%e > %e (got=%v want=%v)", name, i, e, tolRel, got[i], want)
		}
	}
}

// ─── Compression ratio smoke tests ──────────────────────────────────────────
// Verify each dataset actually compresses vs uncompressed float64.

func TestCompressionRatioSensorStream(t *testing.T) {
	testCompressionVsRaw(t, "SensorStream", makeSensorStream(5000),
		codec.EncodeOptions{EntropyMode: codec.EntropyAdaptive, PrecisionBits: 16, Tolerance: 1e-4})
}

func TestCompressionRatioNearConstant(t *testing.T) {
	testCompressionVsRaw(t, "NearConstant", makeNearConstant(5000),
		codec.EncodeOptions{EntropyMode: codec.EntropyAdaptive, PrecisionBits: 16, Tolerance: 1e-4})
}

func TestCompressionRatioFinancial(t *testing.T) {
	// Financial data has high entropy — adaptive may not compress vs uncompressed.
	// Just verify encode+decode work; do not assert smaller.
	values := makeFinancialWalk(5000)
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{EntropyMode: codec.EntropyAdaptive, PrecisionBits: 16, Tolerance: 1e-4}); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if _, err := codec.Decode(&buf); err != nil {
		t.Fatalf("Decode: %v", err)
	}
}

func testCompressionVsRaw(t *testing.T, name string, values []float64, opts codec.EncodeOptions) {
	t.Helper()
	rawBytes := len(values) * 8
	var encBuf bytes.Buffer
	if err := codec.Encode(values, &encBuf, opts); err != nil {
		t.Fatalf("%s enc encode: %v", name, err)
	}
	if encBuf.Len() >= rawBytes {
		t.Errorf("%s: entropy-coded (%d B) not smaller than uncompressed (%d B)",
			name, encBuf.Len(), rawBytes)
	}
}

// ─── Benchmarks ─────────────────────────────────────────────────────────────

// BenchmarkDataClass runs all six data classes × two entropy modes so M3
// can be seeded with Go benchmark output before external codec comparisons.
func BenchmarkDataClass(b *testing.B) {
	const n = 100_000
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
	modes := []struct {
		name string
		opts codec.EncodeOptions
	}{
		{"Quantized8b", codec.EncodeOptions{EntropyMode: codec.EntropyQuantized, PrecisionBits: 8, DriftMode: codec.DriftCompensate}},
		{"Adaptive16b", codec.EncodeOptions{EntropyMode: codec.EntropyAdaptive, PrecisionBits: 16, Tolerance: 1e-4, DriftMode: codec.DriftCompensate}},
	}

	for _, ds := range datasets {
		for _, m := range modes {
			ds, m := ds, m
			b.Run(ds.name+"/"+m.name+"/Encode", func(b *testing.B) {
				b.SetBytes(int64(len(ds.data)) * 8)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var buf bytes.Buffer
					_ = codec.Encode(ds.data, &buf, m.opts)
				}
			})
			// Pre-encode once for decode benchmark.
			var enc bytes.Buffer
			_ = codec.Encode(ds.data, &enc, m.opts)
			encData := enc.Bytes()
			b.Run(ds.name+"/"+m.name+"/Decode", func(b *testing.B) {
				b.SetBytes(int64(len(ds.data)) * 8)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = codec.Decode(bytes.NewReader(encData))
				}
			})
			// Log compressed size once (visible with -v).
			b.Run(ds.name+"/"+m.name+"/Size", func(b *testing.B) {
				rawBytes := len(ds.data) * 8
				b.ReportMetric(float64(len(encData))/float64(rawBytes), "ratio")
				b.ReportMetric(float64(len(encData)), "bytes")
				b.SkipNow()
			})
		}
	}
}

// BenchmarkAnalyzeDrift benchmarks the drift analyser across dataset types.
func BenchmarkAnalyzeDrift(b *testing.B) {
	const n = 10_000
	intervals := []int{64, 128, 256, 512}
	datasets := []struct {
		name string
		data []float64
	}{
		{"Sensor", makeSensorStream(n)},
		{"Financial", makeFinancialWalk(n)},
		{"Volatile", makeVolatileSeries(n)},
	}
	for _, ds := range datasets {
		ds := ds
		b.Run(ds.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = codec.AnalyzeDrift(ds.data, intervals)
			}
		})
	}
}

// BenchmarkAnalyzePrecisionByDataset benchmarks the precision analyser.
func BenchmarkAnalyzePrecisionByDataset(b *testing.B) {
	const n = 10_000
	datasets := []struct {
		name string
		data []float64
	}{
		{"Sensor", makeSensorStream(n)},
		{"Financial", makeFinancialWalk(n)},
		{"MultiScale", makeScientificMultiScale(n)},
	}
	for _, ds := range datasets {
		ds := ds
		b.Run(ds.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = codec.AnalyzePrecision(ds.data)
			}
		})
	}
}

// BenchmarkAdaptiveEncode profiles the adaptive (v4) encode path:
// ratio compute loop, per-ratio ε decision, and rANS table build.
// Run with -cpuprofile=cpu.prof to capture a pprof profile.
func BenchmarkAdaptiveEncode(b *testing.B) {
	const n = 50_000
	data := makeSensorStream(n)
	opts := codec.EncodeOptions{
		EntropyMode:      codec.EntropyAdaptive,
		DriftMode:        codec.DriftReanchor,
		ReanchorInterval: 256,
		PrecisionBits:    16,
		Tolerance:        1e-4,
	}
	b.SetBytes(int64(n) * 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		_ = codec.Encode(data, &buf, opts)
	}
}

// BenchmarkAdaptiveDecode profiles the adaptive (v4) decode path:
// rANS decode, ClassNormalExact fallback handling.
func BenchmarkAdaptiveDecode(b *testing.B) {
	const n = 50_000
	data := makeSensorStream(n)
	opts := codec.EncodeOptions{
		EntropyMode:      codec.EntropyAdaptive,
		DriftMode:        codec.DriftReanchor,
		ReanchorInterval: 256,
		PrecisionBits:    16,
		Tolerance:        1e-4,
	}
	var enc bytes.Buffer
	_ = codec.Encode(data, &enc, opts)
	encData := enc.Bytes()
	b.SetBytes(int64(n) * 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = codec.Decode(bytes.NewReader(encData))
	}
}

// ─── DriftMode comparison benchmark (17c-iv) ─────────────────────────────────
//
// BenchmarkDriftMode compares all three drift modes (Reanchor/Compensate/Quantize)
// across two configurations:
//
//	WithAdaptiveReanchor: AdaptiveReanchor=true, EndToEndTolerance=SigFigsToTolerance(4).
//	  Under adaptive reanchor the end-to-end error check dominates both modes;
//	  all three fire reanchors at the same points → identical output sizes.
//	  The benchmark isolates the per-mode overhead above the shared baseline.
//
//	WithoutAdaptiveReanchor: periodic-only reanchor at interval 256.
//	  Here drift mode IS orthogonal to compression: Compensate drifts less
//	  between anchors than Reanchor, so it fires fewer reanchors and produces
//	  a smaller stream. Quantize uses float32-rounded ratios (lossy; sizes differ).
//	  This configuration shows the original design intent of each mode.
//
// A size check is printed for the WithAdaptiveReanchor case to confirm identical
// output; sizes are expected to differ in the WithoutAdaptiveReanchor case.
//
// Run with:
//
//	go test ./codec/... -bench=BenchmarkDriftMode -benchtime=3s
func BenchmarkDriftMode(b *testing.B) {
	const dataSize = 100_000

	datasets := []struct {
		name string
		data []float64
	}{
		{"Sensor", makeSensorStream(dataSize)},
		{"Financial", makeFinancialWalk(dataSize)},
		{"Drift", makeDriftStream(dataSize)},
		{"NearConstant", makeNearConstant(dataSize)},
	}

	modes := []struct {
		name string
		mode codec.DriftMode
	}{
		{"Reanchor", codec.DriftReanchor},
		{"Compensate", codec.DriftCompensate},
		{"Quantize", codec.DriftQuantize},
	}

	configs := []struct {
		tag              string
		adaptive         bool
		endToEndTol      float64
		reanchorInterval int
	}{
		{"WithAdaptiveReanchor", true, codec.SigFigsToTolerance(4), 0},
		{"WithoutAdaptiveReanchor", false, 0, 256},
	}

	for _, cfg := range configs {
		cfg := cfg
		for _, ds := range datasets {
			ds := ds

			// Size comparison across modes for this config.
			sizes := make(map[string]int, len(modes))
			for _, m := range modes {
				opts := codec.EncodeOptions{
					EntropyMode:       codec.EntropyAdaptive,
					PrecisionBits:     16,
					Tolerance:         math.MaxFloat64,
					DriftMode:         m.mode,
					AdaptiveReanchor:  cfg.adaptive,
					EndToEndTolerance: cfg.endToEndTol,
					ReanchorInterval:  cfg.reanchorInterval,
				}
				var buf bytes.Buffer
				if err := codec.Encode(ds.data, &buf, opts); err != nil {
					b.Fatalf("%s/%s/%s: encode: %v", cfg.tag, ds.name, m.name, err)
				}
				sizes[m.name] = buf.Len()
			}
			if cfg.adaptive {
				// Expect identical sizes when adaptive reanchor dominates.
				ra, co, qu := sizes["Reanchor"], sizes["Compensate"], sizes["Quantize"]
				if ra != co || ra != qu {
					b.Logf("SIZE DIFFER %s/%s: Reanchor=%d Compensate=%d Quantize=%d",
						cfg.tag, ds.name, ra, co, qu)
				}
			} else {
				// Sizes differ by design; log them for the record.
				b.Logf("sizes %s/%s: Reanchor=%d Compensate=%d Quantize=%d",
					cfg.tag, ds.name,
					sizes["Reanchor"], sizes["Compensate"], sizes["Quantize"])
			}

			for _, m := range modes {
				m := m
				name := cfg.tag + "/" + ds.name + "/" + m.name
				b.Run(name, func(b *testing.B) {
					opts := codec.EncodeOptions{
						EntropyMode:       codec.EntropyAdaptive,
						PrecisionBits:     16,
						Tolerance:         math.MaxFloat64,
						DriftMode:         m.mode,
						AdaptiveReanchor:  cfg.adaptive,
						EndToEndTolerance: cfg.endToEndTol,
						ReanchorInterval:  cfg.reanchorInterval,
					}
					b.SetBytes(int64(dataSize) * 8)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						var buf bytes.Buffer
						if err := codec.Encode(ds.data, &buf, opts); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

// ─── v7 vs v9 comparison benchmarks ─────────────────────────────────────────
//
// BenchmarkV7vsV9 compares encode throughput and compressed size between
// v7 (EntropyAdaptive, DynamicOffset=false) and v9 (DynamicOffset=true)
// across all M3 dataset classes and two parallelism levels (N=1, N=4).
//
// Run with:
//
//	go test ./codec/... -bench=BenchmarkV7vsV9 -benchtime=3s -benchmem
//
// The sub-benchmark names encode: DataClass/Mode/N=workers
// Mode is one of: v7, v9

func BenchmarkV7vsV9(b *testing.B) {
	const dataSize = 100_000

	datasets := []struct {
		name string
		data []float64
	}{
		{"Sensor", makeSensorStream(dataSize)},
		{"Financial", makeFinancialWalk(dataSize)},
		{"Scientific", makeScientificMultiScale(dataSize)},
		{"Volatile", makeVolatileSeries(dataSize)},
		{"NearConstant", makeNearConstant(dataSize)},
		{"Drift", makeDriftStream(dataSize)},
	}

	// Realistic production config: mirrors --sig-figs 4 (CLI --auto default).
	// Tolerance=MaxFloat64 lets precision bits drive per-ratio accuracy (fast path);
	// AdaptiveReanchor + EndToEndTolerance give the end-to-end guarantee.
	// This puts v7 and v9 on equal footing — both pay ratio-computation cost in
	// the segmenter, so parallel scaling numbers are comparable.
	baseOpts := codec.EncodeOptions{
		EntropyMode:       codec.EntropyAdaptive,
		PrecisionBits:     16,
		Tolerance:         math.MaxFloat64,
		AdaptiveReanchor:  true,
		EndToEndTolerance: codec.SigFigsToTolerance(4), // 5e-5
	}

	for _, ds := range datasets {
		ds := ds
		for _, dynOffset := range []bool{false, true} {
			dynOffset := dynOffset
			version := "v7"
			if dynOffset {
				version = "v9"
			}
			for _, n := range []int{1, 4} {
				n := n
				name := ds.name + "/" + version + "/N=" + itoa(n)
				b.Run(name, func(b *testing.B) {
					opts := baseOpts
					opts.DynamicOffset = dynOffset
					opts.Parallelism = n
					b.SetBytes(int64(dataSize) * 8)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						var buf bytes.Buffer
						if err := codec.Encode(ds.data, &buf, opts); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

// BenchmarkV7vsV9Size reports encoded byte sizes (not throughput) by running
// each configuration once and using b.ReportMetric. This makes size comparisons
// visible in the benchmark table alongside throughput.
//
// Run with:
//
//	go test ./codec/... -bench=BenchmarkV7vsV9Size -benchtime=1x
func BenchmarkV7vsV9Size(b *testing.B) {
	const dataSize = 100_000

	datasets := []struct {
		name string
		data []float64
	}{
		{"Sensor", makeSensorStream(dataSize)},
		{"Financial", makeFinancialWalk(dataSize)},
		{"Scientific", makeScientificMultiScale(dataSize)},
		{"Volatile", makeVolatileSeries(dataSize)},
		{"NearConstant", makeNearConstant(dataSize)},
		{"Drift", makeDriftStream(dataSize)},
	}

	baseOpts := codec.EncodeOptions{
		EntropyMode:       codec.EntropyAdaptive,
		PrecisionBits:     16,
		Tolerance:         math.MaxFloat64,
		AdaptiveReanchor:  true,
		EndToEndTolerance: codec.SigFigsToTolerance(4), // 5e-5 — matches --sig-figs 4
	}

	for _, ds := range datasets {
		ds := ds
		for _, dynOffset := range []bool{false, true} {
			dynOffset := dynOffset
			version := "v7"
			if dynOffset {
				version = "v9"
			}
			name := ds.name + "/" + version
			b.Run(name, func(b *testing.B) {
				opts := baseOpts
				opts.DynamicOffset = dynOffset
				var buf bytes.Buffer
				if err := codec.Encode(ds.data, &buf, opts); err != nil {
					b.Fatal(err)
				}
				b.ReportMetric(float64(buf.Len()), "B/stream")
				b.ReportMetric(float64(buf.Len())/float64(dataSize*8)*100, "pct_of_raw")
			})
		}
	}
}
