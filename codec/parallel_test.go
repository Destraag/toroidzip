package codec_test

// parallel_test.go covers M17a correctness and throughput benchmarks.
//
// Correctness tests:
//   TestParallelEncode_N1_Identical   -- N=1 output == single-threaded output (byte-for-byte)
//   TestParallelEncode_RoundTrip      -- N=2,4,8 decode to same values as single-threaded
//   TestParallelEncode_AdaptiveReanchor -- parallel path respects --sig-figs end-to-end guarantee
//   TestParallelEncode_ErrorOnBadDrift -- non-default DriftMode returns error
//
// Benchmarks (17a-iii):
//   BenchmarkParallelEncode -- N=1,2,4,8 on smooth 100k-element sensor stream

import (
	"bytes"
	"math"
	"testing"

	"github.com/Destraag/toroidzip/codec"
)

// parallelRoundTrip encodes with N workers and decodes. Returns decoded values.
func parallelRoundTrip(t *testing.T, values []float64, opts codec.EncodeOptions, n int) []float64 {
	t.Helper()
	opts.Parallelism = n
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, opts); err != nil {
		t.Fatalf("Encode (parallel=%d): %v", n, err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("Decode (parallel=%d): %v", n, err)
	}
	return got
}

// TestParallelEncode_N1_Identical confirms that --parallel 1 produces
// byte-identical output to the single-threaded path (v1.0.0 compatibility).
func TestParallelEncode_N1_Identical(t *testing.T) {
	values := makeSensorStream(2000)

	for _, mode := range []struct {
		name string
		opts codec.EncodeOptions
	}{
		{"quantized", codec.EncodeOptions{
			EntropyMode:   codec.EntropyQuantized,
			PrecisionBits: 16,
		}},
		{"adaptive", codec.EncodeOptions{
			EntropyMode:   codec.EntropyAdaptive,
			PrecisionBits: 16,
			Tolerance:     math.MaxFloat64,
		}},
		{"adaptive-reanchor", codec.EncodeOptions{
			EntropyMode:       codec.EntropyAdaptive,
			PrecisionBits:     codec.SigFigsToBits(6),
			Tolerance:         codec.SigFigsToTolerance(6),
			AdaptiveReanchor:  true,
			EndToEndTolerance: codec.SigFigsToTolerance(4),
		}},
	} {
		t.Run(mode.name, func(t *testing.T) {
			// Single-threaded reference (Parallelism=0).
			var refBuf bytes.Buffer
			if err := codec.Encode(values, &refBuf, mode.opts); err != nil {
				t.Fatalf("reference encode: %v", err)
			}

			// N=1 parallel — must be byte-identical.
			mode.opts.Parallelism = 1
			var parBuf bytes.Buffer
			if err := codec.Encode(values, &parBuf, mode.opts); err != nil {
				t.Fatalf("parallel=1 encode: %v", err)
			}

			if !bytes.Equal(refBuf.Bytes(), parBuf.Bytes()) {
				t.Errorf("parallel=1 output differs from single-threaded (%d vs %d bytes)",
					parBuf.Len(), refBuf.Len())
			}
		})
	}
}

// TestParallelEncode_RoundTrip encodes with N>1 workers and verifies that
// decoded values match single-threaded decoded values within float64 equality.
func TestParallelEncode_RoundTrip(t *testing.T) {
	datasets := []struct {
		name   string
		values []float64
	}{
		{"sensor-2k", makeSensorStream(2000)},
		{"financial-5k", makeFinancialWalk(5000)},
		{"volatile-1k", makeVolatileSeries(1000)},
	}

	modes := []struct {
		name string
		opts codec.EncodeOptions
	}{
		{"quantized-16b", codec.EncodeOptions{
			EntropyMode:   codec.EntropyQuantized,
			PrecisionBits: 16,
		}},
		{"adaptive-maxfp", codec.EncodeOptions{
			EntropyMode:   codec.EntropyAdaptive,
			PrecisionBits: 16,
			Tolerance:     math.MaxFloat64,
		}},
	}

	for _, ds := range datasets {
		for _, m := range modes {
			for _, n := range []int{2, 4, 8} {
				name := ds.name + "/" + m.name + "/N=" + itoa(n)
				t.Run(name, func(t *testing.T) {
					// Single-threaded reference decode.
					ref := parallelRoundTrip(t, ds.values, m.opts, 1)

					// Parallel decode.
					got := parallelRoundTrip(t, ds.values, m.opts, n)

					if len(got) != len(ref) {
						t.Fatalf("length mismatch: got %d want %d", len(got), len(ref))
					}
					for i := range ref {
						if got[i] != ref[i] {
							t.Errorf("index %d: got %v want %v", i, got[i], ref[i])
							if i > 5 {
								t.FailNow()
							}
						}
					}
				})
			}
		}
	}
}

// TestParallelEncode_AdaptiveReanchor verifies that the parallel encoder
// respects the end-to-end sig-figs guarantee with adaptive reanchoring.
func TestParallelEncode_AdaptiveReanchor(t *testing.T) {
	const sigFigs = 4
	endToEndTol := codec.SigFigsToTolerance(sigFigs)

	// Adversarial monotone sequence: all quantization errors same sign.
	const n = 1000
	values := make([]float64, n)
	v := 100.0
	for i := range values {
		v *= 1.0001
		values[i] = v
	}

	opts := codec.EncodeOptions{
		EntropyMode:       codec.EntropyAdaptive,
		PrecisionBits:     codec.SigFigsToBits(sigFigs + 2),
		Tolerance:         codec.SigFigsToTolerance(sigFigs + 2),
		AdaptiveReanchor:  true,
		EndToEndTolerance: endToEndTol,
	}

	for _, workers := range []int{2, 4} {
		t.Run(itoa(workers)+"workers", func(t *testing.T) {
			got := parallelRoundTrip(t, values, opts, workers)
			for i, want := range values {
				if want == 0 {
					continue
				}
				relErr := math.Abs(got[i]-want) / math.Abs(want)
				if relErr > endToEndTol*1.01 { // 1% tolerance on tolerance
					t.Errorf("index %d: relErr %e exceeds T_end %e", i, relErr, endToEndTol)
				}
			}
		})
	}
}

// TestParallelEncode_ErrorOnBadDrift confirms parallel mode returns an error
// when a non-default DriftMode is set.
func TestParallelEncode_ErrorOnBadDrift(t *testing.T) {
	values := makeSensorStream(500)
	opts := codec.EncodeOptions{
		EntropyMode: codec.EntropyAdaptive,
		DriftMode:   codec.DriftCompensate,
		Parallelism: 4,
	}
	var buf bytes.Buffer
	err := codec.Encode(values, &buf, opts)
	if err == nil {
		t.Fatal("expected error for DriftCompensate + parallel, got nil")
	}
}

// itoa is a minimal int-to-string helper to avoid importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	pos := len(buf)
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}

// ─── Benchmarks (17a-iii) ────────────────────────────────────────────────────
//
// Run with:
//   go test ./codec/... -bench=BenchmarkParallelEncode -benchtime=3s -benchmem

func BenchmarkParallelEncode(b *testing.B) {
	const dataSize = 100_000
	values := makeSensorStream(dataSize)

	opts := codec.EncodeOptions{
		EntropyMode:       codec.EntropyAdaptive,
		PrecisionBits:     codec.SigFigsToBits(6),
		Tolerance:         codec.SigFigsToTolerance(6),
		AdaptiveReanchor:  true,
		EndToEndTolerance: codec.SigFigsToTolerance(4),
	}

	for _, n := range []int{1, 2, 4, 8} {
		workerCount := n
		b.Run("N="+itoa(n), func(b *testing.B) {
			opts.Parallelism = workerCount
			b.SetBytes(int64(dataSize) * 8) // 8 bytes per float64
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var buf bytes.Buffer
				if err := codec.Encode(values, &buf, opts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
