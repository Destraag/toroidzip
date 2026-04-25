package codec_test

// dynamic_offset_test.go covers M17b correctness: v9 stream format, dynamic
// offset encoding/decoding, and AnalyzeDynamicOffset report accuracy.
//
// Tests:
//   TestDynamicOffset_RoundTrip        -- v9 encodes and decodes correctly for all dataset types
//   TestDynamicOffset_ReductionOnDrift -- drifting data shows payload reduction vs v8
//   TestDynamicOffset_NoBenefitOnSmooth -- near-identity data: no regression vs v8
//   TestDynamicOffset_ErrorOnParallel  -- DynamicOffset + Parallelism > 1 returns error
//   TestAnalyzeDynamicOffset_Report    -- report is non-zero for drifting data

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/Destraag/toroidzip/codec"
)

func dynRoundTrip(t *testing.T, values []float64, opts codec.EncodeOptions) ([]float64, int) {
	t.Helper()
	opts.DynamicOffset = true
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, opts); err != nil {
		t.Fatalf("Encode (dynamic-offset): %v", err)
	}
	size := buf.Len()
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("Decode (dynamic-offset): %v", err)
	}
	return got, size
}

// makeDriftStream builds a sequence that steadily drifts upward so most ratios
// cluster around 1.05 — the prime case where dynamic offset helps.
func makeDriftStream(n int) []float64 {
	out := make([]float64, n)
	v := 100.0
	for i := range out {
		v *= 1.05
		out[i] = v
	}
	return out
}

// TestDynamicOffset_RoundTrip verifies v9 produces values identical to v8 decoding
// for a variety of data patterns and encoding options.
func TestDynamicOffset_RoundTrip(t *testing.T) {
	datasets := []struct {
		name   string
		values []float64
	}{
		{"drift-2k", makeDriftStream(2000)},
		{"sensor-2k", makeSensorStream(2000)},
		{"financial-2k", makeFinancialWalk(2000)},
		{"volatile-1k", makeVolatileSeries(1000)},
	}

	opts := codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: 16,
		Tolerance:     math.MaxFloat64,
	}

	for _, ds := range datasets {
		t.Run(ds.name, func(t *testing.T) {
			got, _ := dynRoundTrip(t, ds.values, opts)
			if len(got) != len(ds.values) {
				t.Fatalf("length mismatch: got %d want %d", len(got), len(ds.values))
			}
			// v9 must decode to same values as v8 (k_center shift cancels in decoder).
			optsV8 := opts
			var refBuf bytes.Buffer
			if err := codec.Encode(ds.values, &refBuf, optsV8); err != nil {
				t.Fatalf("v8 encode: %v", err)
			}
			ref, err := codec.Decode(&refBuf)
			if err != nil {
				t.Fatalf("v8 decode: %v", err)
			}
			for i := range ref {
				if got[i] != ref[i] {
					t.Errorf("index %d: v9=%v v8=%v", i, got[i], ref[i])
					if i > 5 {
						t.FailNow()
					}
				}
			}
		})
	}
}

// TestDynamicOffset_ReductionOnDrift confirms that a steadily drifting stream
// produces a smaller encoded size with --dynamic-offset than without.
func TestDynamicOffset_ReductionOnDrift(t *testing.T) {
	values := makeDriftStream(5000)

	opts := codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: 16,
		Tolerance:     math.MaxFloat64,
	}

	// v8 size.
	var v8Buf bytes.Buffer
	if err := codec.Encode(values, &v8Buf, opts); err != nil {
		t.Fatalf("v8 encode: %v", err)
	}

	// v9 size.
	_, v9Size := dynRoundTrip(t, values, opts)

	if v9Size >= v8Buf.Len() {
		t.Errorf("dynamic offset produced no savings on drift data: v8=%d B v9=%d B",
			v8Buf.Len(), v9Size)
	} else {
		t.Logf("drift data: v8=%d B, v9=%d B, saved=%d B (%.1f%%)",
			v8Buf.Len(), v9Size, v8Buf.Len()-v9Size,
			float64(v8Buf.Len()-v9Size)/float64(v8Buf.Len())*100)
	}
}

// TestDynamicOffset_NoBenefitOnSmooth checks that near-identity smooth data
// decodes to the same values with v9 as with v8 (k_center shift does not
// change reconstruction).
func TestDynamicOffset_NoBenefitOnSmooth(t *testing.T) {
	// Smooth data: ratios ~1.0, already optimal with default k_center.
	n := 2000
	values := make([]float64, n)
	v := 100.0
	for i := range values {
		v *= 1.0 + float64(i%3-1)*0.0001
		values[i] = v
	}

	opts := codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: 16,
		Tolerance:     math.MaxFloat64,
	}

	// v8 reference decode.
	var v8Buf bytes.Buffer
	if err := codec.Encode(values, &v8Buf, opts); err != nil {
		t.Fatalf("v8 encode: %v", err)
	}
	ref, err := codec.Decode(&v8Buf)
	if err != nil {
		t.Fatalf("v8 decode: %v", err)
	}

	// v9 must produce byte-for-byte identical decoded values to v8.
	got, _ := dynRoundTrip(t, values, opts)
	if len(got) != len(ref) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(ref))
	}
	for i := range ref {
		if got[i] != ref[i] {
			t.Errorf("index %d: v9=%v v8=%v", i, got[i], ref[i])
			if i > 5 {
				t.FailNow()
			}
		}
	}
}

// TestDynamicOffset_ParallelRoundTrip verifies that parallel v9 encoding
// produces decoded values identical to single-threaded v9 for all dataset types
// and worker counts.
func TestDynamicOffset_ParallelRoundTrip(t *testing.T) {
	datasets := []struct {
		name   string
		values []float64
	}{
		{"drift-2k", makeDriftStream(2000)},
		{"sensor-2k", makeSensorStream(2000)},
		{"financial-2k", makeFinancialWalk(2000)},
	}

	workerCounts := []int{2, 4, 8}

	opts := codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: 16,
		Tolerance:     math.MaxFloat64,
		DynamicOffset: true,
	}

	for _, ds := range datasets {
		// Single-threaded v9 reference.
		var refBuf bytes.Buffer
		if err := codec.Encode(ds.values, &refBuf, opts); err != nil {
			t.Fatalf("%s: single-threaded v9 encode: %v", ds.name, err)
		}
		ref, err := codec.Decode(&refBuf)
		if err != nil {
			t.Fatalf("%s: single-threaded v9 decode: %v", ds.name, err)
		}

		for _, n := range workerCounts {
			t.Run(ds.name+"/N="+fmt.Sprint(n), func(t *testing.T) {
				parallelOpts := opts
				parallelOpts.Parallelism = n
				var buf bytes.Buffer
				if err := codec.Encode(ds.values, &buf, parallelOpts); err != nil {
					t.Fatalf("parallel v9 encode N=%d: %v", n, err)
				}
				got, err := codec.Decode(&buf)
				if err != nil {
					t.Fatalf("parallel v9 decode N=%d: %v", n, err)
				}
				if len(got) != len(ref) {
					t.Fatalf("N=%d: length mismatch: got %d want %d", n, len(got), len(ref))
				}
				for i := range ref {
					if got[i] != ref[i] {
						t.Errorf("N=%d index %d: parallel=%v serial=%v", n, i, got[i], ref[i])
						if i > 5 {
							t.FailNow()
						}
					}
				}
			})
		}
	}
}

// TestAnalyzeDynamicOffset_Report verifies the report is non-trivial for
// drifting data and zero for data that truly can't benefit.
func TestAnalyzeDynamicOffset_Report(t *testing.T) {
	t.Run("drift-benefits", func(t *testing.T) {
		values := makeDriftStream(5000)
		opts := codec.EncodeOptions{
			EntropyMode:      codec.EntropyAdaptive,
			PrecisionBits:    16,
			Tolerance:        math.MaxFloat64,
			ReanchorInterval: codec.DefaultReanchorInterval,
		}
		rpt := codec.AnalyzeDynamicOffset(values, opts)
		if rpt.TotalSegments == 0 {
			t.Fatal("expected segments > 0")
		}
		if rpt.BenefitSegments == 0 {
			t.Errorf("expected some segments to benefit from dynamic offset on drift data; got 0/%d", rpt.TotalSegments)
		}
		if rpt.SavedBytes() <= 0 {
			t.Errorf("expected positive byte saving on drift data; got %d", rpt.SavedBytes())
		}
		t.Logf("drift: %d/%d segments benefit, saved %d B (%.1f%%)",
			rpt.BenefitSegments, rpt.TotalSegments,
			rpt.SavedBytes(), rpt.PayloadReduction()*100)
	})

	t.Run("identity-no-savings", func(t *testing.T) {
		// All values identical → all ratios ClassIdentity, no Q values → no benefit.
		values := make([]float64, 500)
		for i := range values {
			values[i] = 42.0
		}
		opts := codec.EncodeOptions{
			EntropyMode:      codec.EntropyAdaptive,
			PrecisionBits:    16,
			Tolerance:        math.MaxFloat64,
			ReanchorInterval: codec.DefaultReanchorInterval,
		}
		rpt := codec.AnalyzeDynamicOffset(values, opts)
		if rpt.BenefitSegments != 0 {
			t.Errorf("expected 0 benefit segments for identity data, got %d", rpt.BenefitSegments)
		}
		if rpt.SavedBytes() != 0 {
			t.Errorf("expected 0 saved bytes for identity data, got %d", rpt.SavedBytes())
		}
	})
}
