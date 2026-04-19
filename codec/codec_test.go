package codec_test

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	"github.com/Destraag/toroidzip/codec"
)

// roundTrip encodes then decodes values and returns the reconstructed slice.
func roundTrip(t *testing.T, values []float64, opts codec.EncodeOptions) []float64 {
	t.Helper()
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, opts); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	return got
}

// TestRoundTripSmooth verifies lossless reconstruction of a smooth series.
func TestRoundTripSmooth(t *testing.T) {
	values := make([]float64, 1000)
	v := 100.0
	for i := range values {
		v *= 1.0 + (rand.Float64()-0.5)*0.01 // ±0.5% change per step
		values[i] = v
	}
	got := roundTrip(t, values, codec.EncodeOptions{})
	if len(got) != len(values) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(values))
	}
	for i, want := range values {
		// Re-anchored values should be exact. Between anchors, reconstruction
		// is cumulative float multiply — we check within a tight tolerance.
		if math.Abs(got[i]-want) > math.Abs(want)*1e-12 {
			t.Errorf("index %d: got %v want %v (diff %e)", i, got[i], want, got[i]-want)
		}
	}
}

// TestRoundTripWithReanchor verifies reconstruction stays within tolerance
// even for long sequences with periodic re-anchoring.
func TestRoundTripWithReanchor(t *testing.T) {
	values := make([]float64, 10000)
	v := 1.0
	for i := range values {
		v *= 1.001
		values[i] = v
	}
	got := roundTrip(t, values, codec.EncodeOptions{ReanchorInterval: 64}) // aggressive re-anchor for test
	for i, want := range values {
		if math.Abs(got[i]-want) > math.Abs(want)*1e-10 {
			t.Errorf("index %d: got %v want %v", i, got[i], want)
		}
	}
}

// TestRoundTripWithZero verifies pole-zero events are handled correctly.
func TestRoundTripWithZero(t *testing.T) {
	values := []float64{1.0, 2.0, 0.0, 3.0, 4.0}
	got := roundTrip(t, values, codec.EncodeOptions{})
	if len(got) != len(values) {
		t.Fatalf("length mismatch")
	}
	for i, want := range values {
		if got[i] != want {
			t.Errorf("index %d: got %v want %v", i, got[i], want)
		}
	}
}

// TestRoundTripSingleValue verifies a single-element slice.
func TestRoundTripSingleValue(t *testing.T) {
	values := []float64{42.0}
	got := roundTrip(t, values, codec.EncodeOptions{})
	if len(got) != 1 || got[0] != 42.0 {
		t.Errorf("got %v want [42]", got)
	}
}

// TestDriftModeCompensate verifies Mode B round-trip stays very close to original.
func TestDriftModeCompensate(t *testing.T) {
	values := make([]float64, 5000)
	v := 1.0
	for i := range values {
		v *= 1.0003
		values[i] = v
	}
	opts := codec.EncodeOptions{DriftMode: codec.DriftCompensate, ReanchorInterval: 10000}
	got := roundTrip(t, values, opts)
	if len(got) != len(values) {
		t.Fatalf("length mismatch")
	}
	for i, want := range values {
		if math.Abs(got[i]-want) > math.Abs(want)*1e-12 {
			t.Errorf("index %d: got %v want %v (diff %e)", i, got[i], want, got[i]-want)
		}
	}
}

// TestDriftModeQuantize verifies Mode C round-trip is within float32 precision.
func TestDriftModeQuantize(t *testing.T) {
	values := make([]float64, 500)
	v := 100.0
	for i := range values {
		v *= 1.001
		values[i] = v
	}
	opts := codec.EncodeOptions{DriftMode: codec.DriftQuantize}
	got := roundTrip(t, values, opts)
	if len(got) != len(values) {
		t.Fatalf("length mismatch")
	}
	// Mode C is lossy: each ratio is rounded to float32 (~7 significant digits).
	// Over 500 steps the accumulated error is bounded but not zero.
	// We just check the values are finite and in a reasonable range.
	for i, want := range values {
		if math.IsNaN(got[i]) || math.IsInf(got[i], 0) {
			t.Errorf("index %d: non-finite %v", i, got[i])
		}
		if math.Abs(got[i]-want) > math.Abs(want)*1e-4 {
			t.Errorf("index %d: got %v want %v (diff %e)", i, got[i], want, got[i]-want)
		}
	}
}

// BenchmarkEncode measures encoding throughput on a smooth series.
func BenchmarkEncode(b *testing.B) {
	values := make([]float64, 100_000)
	v := 100.0
	for i := range values {
		v *= 1.001
		values[i] = v
	}
	b.ResetTimer()
	b.SetBytes(int64(len(values)) * 8)
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		_ = codec.Encode(values, &buf, codec.EncodeOptions{})
	}
}

// BenchmarkDecode measures decoding throughput on a smooth series.
func BenchmarkDecode(b *testing.B) {
	values := make([]float64, 100_000)
	v := 100.0
	for i := range values {
		v *= 1.001
		values[i] = v
	}
	var encoded bytes.Buffer
	_ = codec.Encode(values, &encoded, codec.EncodeOptions{})
	data := encoded.Bytes()

	b.ResetTimer()
	b.SetBytes(int64(len(data)))
	for i := 0; i < b.N; i++ {
		_, _ = codec.Decode(bytes.NewReader(data))
	}
}
