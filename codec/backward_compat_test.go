// backward_compat_test.go verifies that streams produced by the legacy v3, v5,
// and v6 encoders can be decoded via the current Decode() dispatcher.
// These tests exercise decode paths (decodeV3, decodeV5, decodeV6, decodeRans7,
// decodeRans7v6, readRansFreqs7, writeRansBody7, gatherRans7, gatherRans7v6)
// that are unreachable from the current encoder.
package codec_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/Destraag/toroidzip/codec"
)

// mixedSeries builds a slice that exercises ClassIdentity (ratio == 1.0 exactly),
// ClassBoundaryInf (ratio > BoundaryInfThreshold = 1e15), and ClassNormal
// in the same stream. Zero values are intentionally excluded — legacy encoders
// (v3/v5/v6) do not handle ratio=0 correctly (no ClassNormalExact guard).
func mixedSeries() []float64 {
	return []float64{
		100.0, 101.0, 102.0,
		102.0, // ClassIdentity (ratio = 1.0 exactly)
		103.0, 104.0,
		2.1e17, // ClassBoundaryInf (ratio ≈ 2.02e15 > 1e15)
		2.1e17 * 1.001, 2.1e17 * 1.002, 2.1e17 * 1.002, 2.1e17 * 1.003,
	}
}

// TestBackwardCompatV3 encodes with the legacy v3 quantized encoder and
// verifies round-trip via Decode().
func TestBackwardCompatV3(t *testing.T) {
	values := makeSmoothSeries(200, 10.0, 1.001)
	var buf bytes.Buffer
	opts := codec.EncodeOptions{
		ReanchorInterval: codec.DefaultReanchorInterval,
		EntropyMode:      codec.EntropyQuantized,
		PrecisionBits:    16,
		Tolerance:        math.MaxFloat64,
	}
	if err := codec.EncodeQuantizedLegacyForTest(values, &buf, opts); err != nil {
		t.Fatalf("encode v3: %v", err)
	}
	got, err := codec.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode v3: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// v3 uses absolute log-space symbols at 16-bit precision; ε_max(16) ≈ 6e-4.
	if e := maxRelErr(got, values); e > 1e-2 {
		t.Errorf("max relative error %e unexpectedly large for v3 round-trip", e)
	}
}

// TestBackwardCompatV3Mixed encodes mixed-class data (ClassIdentity, ClassBoundary,
// ClassNormal) with the legacy v3 encoder to cover additional decodeRans branches.
func TestBackwardCompatV3Mixed(t *testing.T) {
	values := mixedSeries()
	var buf bytes.Buffer
	opts := codec.EncodeOptions{
		ReanchorInterval: codec.DefaultReanchorInterval,
		EntropyMode:      codec.EntropyQuantized,
		PrecisionBits:    16,
		Tolerance:        math.MaxFloat64,
	}
	if err := codec.EncodeQuantizedLegacyForTest(values, &buf, opts); err != nil {
		t.Fatalf("encode v3 mixed: %v", err)
	}
	got, err := codec.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode v3 mixed: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// Boundary events (0.0) are stored verbatim; ClassIdentity is lossless.
	// Non-zero values may have quantization error ≤ ε_max(16).
	for i, v := range values {
		if e := math.Abs(got[i]-v) / math.Abs(v); e > 0.1 {
			t.Errorf("value[%d]: rel err %e > 0.1", i, e)
		}
	}
}

// TestBackwardCompatV5 encodes with the legacy v5 adaptive encoder (u16/u32/f64
// tiers, absolute uint16 payload) and verifies round-trip via Decode().
func TestBackwardCompatV5(t *testing.T) {
	values := makeSmoothSeries(300, 50.0, 1.0005)
	var buf bytes.Buffer
	opts := codec.EncodeOptions{
		ReanchorInterval: codec.DefaultReanchorInterval,
		PrecisionBits:    16,
		Tolerance:        math.MaxFloat64,
	}
	if err := codec.EncodeAdaptiveV5ForTest(values, &buf, opts); err != nil {
		t.Fatalf("encode v5: %v", err)
	}
	got, err := codec.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode v5: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// v5 at 16-bit precision: ε_max(16) ≈ 6e-4; (1+ε)^256 − 1 ≈ 17% max accumulated
	// drift before the periodic reanchor resets at step 256.
	if e := maxRelErr(got, values); e > 0.25 {
		t.Errorf("max relative error %e unexpectedly large for v5 round-trip", e)
	}
}

// TestBackwardCompatV5Mixed encodes mixed-class data with the legacy v5 encoder
// to exercise ClassIdentity, ClassBoundaryZero/Inf, and ClassNormal32 branches
// in gatherRans7 and decodeRans7.
func TestBackwardCompatV5Mixed(t *testing.T) {
	values := mixedSeries()
	var buf bytes.Buffer
	// Tolerance=1e-5 (below delta16 ≈ 6e-4) forces per-ratio checks.
	// Ratios with relErr(u16) >= 1e-5 fall to ClassNormal32 (30-bit).
	opts := codec.EncodeOptions{
		ReanchorInterval: codec.DefaultReanchorInterval,
		PrecisionBits:    16,
		Tolerance:        1e-5,
	}
	if err := codec.EncodeAdaptiveV5ForTest(values, &buf, opts); err != nil {
		t.Fatalf("encode v5 mixed: %v", err)
	}
	got, err := codec.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode v5 mixed: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	for i, v := range values {
		if e := math.Abs(got[i]-v) / math.Abs(v); e > 0.1 {
			t.Errorf("value[%d]: rel err %e > 0.1", i, e)
		}
	}
}

// TestBackwardCompatV5
// signed-offset, u16/u32/f64 tiers) and verifies round-trip via Decode().
func TestBackwardCompatV6(t *testing.T) {
	values := makeSmoothSeries(300, 50.0, 1.0005)
	var buf bytes.Buffer
	opts := codec.EncodeOptions{
		ReanchorInterval: codec.DefaultReanchorInterval,
		PrecisionBits:    30,
		Tolerance:        math.MaxFloat64,
	}
	if err := codec.EncodeAdaptiveV6ForTest(values, &buf, opts); err != nil {
		t.Fatalf("encode v6: %v", err)
	}
	got, err := codec.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode v6: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// v6 at 30-bit precision: ε_max(30) ≈ 1.9e-9; very tight.
	if e := maxRelErr(got, values); e > 1e-6 {
		t.Errorf("max relative error %e unexpectedly large for v6 round-trip", e)
	}
}

// TestBackwardCompatV6Mixed encodes mixed-class data with the legacy v6 encoder
// to exercise ClassIdentity, ClassBoundaryZero/Inf, and ClassNormal32 branches
// in gatherRans7v6 and decodeRans7v6.
func TestBackwardCompatV6Mixed(t *testing.T) {
	values := mixedSeries()
	var buf bytes.Buffer
	opts := codec.EncodeOptions{
		ReanchorInterval: codec.DefaultReanchorInterval,
		PrecisionBits:    30,
		Tolerance:        math.MaxFloat64,
	}
	if err := codec.EncodeAdaptiveV6ForTest(values, &buf, opts); err != nil {
		t.Fatalf("encode v6 mixed: %v", err)
	}
	got, err := codec.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode v6 mixed: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	for i, v := range values {
		if e := math.Abs(got[i]-v) / math.Abs(v); e > 1e-6 {
			t.Errorf("value[%d]: rel err %e > 1e-6", i, e)
		}
	}
}
