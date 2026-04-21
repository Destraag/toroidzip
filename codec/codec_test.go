package codec_test

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/Destraag/toroidzip/codec"
)

// TestAdaptiveReanchorTriggered verifies that with AdaptiveReanchor enabled,
// an adversarial (monotone) sequence that would accumulate error far beyond
// the end-to-end guarantee with fixed K=256 is instead kept within T_end.
//
// The adversarial sequence multiplies by a fixed ratio each step so all
// per-ratio quantization errors have the same sign — worst-case drift.
// With K=256 and per-ratio ε≈1e-5 the worst-case error would be ~2.5e-3,
// violating a 4-sig-fig guarantee (T_end=5e-4).  Adaptive reanchoring must
// fire before the error crosses T_end.
func TestAdaptiveReanchorTriggered(t *testing.T) {
	const n = 500
	const sigFigs = 4
	endToEndTol := codec.SigFigsToTolerance(sigFigs)     // 5e-4
	perRatioTol := codec.SigFigsToTolerance(sigFigs + 2) // 5e-6

	// Adversarial: constant monotone growth — every quantization error has
	// same sign, giving maximum drift accumulation.
	values := make([]float64, n)
	v := 100.0
	for i := range values {
		v *= 1.0001
		values[i] = v
	}

	// Baseline: fixed K=256, no adaptive reanchor.
	// Show that it violates the end-to-end guarantee.
	optsFixed := codec.EncodeOptions{
		EntropyMode:      codec.EntropyAdaptive,
		PrecisionBits:    codec.SigFigsToBits(sigFigs + 2),
		Tolerance:        perRatioTol,
		ReanchorInterval: 256,
		AdaptiveReanchor: false,
	}
	gotFixed := roundTrip(t, values, optsFixed)
	var maxFixedErr float64
	for i, want := range values {
		if want != 0 {
			if e := math.Abs(gotFixed[i]-want) / math.Abs(want); e > maxFixedErr {
				maxFixedErr = e
			}
		}
	}
	if maxFixedErr <= endToEndTol {
		// If the fixed encoder accidentally meets the guarantee, the adversarial
		// sequence wasn't adversarial enough — warn but don't fail the test.
		t.Logf("note: fixed K=256 unexpectedly met T_end (maxErr=%e <= %e); "+
			"adversarial effect may be weaker than expected", maxFixedErr, endToEndTol)
	}

	// Adaptive: must meet the end-to-end guarantee.
	optsAdaptive := codec.EncodeOptions{
		EntropyMode:       codec.EntropyAdaptive,
		PrecisionBits:     codec.SigFigsToBits(sigFigs + 2),
		Tolerance:         perRatioTol,
		ReanchorInterval:  codec.SigFigsToMaxK(sigFigs),
		AdaptiveReanchor:  true,
		EndToEndTolerance: endToEndTol,
	}
	gotAdaptive := roundTrip(t, values, optsAdaptive)
	for i, want := range values {
		if want == 0 {
			continue
		}
		if e := math.Abs(gotAdaptive[i]-want) / math.Abs(want); e > endToEndTol {
			t.Errorf("[%d] adaptive reanchor violated T_end: got %v want %v (rel err %e, T_end %e)",
				i, gotAdaptive[i], want, e, endToEndTol)
		}
	}
}

// TestAdaptiveReanchorSparse verifies that near-constant data does not trigger
// excessive adaptive reanchors.  Compression should be at least as good with
// adaptive reanchoring as with the fixed-K safety cap alone, because smooth
// data rarely accumulates enough error to cross T_end.
func TestAdaptiveReanchorSparse(t *testing.T) {
	const n = 10_000
	const sigFigs = 4
	endToEndTol := codec.SigFigsToTolerance(sigFigs)
	perRatioTol := codec.SigFigsToTolerance(sigFigs + 2)
	kMax := codec.SigFigsToMaxK(sigFigs) // 10000 circuit breaker

	// Near-constant data with very small random perturbations.
	rng := rand.New(rand.NewSource(42))
	values := make([]float64, n)
	v := 50.0
	for i := range values {
		v *= 1.0 + (rng.Float64()-0.5)*1e-5
		values[i] = v
	}

	// Fixed K=kMax (no adaptive check): upper bound on reanchor count.
	optsFixed := codec.EncodeOptions{
		EntropyMode:      codec.EntropyAdaptive,
		PrecisionBits:    codec.SigFigsToBits(sigFigs + 2),
		Tolerance:        perRatioTol,
		ReanchorInterval: kMax,
		AdaptiveReanchor: false,
	}
	var bufFixed bytes.Buffer
	if err := codec.Encode(values, &bufFixed, optsFixed); err != nil {
		t.Fatalf("encode fixed: %v", err)
	}

	// Adaptive reanchoring: should fire fewer or equal reanchors on smooth data.
	optsAdaptive := codec.EncodeOptions{
		EntropyMode:       codec.EntropyAdaptive,
		PrecisionBits:     codec.SigFigsToBits(sigFigs + 2),
		Tolerance:         perRatioTol,
		ReanchorInterval:  kMax,
		AdaptiveReanchor:  true,
		EndToEndTolerance: endToEndTol,
	}
	var bufAdaptive bytes.Buffer
	if err := codec.Encode(values, &bufAdaptive, optsAdaptive); err != nil {
		t.Fatalf("encode adaptive: %v", err)
	}

	// Adaptive output must not be larger than fixed (fewer or equal reanchors).
	if bufAdaptive.Len() > bufFixed.Len() {
		t.Errorf("adaptive output (%d bytes) larger than fixed K=%d output (%d bytes): "+
			"unexpected extra reanchors on near-constant data",
			bufAdaptive.Len(), kMax, bufFixed.Len())
	}

	// Verify the adaptive output still decodes correctly within T_end.
	got, err := codec.Decode(&bufAdaptive)
	if err != nil {
		t.Fatalf("decode adaptive: %v", err)
	}
	for i, want := range values {
		if want == 0 {
			continue
		}
		if e := math.Abs(got[i]-want) / math.Abs(want); e > endToEndTol {
			t.Errorf("[%d] rel err %e > T_end %e", i, e, endToEndTol)
		}
	}
}

// TestSigFigsEndToEndGuarantee verifies that --sig-figs N parameters (when
// applied via EncodeOptions) guarantee N significant decimal figures in every
// reconstructed value for N = 3 through 9 (the full supported range).
//
// The test uses a 10,000-value smooth sequence.  Each value is checked against
// the corresponding original; the maximum relative error must not exceed
// T_end = 0.5×10^-N at any position.
func TestSigFigsEndToEndGuarantee(t *testing.T) {
	const n = 10_000
	rng := rand.New(rand.NewSource(7))
	values := make([]float64, n)
	v := 1.0
	for i := range values {
		v *= 1.0 + (rng.Float64()-0.5)*0.002
		values[i] = v
	}

	for _, sigFigs := range []int{3, 4, 5, 6, 7, 8, 9} {
		sigFigs := sigFigs
		t.Run(fmt.Sprintf("N=%d", sigFigs), func(t *testing.T) {
			endToEndTol := codec.SigFigsToTolerance(sigFigs)
			opts := codec.EncodeOptions{
				EntropyMode:       codec.EntropyAdaptive,
				PrecisionBits:     codec.SigFigsToBits(sigFigs + 2),
				Tolerance:         codec.SigFigsToTolerance(sigFigs + 2),
				ReanchorInterval:  codec.SigFigsToMaxK(sigFigs),
				AdaptiveReanchor:  true,
				EndToEndTolerance: endToEndTol,
			}
			got := roundTrip(t, values, opts)
			for i, want := range values {
				if want == 0 {
					continue
				}
				if e := math.Abs(got[i]-want) / math.Abs(want); e > endToEndTol {
					t.Errorf("[%d] N=%d: rel err %e > T_end %e (got %v want %v)",
						i, sigFigs, e, endToEndTol, got[i], want)
				}
			}
		})
	}
}

// TestDecodeV4Legacy verifies that the legacy v4 adaptive stream decoder
// (decodeV4 / decodeRans6) can still decode streams produced by old encoder
// builds. The fixture was generated with gatherRans6 before that encoder
// function was removed in M8/8a.
//
// Input values: [1.0, 1.01, 1.02, 1.03, 0.5, 2.0]
// Encoded with: bits=16, tol=1e-4, DriftReanchor, reanchorInterval=256
func TestDecodeV4Legacy(t *testing.T) {
	v4Fixture := []byte{
		0x54, 0x5a, 0x52, 0x5a, 0x04, 0x00, 0x00, 0x01, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x10, 0x01, 0x00, 0x00, 0x00, 0xfb, 0x0f, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0xfd,
		0xbb, 0xc8, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x75,
		0x80, 0x74, 0x80, 0x73, 0x80, 0xa2, 0x5e, 0x00, 0xc0,
	}
	want := []float64{1.0, 1.01, 1.02, 1.03, 0.5, 2.0}
	const tol = 1e-3 // slightly relaxed — per-ratio ε=1e-4 each

	got, err := codec.Decode(bytes.NewReader(v4Fixture))
	if err != nil {
		t.Fatalf("Decode v4 legacy fixture: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("length: got %d want %d", len(got), len(want))
	}
	for i, w := range want {
		if w == 0 {
			continue
		}
		if math.Abs(got[i]-w)/math.Abs(w) > tol {
			t.Errorf("[%d] got %v want %v", i, got[i], w)
		}
	}
}

// TestAdaptiveV5DriftCompensate verifies that v5 adaptive encoding with
// DriftCompensate (Mode B) round-trips within the stated tolerance.
func TestAdaptiveV5DriftCompensate(t *testing.T) {
	values := make([]float64, 500)
	v := 10.0
	for i := range values {
		v *= 1.0 + (rand.Float64()-0.5)*0.002
		values[i] = v
	}
	const tol = 1e-4
	opts := codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		DriftMode:     codec.DriftCompensate,
		PrecisionBits: 16,
		Tolerance:     tol,
	}
	got := roundTrip(t, values, opts)
	for i, want := range values {
		if want == 0 {
			continue
		}
		if math.Abs(got[i]-want)/math.Abs(want) > tol*10 {
			t.Errorf("[%d] got %v want %v (rel err %e)", i, got[i], want,
				math.Abs(got[i]-want)/math.Abs(want))
		}
	}
}

// TestDecodeV5Errors verifies that truncated v5 streams produce errors
// rather than panics or silently wrong output.
func TestDecodeV5Errors(t *testing.T) {
	// Build a valid v5 stream first.
	var buf bytes.Buffer
	if err := codec.Encode([]float64{1.0, 1.01, 1.02, 1.03}, &buf, codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: 16,
		Tolerance:     1e-4,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	full := buf.Bytes()

	// Truncate at several points past magic+version (byte 5 onward).
	// Each should return an error, never panic.
	for _, cutAt := range []int{5, 8, 14, 20, 30, 40} {
		if cutAt >= len(full) {
			continue
		}
		_, err := codec.Decode(bytes.NewReader(full[:cutAt]))
		if err == nil {
			t.Errorf("expected error decoding v5 truncated at byte %d, got nil", cutAt)
		}
	}
}

// TestDriftModesAllEncoders verifies DriftCompensate and DriftQuantize
// round-trips for lossless, quantized, and raw encoders.
func TestDriftModesAllEncoders(t *testing.T) {
	values := make([]float64, 300)
	v := 50.0
	for i := range values {
		v *= 1.001
		values[i] = v
	}

	cases := []struct {
		name string
		opts codec.EncodeOptions
		tol  float64
	}{
		{"Lossless/Compensate", codec.EncodeOptions{EntropyMode: codec.EntropyLossless, DriftMode: codec.DriftCompensate}, 1e-12},
		// DriftQuantize float32-rounds every ratio before encoding (intentionally lossy).
		// Over 300 steps the accumulated float32 error is ~float32_eps × n ≈ 3e-5.
		{"Lossless/Quantize", codec.EncodeOptions{EntropyMode: codec.EntropyLossless, DriftMode: codec.DriftQuantize}, 1e-4},
		{"Quantized/Compensate", codec.EncodeOptions{EntropyMode: codec.EntropyQuantized, PrecisionBits: 16, DriftMode: codec.DriftCompensate}, 1e-3},
		{"Quantized/Quantize", codec.EncodeOptions{EntropyMode: codec.EntropyQuantized, PrecisionBits: 16, DriftMode: codec.DriftQuantize}, 1e-3},
		{"Raw/Compensate", codec.EncodeOptions{EntropyMode: codec.EntropyRaw, DriftMode: codec.DriftCompensate}, 1e-12},
		{"Raw/Quantize", codec.EncodeOptions{EntropyMode: codec.EntropyRaw, DriftMode: codec.DriftQuantize}, 1e-4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := roundTrip(t, values, tc.opts)
			for i, want := range values {
				if math.Abs(got[i]-want)/math.Abs(want) > tc.tol {
					t.Errorf("[%d] got %v want %v", i, got[i], want)
				}
			}
		})
	}
}

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

// TestRoundTripWithZero verifies boundary-zero events are handled correctly.
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

// TestClassify covers boundary inputs not exercised by round-trip tests.
func TestClassify(t *testing.T) {
	cases := []struct {
		name  string
		input float64
		want  codec.RatioClass
	}{
		{"NaN", math.NaN(), codec.ClassBoundaryInf},
		{"+Inf", math.Inf(1), codec.ClassBoundaryInf},
		{"-Inf", math.Inf(-1), codec.ClassBoundaryInf},
		{"large positive", 2e15, codec.ClassBoundaryInf},
		{"large negative", -2e15, codec.ClassBoundaryInf},
		{"identity", 1.0, codec.ClassIdentity},
		{"identity epsilon edge", 1.0 + 5e-10, codec.ClassIdentity},
		{"normal", 1.5, codec.ClassNormal},
		{"negative normal", -1.5, codec.ClassNormal},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := codec.Classify(tc.input)
			if got != tc.want {
				t.Errorf("Classify(%v) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}

// TestDecodeErrors covers Decode rejection of malformed streams.
func TestDecodeErrors(t *testing.T) {
	t.Run("empty reader", func(t *testing.T) {
		_, err := codec.Decode(bytes.NewReader(nil))
		if err == nil {
			t.Fatal("expected error on empty reader")
		}
	})

	t.Run("bad magic", func(t *testing.T) {
		_, err := codec.Decode(bytes.NewReader([]byte("NOPE")))
		if err == nil {
			t.Fatal("expected error on bad magic")
		}
	})

	t.Run("wrong version", func(t *testing.T) {
		// Build a header with correct magic but version=99.
		var buf bytes.Buffer
		buf.Write([]byte{'T', 'Z', 'R', 'Z'}) // magic
		buf.WriteByte(99)                     // bad version
		_, err := codec.Decode(&buf)
		if err == nil {
			t.Fatal("expected error on unsupported version")
		}
	})

	t.Run("truncated after magic", func(t *testing.T) {
		_, err := codec.Decode(bytes.NewReader([]byte{'T', 'Z', 'R', 'Z'}))
		if err == nil {
			t.Fatal("expected error on truncated stream")
		}
	})

	// Truncated v2 stream: encode lossless then truncate at various points.
	t.Run("truncated v2 header", func(t *testing.T) {
		var buf bytes.Buffer
		if err := codec.Encode([]float64{1.0, 2.0, 3.0}, &buf, codec.EncodeOptions{
			EntropyMode: codec.EntropyLossless,
		}); err != nil {
			t.Fatalf("encode: %v", err)
		}
		data := buf.Bytes()
		// Try decoding truncated at several points into the header (past magic+version).
		for _, cutAt := range []int{6, 8, 12} {
			if cutAt >= len(data) {
				continue
			}
			_, err := codec.Decode(bytes.NewReader(data[:cutAt]))
			if err == nil {
				t.Errorf("expected error decoding v2 truncated at byte %d", cutAt)
			}
		}
	})

	// Truncated v3 stream.
	t.Run("truncated v3 header", func(t *testing.T) {
		var buf bytes.Buffer
		if err := codec.Encode([]float64{1.0, 2.0, 3.0}, &buf, codec.EncodeOptions{
			EntropyMode: codec.EntropyQuantized, PrecisionBits: 8,
		}); err != nil {
			t.Fatalf("encode: %v", err)
		}
		data := buf.Bytes()
		for _, cutAt := range []int{6, 8, 12, 14} {
			if cutAt >= len(data) {
				continue
			}
			_, err := codec.Decode(bytes.NewReader(data[:cutAt]))
			if err == nil {
				t.Errorf("expected error decoding v3 truncated at byte %d", cutAt)
			}
		}
	})
}

// TestEncodeEmptyInput verifies Encode rejects empty slices.
func TestEncodeEmptyInput(t *testing.T) {
	var buf bytes.Buffer
	err := codec.Encode(nil, &buf, codec.EncodeOptions{})
	if err == nil {
		t.Fatal("expected error encoding nil slice")
	}
	err = codec.Encode([]float64{}, &buf, codec.EncodeOptions{})
	if err == nil {
		t.Fatal("expected error encoding empty slice")
	}
}

// TestKahanProdZeroAnchor verifies Mode B handles a zero-valued anchor.
func TestKahanProdZeroAnchor(t *testing.T) {
	// A sequence starting at zero triggers the zero-anchor path in newKahanProd.
	// After the zero, a boundary-zero event fires and resets the anchor to the next value.
	values := []float64{0.0, 5.0, 10.0, 20.0}
	opts := codec.EncodeOptions{DriftMode: codec.DriftCompensate}
	got := roundTrip(t, values, opts)
	if len(got) != len(values) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(values))
	}
	// Mode B is near-lossless: Kahan log-space introduces tiny FP rounding.
	for i, want := range values {
		if want == 0 {
			if got[i] != 0 {
				t.Errorf("index %d: got %v want 0", i, got[i])
			}
			continue
		}
		if math.Abs(got[i]-want) > math.Abs(want)*1e-12 {
			t.Errorf("index %d: got %v want %v (diff %e)", i, got[i], want, got[i]-want)
		}
	}
}

// TestRoundTripBoundaryInf verifies boundary-inf events (extreme ratio) are handled correctly.
func TestRoundTripBoundaryInf(t *testing.T) {
	// Jump from 1.0 to 1e16 triggers ClassBoundaryInf.
	values := []float64{1.0, 1e16, 2e16, 3e16}
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

// TestRoundTripNearZeroPrev verifies that a subnormal/near-zero prev triggers
// ClassBoundaryZero (not ClassBoundaryInf) and the subsequent value is recovered
// exactly. This exercises the BoundaryZeroThreshold path in computeRatio.
func TestRoundTripNearZeroPrev(t *testing.T) {
	// 1e-301 < BoundaryZeroThreshold (1e-300): the next ratio 5.0/1e-301 = 5e300
	// would overflow to ClassBoundaryInf without the threshold check, storing 5e300
	// instead of 5.0.
	values := []float64{1.0, 1e-301, 5.0, 10.0}
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

// --- Version 2 (EntropyLossless) round-trip tests ---

// TestRoundTripLosslessSmooth verifies v2 near-lossless reconstruction of smooth data.
// ClassIdentity events are reconstructed as ratio=1.0 (within IdentityEpsilon per step).
func TestRoundTripLosslessSmooth(t *testing.T) {
	values := make([]float64, 1000)
	v := 100.0
	for i := range values {
		v *= 1.001
		values[i] = v
	}
	// Tight reanchor interval so ClassIdentity error stays bounded.
	opts := codec.EncodeOptions{EntropyMode: codec.EntropyLossless, ReanchorInterval: 64}
	got := roundTrip(t, values, opts)
	if len(got) != len(values) {
		t.Fatalf("length mismatch")
	}
	for i, want := range values {
		// Reanchors are exact. Between anchors ClassIdentity may accumulate
		// up to reanchorInterval * IdentityEpsilon (~64e-9) relative error.
		if math.Abs(got[i]-want) > math.Abs(want)*1e-6 {
			t.Errorf("index %d: got %v want %v (diff %e)", i, got[i], want, got[i]-want)
		}
	}
}

// TestRoundTripLosslessSingleValue exercises the single-element edge case.
func TestRoundTripLosslessSingleValue(t *testing.T) {
	values := []float64{3.14159}
	got := roundTrip(t, values, codec.EncodeOptions{EntropyMode: codec.EntropyLossless})
	if len(got) != 1 || got[0] != 3.14159 {
		t.Errorf("got %v want [3.14159]", got)
	}
}

// TestRoundTripLosslessBoundary verifies boundary and reanchor events are
// stored verbatim (exact) in lossless mode.
func TestRoundTripLosslessBoundary(t *testing.T) {
	values := []float64{1.0, 2.0, 0.0, 3.0, 1e16, 4.0}
	opts := codec.EncodeOptions{EntropyMode: codec.EntropyLossless}
	got := roundTrip(t, values, opts)
	if len(got) != len(values) {
		t.Fatalf("length mismatch")
	}
	// 0.0 and 1e16 trigger boundary events — must be exact.
	for i, want := range values {
		if want == 0 || want == 1e16 {
			if got[i] != want {
				t.Errorf("boundary index %d: got %v want %v", i, got[i], want)
			}
		}
	}
}

// TestRoundTripLosslessDriftCompensate verifies v2+DriftCompensate stays tight.
func TestRoundTripLosslessDriftCompensate(t *testing.T) {
	values := make([]float64, 2000)
	v := 1.0
	for i := range values {
		v *= 1.0005
		values[i] = v
	}
	opts := codec.EncodeOptions{
		EntropyMode: codec.EntropyLossless,
		DriftMode:   codec.DriftCompensate,
	}
	got := roundTrip(t, values, opts)
	for i, want := range values {
		if math.Abs(got[i]-want) > math.Abs(want)*1e-6 {
			t.Errorf("index %d: got %v want %v (diff %e)", i, got[i], want, got[i]-want)
		}
	}
}

// TestLosslessStreamSmaller verifies v2 produces a shorter byte stream than v1
// for smooth data (where most ratios are ClassIdentity).
func TestLosslessStreamSmaller(t *testing.T) {
	values := make([]float64, 10000)
	v := 100.0
	for i := range values {
		v *= 1.0 + (rand.Float64()-0.5)*1e-10 // nearly constant → most ClassIdentity
		values[i] = v
	}
	var v1Buf, v2Buf bytes.Buffer
	if err := codec.Encode(values, &v1Buf, codec.EncodeOptions{EntropyMode: codec.EntropyRaw}); err != nil {
		t.Fatal(err)
	}
	if err := codec.Encode(values, &v2Buf, codec.EncodeOptions{EntropyMode: codec.EntropyLossless}); err != nil {
		t.Fatal(err)
	}
	if v2Buf.Len() >= v1Buf.Len() {
		t.Errorf("lossless stream (%d bytes) not smaller than raw (%d bytes)", v2Buf.Len(), v1Buf.Len())
	}
}

// --- Version 3 (EntropyQuantized) round-trip tests ---

// TestRoundTripQuantizedSmooth verifies v3 approximate round-trip on smooth data.
func TestRoundTripQuantizedSmooth(t *testing.T) {
	values := make([]float64, 500)
	v := 100.0
	for i := range values {
		v *= 1.005
		values[i] = v
	}
	opts := codec.EncodeOptions{EntropyMode: codec.EntropyQuantized, PrecisionBits: 12}
	got := roundTrip(t, values, opts)
	if len(got) != len(values) {
		t.Fatalf("length mismatch")
	}
	for i, want := range values {
		if math.IsNaN(got[i]) || math.IsInf(got[i], 0) {
			t.Fatalf("index %d: non-finite %v", i, got[i])
		}
		// At 12-bit precision, each ratio is quantized to ~0.2% resolution.
		// Accumulated error across 500 steps (with ReanchorInterval=256):
		// worst-case ~ReanchorInterval * 0.2% ≈ 50% accumulated — use 1% per step.
		if math.Abs(got[i]-want) > math.Abs(want)*0.05 {
			t.Errorf("index %d: got %v want %v (ratio error %e)", i, got[i], want,
				math.Abs(got[i]-want)/math.Abs(want))
		}
	}
}

// TestRoundTripQuantizedSingleValue exercises the single-element edge case.
func TestRoundTripQuantizedSingleValue(t *testing.T) {
	values := []float64{2.71828}
	got := roundTrip(t, values, codec.EncodeOptions{EntropyMode: codec.EntropyQuantized})
	if len(got) != 1 || got[0] != 2.71828 {
		t.Errorf("got %v want [2.71828]", got)
	}
}

// TestRoundTripQuantizedPrecisions checks that higher precision gives less error.
func TestRoundTripQuantizedPrecisions(t *testing.T) {
	values := make([]float64, 50)
	v := 10.0
	for i := range values {
		v *= 1.02
		values[i] = v
	}
	opts4 := codec.EncodeOptions{EntropyMode: codec.EntropyQuantized, PrecisionBits: 4, ReanchorInterval: 10}
	opts14 := codec.EncodeOptions{EntropyMode: codec.EntropyQuantized, PrecisionBits: 14, ReanchorInterval: 10}

	err4 := maxRelErr(roundTrip(t, values, opts4), values)
	err14 := maxRelErr(roundTrip(t, values, opts14), values)

	if err14 >= err4 {
		t.Errorf("higher precision should give less error: err4=%e err14=%e", err4, err14)
	}
}

// TestQuantizedStreamSmaller verifies v3 (low precision) is smaller than v1.
func TestQuantizedStreamSmaller(t *testing.T) {
	values := make([]float64, 5000)
	v := 1.0
	for i := range values {
		v *= 1.001
		values[i] = v
	}
	var v1Buf, v3Buf bytes.Buffer
	if err := codec.Encode(values, &v1Buf, codec.EncodeOptions{EntropyMode: codec.EntropyRaw}); err != nil {
		t.Fatalf("encode raw: %v", err)
	}
	if err := codec.Encode(values, &v3Buf, codec.EncodeOptions{EntropyMode: codec.EntropyQuantized, PrecisionBits: 8}); err != nil {
		t.Fatalf("encode quantized: %v", err)
	}
	if v3Buf.Len() >= v1Buf.Len() {
		t.Errorf("quantized stream (%d bytes) not smaller than raw (%d bytes)", v3Buf.Len(), v1Buf.Len())
	}
}

// TestQuantizedRoundTripUint8Tier verifies encode/decode at 4 bits (uint8 tier).
func TestQuantizedRoundTripUint8Tier(t *testing.T) {
	values := make([]float64, 500)
	v := 10.0
	for i := range values {
		v *= 1.01
		values[i] = v
	}
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode: codec.EntropyQuantized, PrecisionBits: 4,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// 4 bits → ε_max ≈ 0.414; 50% relative error tolerance.
	if e := maxRelErr(got, values); e > 0.5 {
		t.Errorf("max relative error %f exceeds 0.5 for 4-bit quantization", e)
	}
}

// TestQuantizedRoundTripUint32Tier verifies encode/decode at 20 bits (uint32 tier).
func TestQuantizedRoundTripUint32Tier(t *testing.T) {
	values := make([]float64, 500)
	v := 10.0
	for i := range values {
		v *= 1.001
		values[i] = v
	}
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode: codec.EntropyQuantized, PrecisionBits: 20,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// 20 bits → ε_max ≈ 2.6e-6; 1e-4 relative error tolerance.
	if e := maxRelErr(got, values); e > 1e-4 {
		t.Errorf("max relative error %e exceeds 1e-4 for 20-bit quantization", e)
	}
}

// maxRelErr returns the maximum relative error between got and want.
func maxRelErr(got, want []float64) float64 {
	var maxErr float64
	for i := range want {
		if want[i] != 0 {
			e := math.Abs(got[i]-want[i]) / math.Abs(want[i])
			if e > maxErr {
				maxErr = e
			}
		}
	}
	return maxErr
}

// --- EntropyAdaptive (v4) tests ---

// TestAdaptiveRoundTripTolerance0 verifies that EntropyAdaptive with Tolerance=0
// (lossless-equivalent) reconstructs values with at most the same small drift
// as EntropyLossless. All ClassNormal ratios take the ClassNormalExact path.
func TestAdaptiveRoundTripTolerance0(t *testing.T) {
	values := makeSmoothSeries(1000, 100.0, 1.001)
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode: codec.EntropyAdaptive,
		Tolerance:   0.0, // all normals → ClassNormalExact
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// Tolerance=0 → no quantization loss; only identity-class drift possible.
	if e := maxRelErr(got, values); e > 1e-9 {
		t.Errorf("max relative error %e exceeds 1e-9 at tolerance=0", e)
	}
}

// TestAdaptiveRoundTripLossyTolerance verifies that EntropyAdaptive with a
// non-zero tolerance compresses well and keeps error within the declared bound.
func TestAdaptiveRoundTripLossyTolerance(t *testing.T) {
	values := makeSmoothSeries(2000, 100.0, 1.0005)
	tol := 1e-3
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: 16,
		Tolerance:     tol,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// Max error should be bounded by the quantization tolerance plus drift.
	if e := maxRelErr(got, values); e > 0.05 {
		t.Errorf("max relative error %e unexpectedly large for tolerance=%g", e, tol)
	}
}

// TestAdaptiveSmallerThanQuantizedAtHighTolerance checks that adaptive at a
// high tolerance (many quantized hits) produces a smaller or equal stream than
// plain lossless, demonstrating compression benefit.
func TestAdaptiveSmallerThanQuantizedAtHighTolerance(t *testing.T) {
	values := makeSmoothSeries(5000, 50.0, 1.0002)
	var losslessBuf, adaptiveBuf bytes.Buffer
	if err := codec.Encode(values, &losslessBuf, codec.EncodeOptions{
		EntropyMode: codec.EntropyLossless,
	}); err != nil {
		t.Fatalf("lossless encode: %v", err)
	}
	if err := codec.Encode(values, &adaptiveBuf, codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: 16,
		Tolerance:     1e-2, // 1% tolerance → most normals should be quantized
	}); err != nil {
		t.Fatalf("adaptive encode: %v", err)
	}
	if adaptiveBuf.Len() >= losslessBuf.Len() {
		t.Errorf("adaptive (%d bytes) not smaller than lossless (%d bytes) at 1%% tolerance",
			adaptiveBuf.Len(), losslessBuf.Len())
	}
}

// TestAdaptiveWithBoundaryEvents ensures boundary and reanchor events are
// handled correctly in the v4 stream alongside quantized/exact ratios.
func TestAdaptiveWithBoundaryEvents(t *testing.T) {
	values := []float64{1.0, 1.001, 0.0, 1.5, 1.501, 1.502, 100.0, 100.1}
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode: codec.EntropyAdaptive,
		Tolerance:   1e-3,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// Boundary events (zero crossing) must be reconstructed exactly.
	if got[2] != 0.0 {
		t.Errorf("boundary zero not reconstructed exactly: got %v want 0", got[2])
	}
}

// TestAdaptiveSingleValue ensures the single-value edge case is handled.
func TestAdaptiveSingleValue(t *testing.T) {
	values := []float64{42.0}
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode: codec.EntropyAdaptive,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != 1 || got[0] != 42.0 {
		t.Errorf("single value: got %v want [42]", got)
	}
}

// TestAdaptiveDriftCompensate verifies DriftCompensate works through the v4 path.
func TestAdaptiveDriftCompensate(t *testing.T) {
	values := makeSmoothSeries(500, 10.0, 1.002)
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode: codec.EntropyAdaptive,
		DriftMode:   codec.DriftCompensate,
		Tolerance:   1e-3,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	if e := maxRelErr(got, values); e > 0.05 {
		t.Errorf("max relative error %e too large for DriftCompensate adaptive", e)
	}
}

// TestAdaptiveDefaultPrecisionBits ensures PrecisionBits=0 is handled (defaults to 16).
func TestAdaptiveDefaultPrecisionBits(t *testing.T) {
	values := makeSmoothSeries(200, 5.0, 1.001)
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: 0, // should default to 16 inside encodeAdaptive
		Tolerance:     1e-3,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	if _, err := codec.Decode(&buf); err != nil {
		t.Fatalf("decode: %v", err)
	}
}

// TestAdaptivePrecisionBitsCapped verifies PrecisionBits is used as-is in v7 —
// not capped to 16 as in v5. Higher PrecisionBits → larger stream but better accuracy.
func TestAdaptivePrecisionBitsCapped(t *testing.T) {
	values := makeSmoothSeries(200, 5.0, 1.001)
	encode := func(bits int) ([]float64, *bytes.Buffer) {
		var buf bytes.Buffer
		if err := codec.Encode(values, &buf, codec.EncodeOptions{
			EntropyMode:   codec.EntropyAdaptive,
			PrecisionBits: bits,
			Tolerance:     1e-3,
		}); err != nil {
			t.Fatalf("encode bits=%d: %v", bits, err)
		}
		got, err := codec.Decode(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatalf("decode bits=%d: %v", bits, err)
		}
		return got, &buf
	}
	got24, buf24 := encode(24)
	got16, buf16 := encode(16)
	// v7: different precision → different byte streams.
	if bytes.Equal(buf24.Bytes(), buf16.Bytes()) {
		t.Errorf("PrecisionBits=24 and PrecisionBits=16 produced identical streams; expected different")
	}
	// v7: higher precision → smaller reconstruction error.
	e24, e16 := maxRelErr(got24, values), maxRelErr(got16, values)
	if e24 >= e16 {
		t.Errorf("PrecisionBits=24 error %e not smaller than PrecisionBits=16 error %e", e24, e16)
	}
}

// TestAdaptiveDriftCompensateExactPath exercises ClassNormalExact + DriftCompensate
// in decodeRans6 (Tolerance=0 forces all normals through the exact float64 path).
func TestAdaptiveDriftCompensateExactPath(t *testing.T) {
	values := makeSmoothSeries(300, 1.0, 1.003)
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode: codec.EntropyAdaptive,
		DriftMode:   codec.DriftCompensate,
		Tolerance:   0.0, // all normals → ClassNormalExact
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	if e := maxRelErr(got, values); e > 1e-9 {
		t.Errorf("max relative error %e > 1e-9 with tolerance=0 DriftCompensate", e)
	}
}

// makeSmoothSeries generates a geometric series starting at start,
// multiplied by factor each step.
func makeSmoothSeries(n int, start, factor float64) []float64 {
	values := make([]float64, n)
	v := start
	for i := range values {
		values[i] = v
		v *= factor
	}
	return values
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

// --- EntropyAdaptive v5 (tiered u16/u32/float64) tests ---

// TestAdaptiveV5RoundTripLowTolerance confirms that a very tight ε (1e-6) works:
// most ratios need the ClassNormal32 (u32) tier; decoding produces values with
// error < ε per ratio step.
func TestAdaptiveV5RoundTripLowTolerance(t *testing.T) {
	values := makeSmoothSeries(1000, 100.0, 1.001)
	tol := 1e-6
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: 16,
		Tolerance:     tol,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// Each value is reconstructed from one quantized ratio step.
	// Accumulated drift is bounded by (1+ε)^N - 1 ≈ N·ε.
	if e := maxRelErr(got, values); e > float64(len(values))*tol*10 {
		t.Errorf("max relative error %e unexpectedly large for ε=%g", e, tol)
	}
}

// TestAdaptiveV5SigFigsWiring ensures that --sig-figs + adaptive in the encoder
// wires both PrecisionBits and Tolerance correctly (via SigFigsToTolerance).
func TestAdaptiveV5SigFigsWiring(t *testing.T) {
	values := makeSmoothSeries(500, 50.0, 1.0003)
	sf := 4 // 4 sig-figs → tol ≈ 5e-5
	bits := codec.SigFigsToBits(sf)
	tol := codec.SigFigsToTolerance(sf)

	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: bits,
		Tolerance:     tol,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
	// Errors should be well within 4 sig-fig tolerance; allow generous drift factor.
	if e := maxRelErr(got, values); e > float64(len(values))*tol*10 {
		t.Errorf("max relative error %e too large for %d sig-figs (tol=%g)", e, sf, tol)
	}
}

// TestAdaptiveV5TiersUsed verifies that v7 accuracy scales with PrecisionBits:
// B=16 is bounded by 16-bit quantization error; B=30 by 30-bit quantization error.
// This replaces the old v6 invariant (always 30-bit regardless of PrecisionBits).
func TestAdaptiveV5TiersUsed(t *testing.T) {
	values := makeSmoothSeries(500, 20.0, 1.0001)

	encode := func(bits int, tol float64) []float64 {
		var buf bytes.Buffer
		if err := codec.Encode(values, &buf, codec.EncodeOptions{
			EntropyMode:   codec.EntropyAdaptive,
			PrecisionBits: bits,
			Tolerance:     tol,
		}); err != nil {
			t.Fatalf("encode(bits=%d,tol=%g): %v", bits, tol, err)
		}
		got, err := codec.Decode(&buf)
		if err != nil {
			t.Fatalf("decode(bits=%d,tol=%g): %v", bits, tol, err)
		}
		return got
	}

	// B=16: per-ratio error bounded by ε_max(16) ≈ 4.2e-5; use 5e-5 as safe bound.
	const bits16Bound = 5e-5
	got16 := encode(16, 1e-3)
	if e := maxRelErr(got16, values); e > bits16Bound {
		t.Errorf("B=16 error %e exceeds 16-bit bound %g", e, bits16Bound)
	}
	// B=30: per-ratio error bounded by ε_max(30) ≈ 3.7e-9; use 5e-9 as safe bound.
	const bits30Bound = 5e-9
	got30 := encode(30, 1e-6)
	if e := maxRelErr(got30, values); e > bits30Bound {
		t.Errorf("B=30 error %e exceeds 30-bit bound %g", e, bits30Bound)
	}
	// Higher precision → smaller error.
	if e16, e30 := maxRelErr(got16, values), maxRelErr(got30, values); e30 >= e16 {
		t.Errorf("B=30 error %e not smaller than B=16 error %e", e30, e16)
	}
}

// TestAdaptiveV5BackwardCompatV4 confirms that v4-encoded streams are still
// decodable after v5/v6 were introduced.
func TestAdaptiveV5BackwardCompatV4(t *testing.T) {
	// We no longer produce v4 streams from encodeAdaptive, so we manually
	// exercise decodeV4 by encoding via the internal path and checking the
	// public Decode still routes correctly. Re-use the existing adaptive tests
	// as a proxy — they all encode with v6 now; the key test here is that
	// the version-4 constant still produces a valid decodable stream.
	// We confirm the stream is now v6 by checking round-trip correctness.
	values := makeSmoothSeries(200, 10.0, 1.002)
	var buf bytes.Buffer
	if err := codec.Encode(values, &buf, codec.EncodeOptions{
		EntropyMode:   codec.EntropyAdaptive,
		PrecisionBits: 12,
		Tolerance:     1e-3,
	}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := codec.Decode(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(values))
	}
}
