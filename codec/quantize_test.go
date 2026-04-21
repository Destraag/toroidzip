package codec_test

import (
	"math"
	"testing"

	"github.com/Destraag/toroidzip/codec"
)

// --- QuantizeRatio / DequantizeRatio ---

// TestQuantizeRatioCenter verifies ratio=1.0 maps to the midpoint symbol.
func TestQuantizeRatioCenter(t *testing.T) {
	for _, bits := range []int{1, 4, 8, 12, 16, 20, 28, 20, 28} {
		sym := codec.QuantizeRatio(1.0, bits)
		levels := uint32(1) << bits
		want := levels / 2
		if sym != want {
			t.Errorf("bits=%d: QuantizeRatio(1.0) = %d, want %d", bits, sym, want)
		}
	}
}

// TestQuantizeRatioMonotonic verifies that increasing ratio gives non-decreasing symbol.
func TestQuantizeRatioMonotonic(t *testing.T) {
	bits := 8
	prev := uint32(0)
	for i := -200; i <= 200; i++ {
		ratio := math.Pow(2, float64(i)*codec.QuantMaxLog2R/200)
		sym := codec.QuantizeRatio(ratio, bits)
		if sym < prev {
			t.Errorf("not monotonic at log2=%.3f: sym=%d < prev=%d",
				math.Log2(ratio), sym, prev)
		}
		prev = sym
	}
}

// TestDequantizeRoundTrip verifies quantize(dequantize(sym)) == sym for all symbols.
func TestDequantizeRoundTrip(t *testing.T) {
	for _, bits := range []int{1, 4, 8} {
		levels := uint32(1) << bits
		for sym := uint32(0); sym < levels; sym++ {
			ratio := codec.DequantizeRatio(sym, bits)
			back := codec.QuantizeRatio(ratio, bits)
			if back != sym {
				t.Errorf("bits=%d sym=%d: DequantizeRatio=%f → re-quantized=%d",
					bits, sym, ratio, back)
			}
		}
	}
}

// TestQuantizeRatioClamping verifies extreme ratios land on the boundary symbols.
func TestQuantizeRatioClamping(t *testing.T) {
	bits := 8
	levels := uint32(1) << bits

	// Very large ratio → last symbol.
	if sym := codec.QuantizeRatio(1e30, bits); sym != levels-1 {
		t.Errorf("1e30 → sym=%d, want %d", sym, levels-1)
	}
	// Very small positive ratio → symbol 0.
	if sym := codec.QuantizeRatio(1e-30, bits); sym != 0 {
		t.Errorf("1e-30 → sym=%d, want 0", sym)
	}
	// Non-positive ratio → symbol 0.
	for _, r := range []float64{0, -1, -1e10} {
		if sym := codec.QuantizeRatio(r, bits); sym != 0 {
			t.Errorf("ratio=%g → sym=%d, want 0", r, sym)
		}
	}
}

// TestQuantizeBitsClamped verifies clampBits behaviour at both boundaries.
func TestQuantizeBitsClamped(t *testing.T) {
	// bits=0 should behave as bits=1 (2 symbols: 0 or 1).
	sym := codec.QuantizeRatio(1.0, 0)
	if sym > 1 {
		t.Errorf("bits=0 (clamped to 1): sym=%d, want 0 or 1", sym)
	}
	// bits=32 should behave as bits=30 (cap).
	sym30 := codec.QuantizeRatio(1.5, 30)
	sym32 := codec.QuantizeRatio(1.5, 32)
	if sym32 != sym30 {
		t.Errorf("bits=32 (clamped to 30): sym=%d, want %d", sym32, sym30)
	}
}

// TestDequantizeWithinBucket verifies the dequantized value falls inside its
// bucket when re-quantized (basic accuracy check).
func TestDequantizeWithinBucket(t *testing.T) {
	bits := 8
	levels := uint32(1) << bits
	for sym := uint32(0); sym < levels; sym++ {
		ratio := codec.DequantizeRatio(sym, bits)
		if ratio <= 0 {
			t.Errorf("sym=%d: DequantizeRatio returned %f (want > 0)", sym, ratio)
		}
		if got := codec.QuantizeRatio(ratio, bits); got != sym {
			t.Errorf("sym=%d: DequantizeRatio=%f quantizes back to %d", sym, ratio, got)
		}
	}
}

// --- AnalyzePrecision ---

// TestAnalyzePrecisionEmpty verifies a zero-value report for empty input.
func TestAnalyzePrecisionEmpty(t *testing.T) {
	rpt := codec.AnalyzePrecision(nil)
	if rpt.RecommendedBits != 0 || rpt.Coverage != 0 {
		t.Errorf("expected zero report for nil input, got %+v", rpt)
	}
}

// TestAnalyzePrecisionConstant verifies that constant data (ratio=1.0) gives
// zero entropy at all levels and recommends 1 bit.
func TestAnalyzePrecisionConstant(t *testing.T) {
	ratios := make([]float64, 1000)
	for i := range ratios {
		ratios[i] = 1.0
	}
	rpt := codec.AnalyzePrecision(ratios)

	for b := 1; b <= 30; b++ {
		if rpt.Entropy[b] > 1e-10 {
			t.Errorf("Entropy[%d] = %g, want ~0 for constant data", b, rpt.Entropy[b])
		}
	}
	// For constant data the knee is at bit 1, which sits in the u8 tier.
	// The tier-ceiling rule bumps it to 8 — max free precision in that tier.
	if rpt.RecommendedBits != 8 {
		t.Errorf("RecommendedBits = %d, want 8 (u8 tier ceiling) for constant data", rpt.RecommendedBits)
	}
	if rpt.Coverage < 1.0-1e-9 {
		t.Errorf("Coverage = %f, want 1.0 for ratio=1.0", rpt.Coverage)
	}
}

// TestAnalyzePrecisionEntropyIncreases verifies that entropy is non-decreasing
// with bit depth up to 30 bits.
func TestAnalyzePrecisionEntropyIncreases(t *testing.T) {
	ratios := makeLogNormalRatios(5000, 0.5)
	rpt := codec.AnalyzePrecision(ratios)

	for b := 2; b <= 30; b++ {
		if rpt.Entropy[b] < rpt.Entropy[b-1]-1e-12 {
			t.Errorf("Entropy[%d]=%f < Entropy[%d]=%f (should be non-decreasing)",
				b, rpt.Entropy[b], b-1, rpt.Entropy[b-1])
		}
	}
}

// TestAnalyzePrecisionSmoothData verifies realistic behaviour on smooth data.
func TestAnalyzePrecisionSmoothData(t *testing.T) {
	// Small log-normal variation: ratios close to 1.0.
	ratios := makeLogNormalRatios(10000, 0.05)
	rpt := codec.AnalyzePrecision(ratios)

	if rpt.Coverage < 0.99 {
		t.Errorf("Coverage = %f, want >= 0.99 for small-variation data", rpt.Coverage)
	}
	if rpt.Entropy[8] <= rpt.Entropy[2] {
		t.Errorf("Entropy[8]=%f should be > Entropy[2]=%f", rpt.Entropy[8], rpt.Entropy[2])
	}
	if rpt.RecommendedBits < 1 || rpt.RecommendedBits > 30 {
		t.Errorf("RecommendedBits = %d out of range [1,30]", rpt.RecommendedBits)
	}
}

// TestAnalyzePrecisionWideData verifies coverage is < 1 when ratios span a
// range larger than the quantiser window.
func TestAnalyzePrecisionWideData(t *testing.T) {
	// Large log-normal σ: many ratios will fall outside [1/16, 16].
	ratios := makeLogNormalRatios(5000, 8.0)
	rpt := codec.AnalyzePrecision(ratios)
	if rpt.Coverage >= 1.0 {
		t.Errorf("Coverage = %f, want < 1.0 for wide-spread data", rpt.Coverage)
	}
}

// --- SigFigsToBits / BitsToSigFigs / QuantPayloadTier ---

// TestSigFigsToBitsTable verifies known values from the design table.
func TestSigFigsToBitsTable(t *testing.T) {
	cases := []struct{ n, wantB int }{
		{1, 3},
		{2, 6},
		{3, 10},
		{4, 13},
		{5, 16},
		{6, 20},
		{7, 23},
	}
	for _, tc := range cases {
		got := codec.SigFigsToBits(tc.n)
		if got != tc.wantB {
			t.Errorf("SigFigsToBits(%d) = %d, want %d", tc.n, got, tc.wantB)
		}
	}
}

// TestSigFigsToBitsClamped verifies clamping at 0 and 9.
func TestSigFigsToBitsClamped(t *testing.T) {
	low := codec.SigFigsToBits(0)
	if low != codec.SigFigsToBits(1) {
		t.Errorf("SigFigsToBits(0) = %d, want same as SigFigsToBits(1) = %d", low, codec.SigFigsToBits(1))
	}
	high := codec.SigFigsToBits(99)
	if high > 30 {
		t.Errorf("SigFigsToBits(99) = %d, want <= 30", high)
	}
	// n=9 should clamp to 30 (formula gives B=32 without cap).
	if b := codec.SigFigsToBits(9); b != 30 {
		t.Errorf("SigFigsToBits(9) = %d, want 30 (clamped)", b)
	}
}

// TestBitsToSigFigsMonotonic verifies that more bits means >= sig figs.
func TestBitsToSigFigsMonotonic(t *testing.T) {
	prev := 0
	for b := 1; b <= 30; b++ {
		got := codec.BitsToSigFigs(b)
		if got < prev {
			t.Errorf("BitsToSigFigs(%d)=%d < BitsToSigFigs(%d)=%d", b, got, b-1, prev)
		}
		prev = got
	}
}

// TestQuantPayloadTier verifies tier assignment at tier boundaries.
func TestQuantPayloadTier(t *testing.T) {
	cases := []struct{ bits, want int }{
		{1, 1}, {4, 1}, {8, 1},
		{9, 2}, {12, 2}, {16, 2},
		{17, 4}, {20, 4}, {30, 4},
	}
	for _, tc := range cases {
		got := codec.QuantPayloadTier(tc.bits)
		if got != tc.want {
			t.Errorf("QuantPayloadTier(%d) = %d, want %d", tc.bits, got, tc.want)
		}
	}
}

// TestSigFigsRoundTrip verifies SigFigsToBits and BitsToSigFigs are consistent:
// BitsToSigFigs(SigFigsToBits(n)) >= n for all n in [1,7].
func TestSigFigsRoundTrip(t *testing.T) {
	for n := 1; n <= 7; n++ {
		b := codec.SigFigsToBits(n)
		back := codec.BitsToSigFigs(b)
		if back < n {
			t.Errorf("n=%d: SigFigsToBits=%d BitsToSigFigs=%d (want >= %d)", n, b, back, n)
		}
	}
}

// --- Benchmarks ---

func BenchmarkQuantizeRatio(b *testing.B) {
	ratios := makeLogNormalRatios(1000, 0.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, r := range ratios {
			_ = codec.QuantizeRatio(r, 8)
		}
	}
}

// TestSigFigsToToleranceBounds verifies clamping at n<1 and n>11.
func TestSigFigsToToleranceBounds(t *testing.T) {
	low := codec.SigFigsToTolerance(0)
	if low != codec.SigFigsToTolerance(1) {
		t.Errorf("SigFigsToTolerance(0) = %g, want same as SigFigsToTolerance(1) = %g",
			low, codec.SigFigsToTolerance(1))
	}
	high := codec.SigFigsToTolerance(99)
	if high != codec.SigFigsToTolerance(11) {
		t.Errorf("SigFigsToTolerance(99) = %g, want same as SigFigsToTolerance(11) = %g",
			high, codec.SigFigsToTolerance(11))
	}
	// Spot-check: n=4 → 0.5e-4
	if got, want := codec.SigFigsToTolerance(4), 0.5e-4; math.Abs(got-want) > 1e-15 {
		t.Errorf("SigFigsToTolerance(4) = %g, want %g", got, want)
	}
}

// TestSigFigsToMaxK verifies that SigFigsToMaxK returns the circuit-breaker
// value 10,000 for all N in [1,9] regardless of the sig-figs argument.
// The end-to-end guarantee is owned by the adaptive drift check; K_max is
// only a last-resort backstop and no longer needs to satisfy K×ε_per ≤ T_end.
func TestSigFigsToMaxK(t *testing.T) {
	for n := 1; n <= 9; n++ {
		k := codec.SigFigsToMaxK(n)
		if k != 10_000 {
			t.Errorf("SigFigsToMaxK(%d) = %d, want 10000", n, k)
		}
	}
}

// --- helpers ---

// makeLogNormalRatios generates n ratios with log₂(ratio) varying sinusoidally
// with the given amplitude (in log₂ units). Deterministic, no random package needed.
func makeLogNormalRatios(n int, amplitude float64) []float64 {
	ratios := make([]float64, n)
	for i := range ratios {
		logR := amplitude * math.Sin(float64(i)*0.314159)
		ratios[i] = math.Pow(2, logR)
	}
	return ratios
}

// --- QuantizeRatioOffset / DequantizeRatioOffset ---

// TestQuantizeRatioOffsetCenter verifies ratio=1.0 maps to offset 0.
func TestQuantizeRatioOffsetCenter(t *testing.T) {
	for _, bits := range []int{1, 8, 16, 30} {
		off := codec.QuantizeRatioOffset(1.0, bits)
		if off != 0 {
			t.Errorf("bits=%d: QuantizeRatioOffset(1.0) = %d, want 0", bits, off)
		}
	}
}

// TestQuantizeRatioOffsetRoundTrip verifies the offset round-trip recovers the
// same ratio as the absolute round-trip at bits=30.
func TestQuantizeRatioOffsetRoundTrip(t *testing.T) {
	ratios := []float64{0.5, 0.9999, 1.0, 1.0001, 1.5, 2.0, 4.0, 1.0 / 16, 16.0}
	for _, r := range ratios {
		off := codec.QuantizeRatioOffset(r, 30)
		got := codec.DequantizeRatioOffset(off, 30)
		want := codec.DequantizeRatio(codec.QuantizeRatio(r, 30), 30)
		if math.Abs(got-want) > 1e-15 {
			t.Errorf("ratio=%g: offset round-trip=%g, absolute round-trip=%g, delta=%g",
				r, got, want, math.Abs(got-want))
		}
	}
}

// TestQuantizeRatioOffsetEquivalence verifies that the offset encoding is
// algebraically equivalent to the absolute encoding for all symbols at bits=16.
func TestQuantizeRatioOffsetEquivalence(t *testing.T) {
	bits := 16
	levels := uint32(1) << bits
	for sym := uint32(0); sym < levels; sym++ {
		ratio := codec.DequantizeRatio(sym, bits)
		off := codec.QuantizeRatioOffset(ratio, bits)
		got := codec.DequantizeRatioOffset(off, bits)
		want := codec.DequantizeRatio(sym, bits)
		if math.Abs(got-want) > 1e-15 {
			t.Errorf("sym=%d: DequantizeRatioOffset=%g, DequantizeRatio=%g", sym, got, want)
		}
	}
}

// TestQuantizeRatioOffsetSmoothRange verifies that typical smooth sensor-like
// ratios (within ±0.01% of 1.0, i.e. ratio=1.0001) produce offsets that fit
// in int16 at bits=30. This is the primary motivation for the v6 stream:
// smooth-data ClassNormal32 payloads should cluster near 0 rather than near
// 2^29, making most of them expressible as int16 instead of uint32.
func TestQuantizeRatioOffsetSmoothRange(t *testing.T) {
	const bits = 30
	// Ratios within ±0.01% of 1.0 — typical smooth-data step size.
	smoothRatios := []float64{
		1.0 + 1e-4, 1.0 - 1e-4,
		1.0 + 1e-5, 1.0 - 1e-5,
	}
	const int16Max = 1<<15 - 1
	for _, r := range smoothRatios {
		off := codec.QuantizeRatioOffset(r, bits)
		if off > int16Max || off < -int16Max {
			t.Errorf("ratio=%g: offset=%d exceeds int16 range ±%d", r, off, int16Max)
		}
	}
}

// TestQuantizeRatioOffsetRange verifies the offset range fits in int32 for
// all valid bits and for the extreme symbols (0 and 2^bits-1).
func TestQuantizeRatioOffsetRange(t *testing.T) {
	for _, bits := range []int{1, 8, 16, 30} {
		minRatio := codec.DequantizeRatio(0, bits)
		maxRatio := codec.DequantizeRatio((uint32(1)<<bits)-1, bits)
		minOff := codec.QuantizeRatioOffset(minRatio, bits)
		maxOff := codec.QuantizeRatioOffset(maxRatio, bits)
		centre := int32(1 << (bits - 1))
		if minOff != -centre {
			t.Errorf("bits=%d: min offset=%d, want %d", bits, minOff, -centre)
		}
		if maxOff != centre-1 {
			t.Errorf("bits=%d: max offset=%d, want %d", bits, maxOff, centre-1)
		}
	}
}
