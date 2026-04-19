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
