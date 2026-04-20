// Package codec provides log-space ratio quantisation for the quantized entropy mode (EntropyQuantized).
//
// Design:
//   - Quantisation is uniform in log₂ space, centred on ratio = 1.0.
//   - The covered range is [2^−QuantMaxLog2R, 2^QuantMaxLog2R] = [1/16, 16].
//   - Ratios outside this range are clamped to the nearest boundary symbol.
//   - The precision (bits) is caller-controlled; bits ∈ [1, 30].
//
// Payload tier derived from bits (used by codec.go and CLI):
//
//	bits 1–8  → uint8  (1 byte) — ~1–2 sig figs
//	bits 9–16 → uint16 (2 bytes) — ~2–5 sig figs
//	bits 17–30→ uint32 (4 bytes) — ~5–9 sig figs
//
// This is purely a numeric mapping — it knows nothing about rANS or stream
// format. Integration happens in codec.go.
package codec

import (
	"math"
	"sort"
)

// QuantMaxLog2R is the absolute log₂ limit of the quantisation range.
// Ratios in [2^−QuantMaxLog2R, 2^QuantMaxLog2R] = [1/16, 16] map to interior
// symbols. Ratios outside this range are clamped to the boundary symbols.
const QuantMaxLog2R = 4.0

// QuantizeRatio maps a ClassNormal ratio to a symbol in [0, 2^bits).
// The mapping is uniform in log₂ space, centred on ratio = 1.0 (log₂ = 0).
// bits is clamped to [1, 30]. Non-positive ratios are clamped to symbol 0.
func QuantizeRatio(ratio float64, bits int) uint32 {
	bits = clampBits(bits)
	levels := uint32(1) << bits

	var logR float64
	if ratio <= 0 {
		logR = -QuantMaxLog2R // clamp to left boundary
	} else {
		logR = math.Log2(ratio)
	}

	// Normalise log₂ ratio to [0, 1], then scale to [0, levels).
	t := (logR/QuantMaxLog2R + 1.0) * 0.5 // [−QuantMaxLog2R,+QuantMaxLog2R] → [0,1]
	t = math.Max(0, math.Min(1, t))
	sym := uint32(t * float64(levels))
	if sym >= levels {
		sym = levels - 1
	}
	return sym
}

// DequantizeRatio maps a quantized symbol back to the centre of its log₂
// bucket. bits must match the value used in QuantizeRatio.
func DequantizeRatio(sym uint32, bits int) float64 {
	bits = clampBits(bits)
	levels := uint32(1) << bits
	if sym >= levels {
		sym = levels - 1
	}
	// Centre of bucket sym in normalised [0, 1] space.
	t := (float64(sym) + 0.5) / float64(levels)
	// Map [0,1] → [−QuantMaxLog2R, +QuantMaxLog2R].
	logR := (t*2.0 - 1.0) * QuantMaxLog2R
	return math.Pow(2, logR)
}

// PrecisionReport is returned by AnalyzePrecision.
type PrecisionReport struct {
	// Entropy[b] is the Shannon entropy (bits/symbol) of the quantised symbol
	// distribution at b-bit precision, for b ∈ [1..30]. Entropy[0] is unused.
	Entropy [31]float64

	// RecommendedBits is the entropy-curve knee, then rounded UP to the top
	// of its payload tier (8, 16, or 30). Within a tier all bit depths produce
	// the same encoded size, so higher precision is always free.
	// Returns 1 when the data is essentially constant, 30 when there is fine
	// structure at all precision levels.
	RecommendedBits int

	// RecommendedSigFigs is the significant-figure count implied by
	// RecommendedBits, derived via BitsToSigFigs.
	RecommendedSigFigs int

	// Coverage is the fraction of ratios that fall within the quantiser range
	// [2^−QuantMaxLog2R, 2^QuantMaxLog2R] without clamping.
	Coverage float64

	// IdentityFraction is the fraction of input ratios that fall within
	// IdentityEpsilon of 1.0 (i.e. would be classified as ClassIdentity).
	// When this is low (< ~5%), lossless mode gains little over uncompressed
	// because almost no payload bytes are eliminated.
	IdentityFraction float64
}

// clampBits restricts precision bits to the valid range [1, 30].
func clampBits(bits int) int {
	if bits < 1 {
		return 1
	}
	if bits > 30 {
		return 30
	}
	return bits
}

// AnalyzePrecision surveys ratios to recommend a quantisation bit depth.
// Only ClassNormal values should be passed; boundary and reanchor events are
// stored verbatim and do not need quantisation.
//
// Returns a zero-value report for empty input.
func AnalyzePrecision(ratios []float64) PrecisionReport {
	if len(ratios) == 0 {
		return PrecisionReport{}
	}

	// Count ratios within the quantiser range (not clamped).
	var inRange int
	for _, r := range ratios {
		if r > 0 {
			lg := math.Log2(r)
			if lg >= -QuantMaxLog2R && lg <= QuantMaxLog2R {
				inRange++
			}
		}
	}

	var rpt PrecisionReport
	rpt.Coverage = float64(inRange) / float64(len(ratios))

	// Compute Shannon entropy at each precision using sort-based counting.
	// This avoids allocating a histogram of size 2^b (which reaches gigabytes
	// at b=28+). Sorting the n symbols and counting runs is O(n log n) per b.
	n := float64(len(ratios))
	syms := make([]uint32, len(ratios))
	for b := 1; b <= 30; b++ {
		for j, r := range ratios {
			syms[j] = QuantizeRatio(r, b)
		}
		sort.Slice(syms, func(i, k int) bool { return syms[i] < syms[k] })
		var H float64
		for i := 0; i < len(syms); {
			j := i
			for j < len(syms) && syms[j] == syms[i] {
				j++
			}
			p := float64(j-i) / n
			H -= p * math.Log2(p)
			i = j
		}
		rpt.Entropy[b] = H
	}

	// Knee detection: walk b from 2 to 30; keep updating RecommendedBits to
	// the last b where the marginal gain is still ≥ 5 % of the total range.
	// If total range ≈ 0 (constant data), 1 bit suffices.
	totalRange := rpt.Entropy[30] - rpt.Entropy[1]
	rpt.RecommendedBits = 1
	if totalRange > 1e-12 {
		threshold := 0.05 * totalRange
		for b := 2; b <= 30; b++ {
			if rpt.Entropy[b]-rpt.Entropy[b-1] >= threshold {
				rpt.RecommendedBits = b
			}
		}
	}
	// Round up to the ceiling of the payload tier: within a tier all bit
	// depths encode to the same byte count, so extra precision is free.
	switch QuantPayloadTier(rpt.RecommendedBits) {
	case 1: // u8 tier: bits 1–8
		rpt.RecommendedBits = 8
	case 2: // u16 tier: bits 9–16
		rpt.RecommendedBits = 16
	default: // u32 tier: bits 17–30
		rpt.RecommendedBits = 30
	}
	rpt.RecommendedSigFigs = BitsToSigFigs(rpt.RecommendedBits)

	// Identity fraction: how many ratios are within IdentityEpsilon of 1.0.
	var identityCount int
	for _, r := range ratios {
		if r > 0 && math.Abs(r-1.0) < IdentityEpsilon {
			identityCount++
		}
	}
	rpt.IdentityFraction = float64(identityCount) / float64(len(ratios))

	return rpt
}

// SigFigsToBits returns the minimum bit depth B such that the worst-case
// relative quantisation error is < 5×10^(−n).
//
// Derivation: ε_max = 2^(QuantMaxLog2R/2^B) − 1 < 5×10^(−n)
// → B = ⌈log₂(4 / log₂(1 + 5×10^(−n)))⌉
//
//	n=1→3, n=2→6, n=3→10, n=4→13, n=5→16, n=6→20, n=7→23, n=8→26, n=9→30
//
// n is clamped to [1, 9]; result is clamped to [1, 30].
func SigFigsToBits(n int) int {
	if n < 1 {
		n = 1
	}
	if n > 9 {
		n = 9
	}
	epsilon := 5 * math.Pow(10, float64(-n))
	b := int(math.Ceil(math.Log2(4 / math.Log2(1+epsilon))))
	if b < 1 {
		b = 1
	}
	if b > 30 {
		b = 30
	}
	return b
}

// SigFigsToTolerance returns the per-ratio ε tolerance corresponding to N
// significant figures: ε = 0.5 × 10^(−N). This is the maximum relative error
// that still rounds to the correct Nth significant figure.
// n is clamped to [1, 9].
func SigFigsToTolerance(n int) float64 {
	if n < 1 {
		n = 1
	}
	if n > 9 {
		n = 9
	}
	return 0.5 * math.Pow(10, float64(-n))
}

// BitsToSigFigs returns the number of significant figures guaranteed by B bits
// of log-space quantisation. This is the floor of −log₁₀(ε_max/5), where
// ε_max = 2^(QuantMaxLog2R/2^(B−1)) − 1.
func BitsToSigFigs(b int) int {
	b = clampBits(b)
	// ε_max = 2^(4/2^b) − 1
	epsilon := math.Pow(2, QuantMaxLog2R/math.Pow(2, float64(b))) - 1
	if epsilon <= 0 {
		return 9
	}
	sf := int(math.Floor(-math.Log10(epsilon / 5)))
	if sf < 1 {
		return 1
	}
	return sf
}

// QuantPayloadTier returns the payload byte width for a given bit depth:
//
//	bits 1–8  → 1 (uint8)
//	bits 9–16 → 2 (uint16)
//	bits 17–30→ 4 (uint32)
func QuantPayloadTier(bits int) int {
	bits = clampBits(bits)
	switch {
	case bits <= 8:
		return 1
	case bits <= 16:
		return 2
	default:
		return 4
	}
}
