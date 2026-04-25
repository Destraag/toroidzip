package codec

import "math"

// RatioClass categorizes a computed ratio for encoding decisions.
type RatioClass byte

const (
	// ClassIdentity is a ratio within epsilon of 1.0 — "no change" state.
	// This is the most common class in smooth data and will receive the
	// shortest codeword when entropy coding is added.
	ClassIdentity RatioClass = iota

	// ClassNormal16 is a well-behaved ratio outside the identity band but within
	// normal floating-point range. Arithmetic is valid and reversible.
	ClassNormal16

	// ClassBoundaryZero indicates the previous value was zero or near-zero,
	// making the ratio undefined or extreme. The current value is stored
	// verbatim as a boundary event. In the toroidal model, zero is the inner
	// boundary where ratio arithmetic loses meaning.
	ClassBoundaryZero

	// ClassBoundaryInf indicates the ratio exceeds the infinity threshold,
	// meaning the values have changed by an extreme factor. In the toroidal
	// model, zero and infinity are the same inner boundary approached from
	// opposite directions. Both collapse ratio arithmetic. Stored verbatim.
	ClassBoundaryInf

	// ClassReanchor is not a ratio — this position holds a verbatim float64
	// anchor used to reset reconstruction and bound cumulative drift.
	ClassReanchor

	// ClassNormalExact signals that a ratio is stored verbatim as a float64
	// payload. Fires when the per-ratio error check fails (Tolerance > 0 and
	// the quantized value exceeds tolerance) or when ratio == 0.
	// Used in the EntropyAdaptive stream (v4+). Not produced by Classify.
	ClassNormalExact

	// ClassNormal32 signals a ClassNormal16 ratio stored as a signed int32 offset
	// from log-space centre at configured precision. |offset| > 8,388,607.
	// Payload: 4 bytes (int32, little-endian). Decoded via DequantizeRatioOffset(off, bits).
	// Used in EntropyAdaptive v5 (absolute uint32) and v6/v7 (signed offset).
	// Not produced by Classify.
	ClassNormal32

	// ClassNormal8 is used in the v4-quantized and v7-adaptive offset streams.
	// It signals a ClassNormal16 ratio stored as a signed int8 offset from the
	// log-space centre at configured precision. |offset| ≤ 127.
	// Payload: 1 byte (int8). Decoded via DequantizeRatioOffset(off, bits).
	ClassNormal8

	// ClassNormal24 is used in the v4-quantized and v7-adaptive offset streams.
	// It signals a ClassNormal16 ratio stored as a signed int24 offset from the
	// log-space centre at configured precision. |offset| ≤ 8,388,607.
	// Payload: 3 bytes (int24, little-endian signed). Decoded via DequantizeRatioOffset(off, bits).
	ClassNormal24

	// ClassReanchorDynamic is used in v9 (EntropyAdaptive with dynamic offset).
	// Like ClassReanchor, but the anchor payload is followed by a 4-byte int32
	// k_center (absolute quantized symbol index). The per-ratio offsets in this
	// segment are encoded relative to k_center rather than the default centre
	// (2^(bits-1)). Reconstruction is identical — k_center only affects which
	// payload tier (u8/u16/u24/u32) each offset falls into.
	// Payload: float64 anchor (8 bytes) + int32 k_center (4 bytes, LE).
	ClassReanchorDynamic
)

// Thresholds for boundary classification. These are tunable.
const (
	// IdentityEpsilon: ratios within this distance of 1.0 are classified
	// as ClassIdentity. Reflects the Identity Law: x/x = 1 is the dominant
	// state in smooth data and should be treated as a special case.
	IdentityEpsilon = 1e-9

	// BoundaryInfThreshold: ratios with absolute value above this are classified
	// as ClassBoundaryInf (infinity boundary event).
	BoundaryInfThreshold = 1e15

	// BoundaryZeroThreshold: |prev| below this triggers ClassBoundaryZero (zero boundary event).
	// Separate from ratio classification — triggered on the input value, not the ratio.
	BoundaryZeroThreshold = 1e-300
)

// DriftMode selects the error-management strategy for cumulative reconstruction drift.
type DriftMode byte

const (
	// DriftReanchor is mode A: insert verbatim anchors every K values. Lossless.
	DriftReanchor DriftMode = iota

	// DriftCompensate is mode B: Kahan log-space product accumulation.
	// Near-lossless — reduces drift without increasing file size.
	DriftCompensate

	// DriftQuantize is mode C: the encoder rounds each ratio to float32 precision
	// before updating prev, mirroring what a float32 decoder would accumulate.
	// Use when the decoder is known to use float32 arithmetic — this eliminates
	// systematic divergence between the float64 encoder and float32 decoder.
	// Not a performance mode; --auto never recommends it.
	DriftQuantize
)

// DefaultDriftMode is used when EncodeOptions.DriftMode is zero-valued.
const DefaultDriftMode = DriftReanchor

// EntropyMode selects the entropy coding layer applied to the ratio stream.
type EntropyMode byte

const (
	// EntropyQuantized is explicitly lossy: ClassNormal16 ratios are mapped to
	// N-bit log-space symbols (see PrecisionBits / DefaultPrecisionBits) and
	// stored as signed offsets in the smallest fitting payload tier (u8/u16/u24/u32).
	// The class stream is rANS-coded. Use AnalyzePrecision to find the recommended
	// precision for a data set.
	EntropyQuantized EntropyMode = iota

	// EntropyAdaptive is the current adaptive stream (v8). Each ClassNormal16 ratio
	// is quantized at EncodeOptions.PrecisionBits (default: DefaultPrecisionBits=16)
	// using signed log-space offsets, then stored in the smallest payload tier
	// that fits the offset magnitude:
	//   - |offset| ≤ 127       → ClassNormal8  + int8  (1 byte)
	//   - |offset| ≤ 32,767    → ClassNormal16 + int16 (2 bytes)
	//   - |offset| ≤ 8,388,607 → ClassNormal24 + int24 (3 bytes)
	//   - |offset| > 8,388,607 → ClassNormal32 + int32 (4 bytes)
	//   - ratio == 0 or tol check fails → ClassNormalExact + float64 (8 bytes)
	// The class stream uses a 9-symbol rANS alphabet.
	// Accuracy is controlled by PrecisionBits: use SigFigsToBits(N) for N sig figs.
	// Set Tolerance = math.MaxFloat64 to quantize all normals at PrecisionBits.
	// Set Tolerance = SigFigsToTolerance(N) for per-ratio error gating.
	// At Tolerance=0 (default): all normals take ClassNormalExact — lossless-equivalent.
	EntropyAdaptive
)

// DefaultPrecisionBits is the quantisation depth used when
// EncodeOptions.PrecisionBits is 0 and EntropyMode is EntropyQuantized.
// 16 bits ≈ 4–5 significant figures (u16 payload tier, 2 bytes/symbol).
// Within the u16 tier (bits 9–16), all depths encode to the same byte count,
// so 16 is both the ceiling and the safest default. Use AnalyzePrecision to
// select precision appropriate for the data, or set PrecisionBits explicitly.
const DefaultPrecisionBits = 16

// Classify returns the RatioClass for a computed ratio value.
// The input should be the result of current/prev where prev != 0.
// For the case where prev == 0 or is near-zero, use ClassBoundaryZero directly.
func Classify(ratio float64) RatioClass {
	if math.IsNaN(ratio) || math.IsInf(ratio, 0) {
		return ClassBoundaryInf
	}
	if math.Abs(ratio-1.0) < IdentityEpsilon {
		return ClassIdentity
	}
	if math.Abs(ratio) > BoundaryInfThreshold {
		return ClassBoundaryInf
	}
	return ClassNormal16
}

// IsBoundary returns true if the class represents a boundary event (not a normal ratio).
func IsBoundary(c RatioClass) bool {
	return c == ClassBoundaryZero || c == ClassBoundaryInf
}
