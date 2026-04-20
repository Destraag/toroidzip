package codec

import "math"

// RatioClass categorizes a computed ratio for encoding decisions.
type RatioClass byte

const (
	// ClassIdentity is a ratio within epsilon of 1.0 — "no change" state.
	// This is the most common class in smooth data and will receive the
	// shortest codeword when entropy coding is added.
	ClassIdentity RatioClass = iota

	// ClassNormal is a well-behaved ratio outside the identity band but within
	// normal floating-point range. Arithmetic is valid and reversible.
	ClassNormal

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

	// ClassNormalExact is used only in the v4 adaptive stream (EntropyAdaptive).
	// It signals that the ratio could not be quantized within the declared
	// tolerance ε and is stored as a full float64 payload instead. This class
	// never appears in v1–v3 streams and is not produced by Classify.
	ClassNormalExact
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

	// DriftQuantize is mode C: ratios rounded to float32 precision before storage.
	// Explicitly lossy. Produces more compressible ratio distributions.
	DriftQuantize
)

// DefaultDriftMode is used when EncodeOptions.DriftMode is zero-valued.
const DefaultDriftMode = DriftReanchor

// EntropyMode selects the entropy coding layer applied to the ratio stream.
type EntropyMode byte

const (
	// EntropyRaw is the Milestone 1 baseline: no entropy coding, one class byte
	// plus a full float64 per value regardless of class.
	EntropyRaw EntropyMode = iota

	// EntropyLossless compresses the class byte stream with rANS and omits the
	// float64 payload for ClassIdentity events (decoder reconstructs as ratio=1.0).
	// Provides large reductions on smooth data. Near-lossless: ClassIdentity
	// events introduce at most IdentityEpsilon relative error, bounded by reanchors.
	EntropyLossless

	// EntropyQuantized is explicitly lossy: ClassNormal ratios are mapped to
	// N-bit log-space symbols (see PrecisionBits / DefaultPrecisionBits) and
	// stored as uint16 instead of float64. The class stream is rANS-coded.
	// Use AnalyzePrecision to find the recommended precision for a data set.
	EntropyQuantized

	// EntropyAdaptive is the v4 hybrid stream. For each ClassNormal ratio,
	// the encoder checks whether quantization error is within tolerance ε:
	//   - error < ε  → store as uint16 quantized symbol (ClassNormal)
	//   - error ≥ ε  → store as float64 verbatim (ClassNormalExact)
	// The class stream uses a 6-symbol rANS alphabet.
	// At ε=0 (default Tolerance): output is bit-identical to EntropyLossless.
	// At ε=∞: output matches EntropyQuantized at bits≤16.
	// Set EncodeOptions.Tolerance to control ε (e.g. 1e-4 ≈ 4 sig-fig loss bound).
	// Set EncodeOptions.PrecisionBits for the quantized symbols (capped at 16).
	EntropyAdaptive
)

// DefaultPrecisionBits is the quantisation depth used when
// EncodeOptions.PrecisionBits is 0 and EntropyMode is EntropyQuantized.
// 8 bits ≈ 2 significant figures (u8 payload tier, 1 byte/symbol).
// Use AnalyzePrecision to select precision appropriate for the data, or
// set PrecisionBits explicitly: 16 = u16 tier (~4 sf), 30 = u32 tier (~9 sf).
const DefaultPrecisionBits = 8

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
	return ClassNormal
}

// IsBoundary returns true if the class represents a boundary event (not a normal ratio).
func IsBoundary(c RatioClass) bool {
	return c == ClassBoundaryZero || c == ClassBoundaryInf
}
