package codec

import "math"

// RatioClass categorizes a computed ratio for encoding decisions.
type RatioClass byte

const (
	// ClassIdentity: ratio is within epsilon of 1.0 — "no change" state.
	// This is the most common class in smooth data and will receive the
	// shortest codeword when entropy coding is added.
	ClassIdentity RatioClass = iota

	// ClassNormal: well-behaved ratio outside the identity band but within
	// normal floating-point range. Arithmetic is valid and reversible.
	ClassNormal

	// ClassBoundaryZero: the previous value was zero or near-zero, making the
	// ratio undefined or extreme. The current value is stored verbatim as a
	// boundary event. In the toroidal model, zero is the inner boundary where
	// ratio arithmetic loses meaning.
	ClassBoundaryZero

	// ClassBoundaryInf: the ratio exceeds the infinity threshold, meaning the
	// values have changed by an extreme factor. In the toroidal model, zero and
	// infinity are the same inner boundary approached from opposite directions.
	// Both collapse ratio arithmetic. Stored verbatim as a boundary event.
	ClassBoundaryInf

	// ClassReanchor: not a ratio — this position holds a verbatim float64
	// anchor used to reset reconstruction and bound cumulative drift.
	ClassReanchor
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
