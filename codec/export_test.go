// export_test.go exposes unexported symbols for use in external test files
// (package codec_test). This file is compiled only during testing.
package codec

import "io"

// ComputeRatioExported is a test-only shim for the unexported computeRatio.
func ComputeRatioExported(current, prev float64) (float64, RatioClass) {
	return computeRatio(current, prev)
}

// GatherV7ClassesForTest calls gatherRans7v7 and returns only the class
// stream.  Used by mechanism_test.go to assert structural invariants on the
// class stream without round-tripping through the full encode/decode pipeline.
func GatherV7ClassesForTest(values []float64, opts EncodeOptions) []byte {
	classes, _ := gatherRans7v7(values, opts)
	return classes
}

// EncodeQuantizedLegacyForTest encodes using the legacy v3 quantized encoder
// (encodeQuantized). Used by backward-compat tests to produce v3 streams for
// decode verification.
func EncodeQuantizedLegacyForTest(values []float64, w io.Writer, opts EncodeOptions) error {
	return encodeQuantized(values, w, opts)
}

// EncodeAdaptiveV5ForTest encodes using the legacy v5 adaptive encoder.
// Used by backward-compat tests to produce v5 streams for decode verification.
func EncodeAdaptiveV5ForTest(values []float64, w io.Writer, opts EncodeOptions) error {
	return encodeAdaptiveV5(values, w, opts)
}

// EncodeAdaptiveV6ForTest encodes using the legacy v6 adaptive encoder.
// Used by backward-compat tests to produce v6 streams for decode verification.
func EncodeAdaptiveV6ForTest(values []float64, w io.Writer, opts EncodeOptions) error {
	return encodeAdaptiveV6(values, w, opts)
}
