// export_test.go exposes unexported symbols for use in external test files
// (package codec_test). This file is compiled only during testing.
package codec

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
