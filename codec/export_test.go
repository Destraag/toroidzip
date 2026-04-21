// export_test.go exposes unexported symbols for use in external test files
// (package codec_test). This file is compiled only during testing.
package codec

// ComputeRatioExported is a test-only shim for the unexported computeRatio.
func ComputeRatioExported(current, prev float64) (float64, RatioClass) {
	return computeRatio(current, prev)
}
