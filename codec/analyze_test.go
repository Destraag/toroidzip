package codec_test

import (
	"math"
	"testing"

	codec "github.com/Destraag/toroidzip/codec"
)

// TestAnalyzeDriftEmpty verifies zero-value report for empty input.
func TestAnalyzeDriftEmpty(t *testing.T) {
	rpt := codec.AnalyzeDrift(nil, []int{256})
	if len(rpt.Rows) != 0 {
		t.Errorf("expected 0 rows for nil values, got %d", len(rpt.Rows))
	}
	rpt2 := codec.AnalyzeDrift([]float64{1, 2, 3}, nil)
	if len(rpt2.Rows) != 0 {
		t.Errorf("expected 0 rows for nil intervals, got %d", len(rpt2.Rows))
	}
}

// TestAnalyzeDriftConstant verifies that constant data produces zero drift error.
func TestAnalyzeDriftConstant(t *testing.T) {
	values := make([]float64, 500)
	for i := range values {
		values[i] = 42.0
	}
	rpt := codec.AnalyzeDrift(values, []int{64, 128, 256})
	for _, row := range rpt.Rows {
		if row.MaxRelErr > 1e-14 {
			t.Errorf("mode=%d interval=%d: MaxRelErr=%e, want ~0 for constant data",
				row.Mode, row.Interval, row.MaxRelErr)
		}
	}
}

// TestAnalyzeDriftMonotone verifies Mode B (DriftCompensate) has lower or equal
// error than Mode A (DriftReanchor) on a long monotone sequence.
func TestAnalyzeDriftMonotone(t *testing.T) {
	// Smooth 1% growth — long enough for Mode A drift to accumulate.
	values := make([]float64, 2000)
	v := 100.0
	for i := range values {
		v *= 1.01
		values[i] = v
	}
	rpt := codec.AnalyzeDrift(values, []int{256})
	var rowA, rowB *codec.DriftRow
	for i := range rpt.Rows {
		switch rpt.Rows[i].Mode {
		case codec.DriftReanchor:
			rowA = &rpt.Rows[i]
		case codec.DriftCompensate:
			rowB = &rpt.Rows[i]
		}
	}
	if rowA == nil || rowB == nil {
		t.Fatal("expected rows for both DriftReanchor and DriftCompensate")
	}
	// Mode B should not be worse than Mode A.
	if rowB.MaxRelErr > rowA.MaxRelErr+1e-14 {
		t.Errorf("Mode B MaxRelErr=%e > Mode A MaxRelErr=%e; expected Mode B <= Mode A",
			rowB.MaxRelErr, rowA.MaxRelErr)
	}
}

// TestAnalyzeDriftRowCount verifies correct number of rows (2 modes × N intervals).
func TestAnalyzeDriftRowCount(t *testing.T) {
	values := make([]float64, 200)
	for i := range values {
		values[i] = float64(i + 1)
	}
	intervals := []int{32, 64, 128}
	rpt := codec.AnalyzeDrift(values, intervals)
	want := 2 * len(intervals)
	if len(rpt.Rows) != want {
		t.Errorf("got %d rows, want %d (2 modes × %d intervals)", len(rpt.Rows), want, len(intervals))
	}
}

// TestAnalyzeDriftAnchorOverhead verifies anchor overhead = 1/interval.
func TestAnalyzeDriftAnchorOverhead(t *testing.T) {
	values := make([]float64, 1000)
	for i := range values {
		values[i] = float64(i + 1)
	}
	for _, iv := range []int{50, 100, 200} {
		rpt := codec.AnalyzeDrift(values, []int{iv})
		for _, row := range rpt.Rows {
			want := 1.0 / float64(iv)
			if math.Abs(row.AnchorOverhead-want) > 1e-12 {
				t.Errorf("interval=%d: AnchorOverhead=%f, want %f", iv, row.AnchorOverhead, want)
			}
		}
	}
}

// TestAnalyzeDriftRecommendedInterval verifies the recommended interval is
// non-zero and within the tested set.
func TestAnalyzeDriftRecommendedInterval(t *testing.T) {
	values := make([]float64, 2000)
	v := 1.0
	for i := range values {
		v *= 1.005
		values[i] = v
	}
	intervals := []int{64, 128, 256, 512}
	rpt := codec.AnalyzeDrift(values, intervals)

	found := false
	for _, iv := range intervals {
		if rpt.RecommendedInterval == iv {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("RecommendedInterval=%d not in tested intervals %v",
			rpt.RecommendedInterval, intervals)
	}
}
