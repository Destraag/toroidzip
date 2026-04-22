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

// TestExtractNormalRatios verifies that ExtractNormalRatios returns the correct
// ClassNormal ratios for a simple monotone series.
func TestExtractNormalRatios(t *testing.T) {
	// Simple 1% growth — all values should be ClassNormal.
	values := make([]float64, 100)
	v := 10.0
	for i := range values {
		v *= 1.01
		values[i] = v
	}
	ratios := codec.ExtractNormalRatios(values, codec.DriftReanchor, 256)
	if len(ratios) == 0 {
		t.Fatal("ExtractNormalRatios returned empty slice for normal series")
	}
	for i, r := range ratios {
		if math.IsNaN(r) || math.IsInf(r, 0) || r <= 0 {
			t.Errorf("ratio[%d]=%v is not a valid positive finite ratio", i, r)
		}
	}
}

// TestExtractNormalRatiosEdgeCases verifies ExtractNormalRatios edge cases.
func TestExtractNormalRatiosEdgeCases(t *testing.T) {
	// Empty input.
	if r := codec.ExtractNormalRatios(nil, codec.DriftReanchor, 256); len(r) != 0 {
		t.Errorf("nil input: expected empty, got %d ratios", len(r))
	}
	// Single value — no ratios possible.
	if r := codec.ExtractNormalRatios([]float64{1.0}, codec.DriftReanchor, 256); len(r) != 0 {
		t.Errorf("single value: expected empty, got %d ratios", len(r))
	}
	// Zero interval — treated as invalid.
	if r := codec.ExtractNormalRatios([]float64{1.0, 2.0}, codec.DriftReanchor, 0); len(r) != 0 {
		t.Errorf("zero interval: expected empty, got %d ratios", len(r))
	}
}

// TestExtractNormalRatiosBoundaryEvents verifies boundary events: only
// ClassBoundaryZero (triggered when prev is near-zero) is excluded; the
// zero-ratio itself is ClassNormal and is included.
func TestExtractNormalRatiosBoundaryEvents(t *testing.T) {
	// values[1]/values[0]=2 → ClassNormal
	// values[2]/values[1]=0 → ClassNormal (ratio=0; prev becomes 0)
	// computeRatio(values[3], 0.0) → ClassBoundaryZero (prev near-zero), excluded
	// values[4]/values[3]=2 → ClassNormal
	values := []float64{1.0, 2.0, 0.0, 3.0, 6.0}
	ratios := codec.ExtractNormalRatios(values, codec.DriftReanchor, 256)
	if len(ratios) != 3 {
		t.Fatalf("expected 3 ratios, got %d: %v", len(ratios), ratios)
	}
	if math.Abs(ratios[0]-2.0) > 1e-9 {
		t.Errorf("ratio[0]=%v, want ~2.0", ratios[0])
	}
	if ratios[1] != 0.0 {
		t.Errorf("ratio[1]=%v, want 0.0 (zero-value ClassNormal)", ratios[1])
	}
	if math.Abs(ratios[2]-2.0) > 1e-9 {
		t.Errorf("ratio[2]=%v, want ~2.0", ratios[2])
	}
}

// TestExtractNormalRatiosModeB verifies Mode B (DriftCompensate) path is exercised.
func TestExtractNormalRatiosModeB(t *testing.T) {
	values := make([]float64, 500)
	v := 100.0
	for i := range values {
		v *= 1.001
		values[i] = v
	}
	ratiosA := codec.ExtractNormalRatios(values, codec.DriftReanchor, 256)
	ratiosB := codec.ExtractNormalRatios(values, codec.DriftCompensate, 256)
	if len(ratiosA) != len(ratiosB) {
		t.Errorf("mode A returned %d ratios, mode B returned %d", len(ratiosA), len(ratiosB))
	}
}

// TestAnalyzeDriftInvalidInterval verifies interval=0 produces no rows.
func TestAnalyzeDriftInvalidInterval(t *testing.T) {
	values := []float64{1.0, 2.0, 3.0, 4.0}
	rpt := codec.AnalyzeDrift(values, []int{0, -5})
	if len(rpt.Rows) != 0 {
		t.Errorf("expected 0 rows for invalid intervals, got %d", len(rpt.Rows))
	}
}

// TestAnalyzePrecisionTierCeiling verifies that RecommendedBits is always
// rounded up to the ceiling of its payload tier (8, 16, or 30), so that
// callers get the maximum free precision within the same byte cost.
func TestAnalyzePrecisionTierCeiling(t *testing.T) {
	// Any data should produce a RecommendedBits at a tier boundary.
	datasets := []struct {
		name   string
		ratios []float64
	}{
		{"constant", func() []float64 {
			r := make([]float64, 1000)
			for i := range r {
				r[i] = 1.0
			}
			return r
		}()},
		{"smooth", makeLogNormalRatios(5000, 0.01)},
		{"wide", makeLogNormalRatios(5000, 2.0)},
	}

	validCeilings := map[int]bool{8: true, 16: true, 30: true}
	for _, ds := range datasets {
		rpt := codec.AnalyzePrecision(ds.ratios)
		if !validCeilings[rpt.RecommendedBits] {
			t.Errorf("%s: RecommendedBits=%d is not a tier ceiling (want 8, 16, or 30)",
				ds.name, rpt.RecommendedBits)
		}
		// RecommendedSigFigs must be consistent with the (possibly elevated) bits.
		wantSF := codec.BitsToSigFigs(rpt.RecommendedBits)
		if rpt.RecommendedSigFigs != wantSF {
			t.Errorf("%s: RecommendedSigFigs=%d, want %d (from %d bits)",
				ds.name, rpt.RecommendedSigFigs, wantSF, rpt.RecommendedBits)
		}
	}
}

// TestAnalyzePrecisionIdentityFraction verifies IdentityFraction is in [0,1]
// and is near 1.0 for constant data and near 0 for highly variable data.
func TestAnalyzePrecisionIdentityFraction(t *testing.T) {
	// Constant data: all ratios == 1.0 → all within IdentityEpsilon.
	constant := make([]float64, 1000)
	for i := range constant {
		constant[i] = 1.0
	}
	rptConst := codec.AnalyzePrecision(constant)
	if rptConst.IdentityFraction < 0.99 {
		t.Errorf("constant data: IdentityFraction=%.4f, want ~1.0", rptConst.IdentityFraction)
	}

	// Highly variable data: very few ratios within 1e-9 of 1.0.
	variable := makeLogNormalRatios(5000, 1.0)
	rptVar := codec.AnalyzePrecision(variable)
	if rptVar.IdentityFraction < 0 || rptVar.IdentityFraction > 1 {
		t.Errorf("variable data: IdentityFraction=%.4f out of [0,1]", rptVar.IdentityFraction)
	}
	// With σ=1.0 log-normal variation, near-zero identity fraction is expected.
	if rptVar.IdentityFraction > 0.1 {
		t.Errorf("variable data: IdentityFraction=%.4f, want < 0.1 for high-variation data",
			rptVar.IdentityFraction)
	}
}

// TestAnalyzeTiersEmpty verifies zero-count behaviour.
func TestAnalyzeTiersEmpty(t *testing.T) {
	row := codec.AnalyzeTiers(nil, 16, 1e-4)
	if row.Total != 0 || row.U16 != 0 || row.U32 != 0 || row.F64 != 0 {
		t.Errorf("expected all-zero TierRow for nil input, got %+v", row)
	}
	if eff := row.EffectiveBytesPerRatio(); eff != 0 {
		t.Errorf("EffectiveBytesPerRatio: got %v want 0 for empty row", eff)
	}
}

// TestAnalyzeTiersTolZero verifies that tol=0 sends everything to F64.
func TestAnalyzeTiersTolZero(t *testing.T) {
	ratios := []float64{1.001, 1.002, 1.003, 0.999, 0.998}
	row := codec.AnalyzeTiers(ratios, 16, 0)
	if row.Total != 5 || row.F64 != 5 || row.U16 != 0 || row.U32 != 0 {
		t.Errorf("tol=0: expected all F64, got %+v", row)
	}
}

// TestAnalyzeTiersHighTol verifies that a loose tolerance puts all normal
// ratios into the u16 bucket (fast path should fire or all pass relErr check).
func TestAnalyzeTiersHighTol(t *testing.T) {
	// Ratios very close to 1 — easily within 1% tolerance at 16-bit precision.
	ratios := make([]float64, 100)
	for i := range ratios {
		ratios[i] = 1.0 + float64(i+1)*1e-5
	}
	row := codec.AnalyzeTiers(ratios, 16, 1e-2)
	if row.U16 != len(ratios) {
		t.Errorf("high tol: expected all U16 (%d), got u16=%d u32=%d f64=%d",
			len(ratios), row.U16, row.U32, row.F64)
	}
}

// TestAnalyzeTiersTightTol verifies that a very tight tolerance forces the
// u32 mid-tier for ratios that 16-bit precision cannot satisfy.
func TestAnalyzeTiersTightTol(t *testing.T) {
	// Ratios that are ClassNormal but need more than 16-bit precision to stay
	// within ε=1e-6. Build them so they're clearly in range but spaced finely.
	ratios := make([]float64, 50)
	for i := range ratios {
		ratios[i] = 1.0 + float64(i+1)*1e-4
	}
	row := codec.AnalyzeTiers(ratios, 16, 1e-6)
	// With ε=1e-6 and 16 bits, delta16 ≈ 2^(4/65536)−1 ≈ 4.3e-5 >> 1e-6 so
	// fastPath is off. Most ratios will need u32 or exact.
	if row.U32+row.F64 == 0 {
		t.Errorf("tight tol: expected some U32/F64 symbols, all went to U16: %+v", row)
	}
	if row.Total != len(ratios) {
		t.Errorf("Total mismatch: got %d want %d", row.Total, len(ratios))
	}
}

// TestAnalyzeTiersCountsAdd verifies U16+U32+F64 == Total always.
func TestAnalyzeTiersCountsAdd(t *testing.T) {
	ratios := make([]float64, 200)
	for i := range ratios {
		ratios[i] = 1.0 + float64(i+1)*1e-5
	}
	for _, tol := range []float64{1e-2, 1e-4, 1e-6} {
		row := codec.AnalyzeTiers(ratios, 16, tol)
		if row.U16+row.U32+row.F64 != row.Total {
			t.Errorf("tol=%g: counts %d+%d+%d != Total %d", tol, row.U16, row.U32, row.F64, row.Total)
		}
	}
}

// TestAnalyzeTiersEffectiveBytesRange verifies the effective bytes/ratio is
// between 2 (all u16) and 8 (all f64).
func TestAnalyzeTiersEffectiveBytesRange(t *testing.T) {
	ratios := make([]float64, 100)
	for i := range ratios {
		ratios[i] = 1.0 + float64(i+1)*1e-4
	}
	for _, tol := range []float64{1e-2, 1e-4, 1e-7} {
		row := codec.AnalyzeTiers(ratios, 16, tol)
		eff := row.EffectiveBytesPerRatio()
		if eff < 2.0 || eff > 8.0 {
			t.Errorf("tol=%g: EffectiveBytesPerRatio=%v out of [2,8]", tol, eff)
		}
	}
}

// --- AnalyzeTiersV8 ---

// TestAnalyzeTiersV8Empty verifies zero-value report for nil/empty input.
func TestAnalyzeTiersV8Empty(t *testing.T) {
	row := codec.AnalyzeTiersV8(nil, 16, 1e-4)
	if row.Total != 0 || row.U8 != 0 || row.U16 != 0 || row.U24 != 0 || row.U32 != 0 || row.F64 != 0 {
		t.Errorf("expected all-zero row for nil input, got %+v", row)
	}
	if row.EffectiveBytesPerRatio() != 0 {
		t.Errorf("EffectiveBytesPerRatio on empty row should be 0")
	}
}

// TestAnalyzeTiersV8TolZero verifies that tol==0 routes everything to F64.
func TestAnalyzeTiersV8TolZero(t *testing.T) {
	ratios := []float64{1.001, 1.01, 1.1}
	row := codec.AnalyzeTiersV8(ratios, 16, 0)
	if row.F64 != 3 || row.U8+row.U16+row.U24+row.U32 != 0 {
		t.Errorf("tol=0: expected all F64, got %+v", row)
	}
}

// TestAnalyzeTiersV8CountsAddUp verifies U8+U16+U24+U32+F64 == Total.
func TestAnalyzeTiersV8CountsAddUp(t *testing.T) {
	ratios := make([]float64, 200)
	for i := range ratios {
		ratios[i] = 1.0 + float64(i+1)*2e-4
	}
	row := codec.AnalyzeTiersV8(ratios, 16, math.MaxFloat64)
	got := row.U8 + row.U16 + row.U24 + row.U32 + row.F64
	if got != row.Total {
		t.Errorf("counts sum %d != Total %d", got, row.Total)
	}
}

// TestAnalyzeTiersV8SmoothDataU8 verifies that near-identity ratios land in
// U8 at B=16 with fastPath (tol=MaxFloat64) — mirroring encoder behaviour.
func TestAnalyzeTiersV8SmoothDataU8(t *testing.T) {
	// Ratios of 1±1e-5 produce very small signed offsets at B=16 → U8.
	ratios := make([]float64, 500)
	for i := range ratios {
		ratios[i] = 1.0 + float64(i%3-1)*1e-5
	}
	row := codec.AnalyzeTiersV8(ratios, 16, math.MaxFloat64)
	if row.U8 == 0 {
		t.Errorf("expected U8 ratios for near-identity input, got %+v", row)
	}
	if row.F64 != 0 {
		t.Errorf("fastPath should prevent F64 on smooth data, got %d F64", row.F64)
	}
}

// TestAnalyzeTiersV8HighPrecision verifies that bits=24 is not capped and
// routes correctly — U8 for tiny steps, no U16/U24/U32 overflow for smooth data.
func TestAnalyzeTiersV8HighPrecision(t *testing.T) {
	ratios := make([]float64, 100)
	for i := range ratios {
		ratios[i] = 1.0 + float64(i+1)*1e-7
	}
	row := codec.AnalyzeTiersV8(ratios, 24, math.MaxFloat64)
	if row.Bits != 24 {
		t.Errorf("Bits should be 24, got %d", row.Bits)
	}
	total := row.U8 + row.U16 + row.U24 + row.U32 + row.F64
	if total != row.Total {
		t.Errorf("counts sum %d != Total %d", total, row.Total)
	}
}

// TestAnalyzeTiersV8FastPathVsPerRatio verifies that fastPath (tol=MaxFloat64)
// never produces F64, while a tight tol may produce F64 for large offsets.
func TestAnalyzeTiersV8FastPathVsPerRatio(t *testing.T) {
	// Large ratio step — will fail tight per-ratio check at low bits.
	ratios := []float64{8.0} // ratio = 8x, large offset at any precision
	rowFast := codec.AnalyzeTiersV8(ratios, 8, math.MaxFloat64)
	if rowFast.F64 != 0 {
		t.Errorf("fastPath (MaxFloat64) should never produce F64, got %+v", rowFast)
	}
	// With very tight tol that ε_max(8) cannot satisfy: fastPath won't fire.
	rowTight := codec.AnalyzeTiersV8(ratios, 8, 1e-15)
	if rowTight.F64 == 0 {
		t.Errorf("tight tol with low bits should produce F64 for large ratio, got %+v", rowTight)
	}
}

// TestAnalyzeTiersV8EffectiveBytesRange verifies EffectiveBytesPerRatio is in [1,8].
func TestAnalyzeTiersV8EffectiveBytesRange(t *testing.T) {
	ratios := make([]float64, 100)
	for i := range ratios {
		ratios[i] = 1.0 + float64(i+1)*1e-4
	}
	for _, tol := range []float64{math.MaxFloat64, 1e-4, 1e-7} {
		row := codec.AnalyzeTiersV8(ratios, 16, tol)
		eff := row.EffectiveBytesPerRatio()
		if eff < 1.0 || eff > 8.0 {
			t.Errorf("tol=%g: EffectiveBytesPerRatio=%v out of [1,8]", tol, eff)
		}
	}
}
