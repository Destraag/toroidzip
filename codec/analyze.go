// Package codec — drift analysis helper.
//
// AnalyzeDrift simulates lossless (Mode A / DriftReanchor) and compensated
// (Mode B / DriftCompensate) encoding at each candidate reanchor interval and
// measures the round-trip relative error for the caller's data.  The results
// help the user pick a DriftMode and ReanchorInterval before committing to a
// full encode.
package codec

import "math"

// DriftRow holds the simulation result for one (DriftMode, Interval) pair.
type DriftRow struct {
	Mode           DriftMode
	Interval       int
	MaxRelErr      float64 // worst-case |got−want|/|want| over all ClassNormal values
	MeanRelErr     float64 // mean  |got−want|/|want| over all ClassNormal values
	AnchorOverhead float64 // reanchor events / total values (1.0/Interval)
}

// DriftReport is returned by AnalyzeDrift.
type DriftReport struct {
	Rows []DriftRow

	// RecommendedMode is DriftCompensate when Mode B offers lower error than
	// Mode A at any common interval, otherwise DriftReanchor.
	RecommendedMode DriftMode

	// RecommendedInterval is the largest interval (fewer anchors) where
	// max_err ≤ 10× min_err across all Mode B rows.  Falls back to the
	// smallest tested interval if no rows qualify.
	RecommendedInterval int
}

// AnalyzeDrift simulates lossless encode+decode at each interval in intervals
// for DriftReanchor and DriftCompensate, returning per-row error statistics.
//
// Only ClassNormal values contribute to error statistics; boundary events and
// reanchor events are measured verbatim and are not subject to drift.
//
// Empty input or empty intervals returns a zero-value report.
func AnalyzeDrift(values []float64, intervals []int) DriftReport {
	if len(values) == 0 || len(intervals) == 0 {
		return DriftReport{}
	}

	var rows []DriftRow
	for _, iv := range intervals {
		if iv <= 0 {
			continue
		}
		for _, mode := range []DriftMode{DriftReanchor, DriftCompensate} {
			row := simulateDrift(values, mode, iv)
			rows = append(rows, row)
		}
	}

	if len(rows) == 0 {
		return DriftReport{}
	}

	rpt := DriftReport{Rows: rows}

	// Recommended mode: DriftCompensate if it beats DriftReanchor at any interval.
	modeB := filterRows(rows, DriftCompensate)
	modeA := filterRows(rows, DriftReanchor)
	rpt.RecommendedMode = DriftReanchor
	for i, b := range modeB {
		if i < len(modeA) && b.MaxRelErr < modeA[i].MaxRelErr {
			rpt.RecommendedMode = DriftCompensate
			break
		}
	}

	// Recommended interval: largest Mode B interval with max_err ≤ 10× min_err(Mode B).
	if len(modeB) > 0 {
		minErr := math.MaxFloat64
		for _, r := range modeB {
			if r.MaxRelErr < minErr {
				minErr = r.MaxRelErr
			}
		}
		threshold := 10 * minErr
		rpt.RecommendedInterval = modeB[0].Interval // fallback: smallest
		for _, r := range modeB {
			if r.MaxRelErr <= threshold {
				rpt.RecommendedInterval = r.Interval // keep updating to the largest that qualifies
			}
		}
	} else if len(rows) > 0 {
		rpt.RecommendedInterval = rows[0].Interval
	}

	return rpt
}

// simulateDrift replays the values through the lossless codec path (no
// quantisation — EntropyLossless encoding of ratios) and measures
// round-trip relative error.
func simulateDrift(values []float64, mode DriftMode, interval int) DriftRow {
	row := DriftRow{
		Mode:           mode,
		Interval:       interval,
		AnchorOverhead: 1.0 / float64(interval),
	}
	if len(values) == 0 {
		return row
	}

	var kp kahanProd
	prev := values[0]
	if mode == DriftCompensate {
		kp = newKahanProd(prev)
	}

	var sumErr float64
	var count int

	for i := 1; i < len(values); i++ {
		// Reanchor event: reset accumulator, no error.
		if i%interval == 0 {
			prev = values[i]
			if mode == DriftCompensate {
				kp = newKahanProd(prev)
			}
			continue
		}

		ratio, class := computeRatio(values[i], prev)
		if class != ClassNormal {
			// Boundary or identity — no drift error contribution.
			if class != ClassIdentity {
				prev = values[i]
				if mode == DriftCompensate {
					kp = newKahanProd(prev)
				}
			}
			continue
		}

		// Lossless: ratio is stored exactly as float64 and re-applied.
		var got float64
		if mode == DriftCompensate {
			got = kp.multiply(ratio)
			prev = got // encoder also updates prev to accumulated value
		} else {
			got = prev * ratio
			prev = got
		}

		want := values[i]
		if want != 0 {
			e := math.Abs(got-want) / math.Abs(want)
			sumErr += e
			if e > row.MaxRelErr {
				row.MaxRelErr = e
			}
			count++
		}
	}

	if count > 0 {
		row.MeanRelErr = sumErr / float64(count)
	}
	return row
}

// filterRows returns only the rows matching the given DriftMode.
func filterRows(rows []DriftRow, mode DriftMode) []DriftRow {
	var out []DriftRow
	for _, r := range rows {
		if r.Mode == mode {
			out = append(out, r)
		}
	}
	return out
}

// ExtractNormalRatios returns the ClassNormal ratios from a raw value sequence,
// suitable for passing to AnalyzePrecision.  The reanchorInterval and driftMode
// match those used during encoding so the extracted ratios reflect the actual
// symbols the codec will encounter.
func ExtractNormalRatios(values []float64, driftMode DriftMode, reanchorInterval int) []float64 {
	if len(values) < 2 || reanchorInterval <= 0 {
		return nil
	}
	out := make([]float64, 0, len(values))
	prev := values[0]
	var kp kahanProd
	if driftMode == DriftCompensate {
		kp = newKahanProd(prev)
	}
	for i := 1; i < len(values); i++ {
		if i%reanchorInterval == 0 {
			prev = values[i]
			if driftMode == DriftCompensate {
				kp = newKahanProd(prev)
			}
			continue
		}
		ratio, class := computeRatio(values[i], prev)
		switch class {
		case ClassNormal:
			out = append(out, ratio)
			if driftMode == DriftCompensate {
				prev = kp.multiply(ratio)
			} else {
				prev = prev * ratio
			}
		case ClassIdentity:
			// prev unchanged
		default: // boundary / reanchor events
			prev = values[i]
			if driftMode == DriftCompensate {
				kp = newKahanProd(prev)
			}
		}
	}
	return out
}

// TierRow holds the per-tier count for a single ε value.
type TierRow struct {
	Epsilon float64
	Bits    int // u16 precision bits used for the fast-path check
	Total   int
	U16     int // ClassNormal  — uint16 payload (2 bytes)
	U32     int // ClassNormal32 — uint32 payload (4 bytes)
	F64     int // ClassNormalExact — float64 payload (8 bytes)
}

// EffectiveBytesPerRatio returns the weighted average payload size in bytes.
func (r TierRow) EffectiveBytesPerRatio() float64 {
	if r.Total == 0 {
		return 0
	}
	return float64(r.U16*2+r.U32*4+r.F64*8) / float64(r.Total)
}

// AnalyzeTiers counts how many of the supplied ClassNormal ratios land in each
// payload tier for the given per-ratio tolerance ε and u16 precision bits.
// This function models the pre-v7 (EntropyQuantized / v5 adaptive) two-tier
// scheme: U16 at configured bits, U32 at a hardcoded 30-bit fallback.
//
// Note: the v7 (EntropyAdaptive) encoder uses a four-tier offset-magnitude
// scheme (u8/u16/u24/u32) that this function does NOT model. To inspect v7
// tier distributions, use GatherV7ClassesForTest (export_test.go) or examine
// the class stream directly.
//
//   - ratio == 0                        → F64 (cannot be quantized)
//   - relErr(u16/bits) < ε              → U16
//   - relErr(u32/30-bit) < ε            → U32
//   - otherwise                         → F64
//
// tol is the per-ratio ε; if tol == 0 every ratio is classified as F64.
// bits is clamped to [1, 16].
func AnalyzeTiers(ratios []float64, bits int, tol float64) TierRow {
	if bits < 1 {
		bits = 1
	}
	if bits > 16 {
		bits = 16
	}

	const bits30 = 30
	levels16 := uint32(1) << bits
	delta16 := math.Pow(2, QuantMaxLog2R/float64(levels16)) - 1
	levels30 := uint32(1) << bits30
	delta30 := math.Pow(2, QuantMaxLog2R/float64(levels30)) - 1
	fastPath := tol > 0 && delta16 < tol

	row := TierRow{Epsilon: tol, Bits: bits, Total: len(ratios)}
	for _, ratio := range ratios {
		if tol == 0 || ratio == 0 {
			row.F64++
			continue
		}
		sym16 := QuantizeRatio(ratio, bits)
		dequant16 := DequantizeRatio(sym16, bits)
		if fastPath || math.Abs(dequant16/ratio-1.0) < tol {
			row.U16++
			continue
		}
		sym30 := QuantizeRatio(ratio, bits30)
		dequant30 := DequantizeRatio(sym30, bits30)
		if delta30 < tol || math.Abs(dequant30/ratio-1.0) < tol {
			row.U32++
		} else {
			row.F64++
		}
	}
	return row
}

// TierRowV8 holds the per-tier count for a single (bits, ε) configuration
// as produced by the v8 (AdaptiveV7) encoder's 4-tier offset-magnitude scheme.
type TierRowV8 struct {
	Epsilon float64
	Bits    int // PrecisionBits used for quantisation
	Total   int
	U8      int // ClassNormal8  — int8  payload (1 byte)
	U16     int // ClassNormal   — int16 payload (2 bytes)
	U24     int // ClassNormal24 — int24 payload (3 bytes)
	U32     int // ClassNormal32 — int32 payload (4 bytes)
	F64     int // ClassNormalExact — float64 payload (8 bytes)
}

// EffectiveBytesPerRatio returns the weighted average payload size in bytes
// across all four tiers plus the ClassNormalExact fallback.
func (r TierRowV8) EffectiveBytesPerRatio() float64 {
	if r.Total == 0 {
		return 0
	}
	return float64(r.U8*1+r.U16*2+r.U24*3+r.U32*4+r.F64*8) / float64(r.Total)
}

// AnalyzeTiersV8 counts how many of the supplied ClassNormal ratios land in
// each payload tier for the v8 (AdaptiveV7) encoder.
//
// The routing logic mirrors gatherRans7v7 exactly so that --analyze output
// accurately predicts the tier distribution the encoder will produce:
//
//   - ratio == 0 or tol == 0         → F64 (ClassNormalExact)
//   - fastPath fires (deltaB < tol)  → tier by |offset| magnitude only
//   - per-ratio check fails           → F64 (ClassNormalExact)
//   - |offset| ≤ 127                 → U8
//   - |offset| ≤ 32,767              → U16
//   - |offset| ≤ 8,388,607           → U24
//   - else                           → U32
//
// bits is not capped; it is used as-is (range 1–30) to match the encoder.
// tol is the fast-path gate (Tolerance in EncodeOptions). Pass math.MaxFloat64
// to mirror the --sig-figs fast-path behaviour (fastPath always fires).
func AnalyzeTiersV8(ratios []float64, bits int, tol float64) TierRowV8 {
	if bits < 1 {
		bits = 1
	}
	if bits > 30 {
		bits = 30
	}

	levels := uint32(1) << bits
	deltaB := math.Pow(2, QuantMaxLog2R/float64(levels)) - 1
	fastPath := tol > 0 && deltaB < tol

	row := TierRowV8{Epsilon: tol, Bits: bits, Total: len(ratios)}
	for _, ratio := range ratios {
		if tol == 0 || ratio == 0 {
			row.F64++
			continue
		}
		off := QuantizeRatioOffset(ratio, bits)
		dequant := DequantizeRatioOffset(off, bits)
		if !fastPath && math.Abs(dequant/ratio-1.0) >= tol {
			row.F64++
			continue
		}
		absOff := off
		if absOff < 0 {
			absOff = -absOff
		}
		switch {
		case absOff <= 127:
			row.U8++
		case absOff <= 32767:
			row.U16++
		case absOff <= 8388607:
			row.U24++
		default:
			row.U32++
		}
	}
	return row
}
