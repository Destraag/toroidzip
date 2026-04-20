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
// payload tier for the given tolerance ε and u16 precision bits.  The
// classification mirrors gatherRans7 exactly:
//
//   - ratio == 0                        → F64 (cannot be quantized)
//   - fastPath or relErr(u16) < ε       → U16
//   - relErr(u32/30-bit) < ε            → U32
//   - otherwise                         → F64
//
// bits is clamped to [1, 16]; it is the PrecisionBits setting passed to the
// encoder.  tol must be > 0; if tol == 0 every ratio is classified as F64.
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
