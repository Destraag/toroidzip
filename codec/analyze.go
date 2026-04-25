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
		if class != ClassNormal16 {
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
		case ClassNormal16:
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

// DynamicOffsetReport summarises the potential savings from enabling --dynamic-offset
// across all segments in a value stream.
type DynamicOffsetReport struct {
	// TotalSegments is the number of reanchor segments analysed.
	TotalSegments int
	// BenefitSegments is the count of segments where dynamic offset would save bytes.
	BenefitSegments int
	// DefaultPayloadBytes is the total payload bytes with standard k_center.
	DefaultPayloadBytes int
	// OptimalPayloadBytes is the total payload bytes with per-segment optimal k_center.
	// Includes 4 bytes per benefiting segment for storing k_center.
	OptimalPayloadBytes int

	// AvgEffectiveSegLen is the mean number of quantised Q symbols per segment,
	// accounting for adaptive reanchor splits that shorten segments in practice.
	// Short segments (< ~20 symbols) yield little or no dynamic offset benefit.
	AvgEffectiveSegLen float64

	// AvgQOffsetMagnitude is the mean absolute distance of Q symbols from
	// defaultCenter. Large values indicate sustained drift; small values
	// suggest the data is already well-centred and dynamic offset won't help.
	AvgQOffsetMagnitude float64

	// MinSigFigsForBenefit is the smallest sig-fig level (3–9) at which
	// dynamic offset produces net savings on this data at full (256-value)
	// segments. Zero means no benefit was found at any level up to 9.
	// This is computed without AdaptiveReanchor to show the raw potential
	// of the data; actual benefit at lower sig-figs may be zero if adaptive
	// reanchor shortens segments below the useful threshold.
	MinSigFigsForBenefit int

	// ShortSegmentWarning is true when AvgEffectiveSegLen < 20, indicating
	// that adaptive reanchor is fragmenting segments too aggressively for
	// dynamic offset to produce meaningful savings. Consider using dynamic
	// offset only without AdaptiveReanchor, or at higher precision (more
	// sig figs) where reanchors are less frequent.
	ShortSegmentWarning bool
}

// SavedBytes returns the net payload byte reduction from using dynamic offset.
func (r DynamicOffsetReport) SavedBytes() int {
	return r.DefaultPayloadBytes - r.OptimalPayloadBytes
}

// BenefitFraction returns the fraction of segments that benefit from dynamic offset.
func (r DynamicOffsetReport) BenefitFraction() float64 {
	if r.TotalSegments == 0 {
		return 0
	}
	return float64(r.BenefitSegments) / float64(r.TotalSegments)
}

// PayloadReduction returns the fractional reduction in total payload bytes (0–1).
func (r DynamicOffsetReport) PayloadReduction() float64 {
	if r.DefaultPayloadBytes == 0 {
		return 0
	}
	return float64(r.SavedBytes()) / float64(r.DefaultPayloadBytes)
}

// AnalyzeDynamicOffset estimates the payload savings from --dynamic-offset for
// the given value stream at the specified encoding options.
//
// It simulates the actual segment structure the encoder will produce, including
// adaptive reanchor splits when opts.AdaptiveReanchor && opts.EndToEndTolerance > 0.
// This gives accurate predictions rather than optimistic estimates based on
// full-length periodic segments.
//
// The report includes diagnostic fields (AvgEffectiveSegLen, AvgQOffsetMagnitude,
// MinSigFigsForBenefit, ShortSegmentWarning) that explain whether and at what
// precision dynamic offset is worth enabling on this data.
func AnalyzeDynamicOffset(values []float64, opts EncodeOptions) DynamicOffsetReport {
	if opts.ReanchorInterval <= 0 {
		opts.ReanchorInterval = DefaultReanchorInterval
	}
	bits := opts.PrecisionBits
	if bits <= 0 {
		bits = DefaultPrecisionBits
	}
	defaultCenter := int32(1) << (bits - 1)
	tol := opts.Tolerance
	levels := uint32(1) << bits
	deltaB := math.Pow(2, QuantMaxLog2R/float64(levels)) - 1
	fastPath := tol > 0 && deltaB < tol
	ri := opts.ReanchorInterval
	adaptiveReanchor := opts.AdaptiveReanchor && opts.EndToEndTolerance > 0
	endToEndTol := opts.EndToEndTolerance

	var rpt DynamicOffsetReport
	segQSyms := make([]int32, 0, ri)
	var totalQOffsetMag int64
	var totalQSyms int

	analyseSegment := func(seqIdx int) {
		rpt.TotalSegments++
		n := len(segQSyms)
		totalQSyms += n
		for _, q := range segQSyms {
			d := q - defaultCenter
			if d < 0 {
				d = -d
			}
			totalQOffsetMag += int64(d)
		}
		if n == 0 {
			segQSyms = segQSyms[:0]
			return
		}
		defCost := DynOffsetCost(segQSyms, defaultCenter)
		optCenter := OptimalKCenter(segQSyms, bits)
		optCost := DynOffsetCost(segQSyms, optCenter)
		rpt.DefaultPayloadBytes += defCost
		// Segment 0 always uses defaultCenter (decoder invariant).
		if seqIdx > 0 && optCenter != defaultCenter && defCost-optCost > dynOffsetThreshold {
			rpt.BenefitSegments++
			rpt.OptimalPayloadBytes += optCost + dynOffsetThreshold // +4 for k_center storage
		} else {
			rpt.OptimalPayloadBytes += defCost
		}
		segQSyms = segQSyms[:0]
	}

	prev := values[0]
	seqIdx := 0
	for i := 1; i < len(values); i++ {
		if i%ri == 0 {
			analyseSegment(seqIdx)
			seqIdx++
			prev = values[i]
			continue
		}
		ratio, class := computeRatio(values[i], prev)

		switch class {
		case ClassIdentity:
			if adaptiveReanchor && values[i] != 0 &&
				math.Abs(prev-values[i])/math.Abs(values[i]) > endToEndTol {
				analyseSegment(seqIdx)
				seqIdx++
				prev = values[i]
			}
			// prev unchanged for ClassIdentity

		case ClassNormal16:
			defOff := QuantizeRatioOffset(ratio, bits)
			dequant := DequantizeRatioOffset(defOff, bits)
			withinTol := fastPath || deltaB < tol || (ratio != 0 && math.Abs(dequant/ratio-1.0) < tol)

			if adaptiveReanchor && values[i] != 0 &&
				math.Abs(prev*dequant-values[i])/math.Abs(values[i]) > endToEndTol {
				analyseSegment(seqIdx)
				seqIdx++
				prev = values[i]
				continue
			}
			if withinTol {
				segQSyms = append(segQSyms, defOff+defaultCenter)
			}
			prev = prev * dequant

		case ClassBoundaryZero, ClassBoundaryInf:
			prev = values[i]
		}
	}
	analyseSegment(seqIdx)

	// AvgEffectiveSegLen: mean Q symbols per segment.
	if rpt.TotalSegments > 0 {
		rpt.AvgEffectiveSegLen = float64(totalQSyms) / float64(rpt.TotalSegments)
	}
	// AvgQOffsetMagnitude: mean |qSym - defaultCenter|.
	if totalQSyms > 0 {
		rpt.AvgQOffsetMagnitude = float64(totalQOffsetMag) / float64(totalQSyms)
	}
	// ShortSegmentWarning: average effective segment too short for dynamic offset.
	const shortSegThreshold = 20
	rpt.ShortSegmentWarning = rpt.TotalSegments > 0 && rpt.AvgEffectiveSegLen < shortSegThreshold

	// MinSigFigsForBenefit: sweep sf=3..9 (without adaptive reanchor, to show
	// the raw potential of the data independent of reanchor frequency).
	rpt.MinSigFigsForBenefit = dynOffsetMinSigFigs(values, opts)

	return rpt
}

// dynOffsetMinSigFigs sweeps sig-fig levels 3–9 to find the minimum precision
// at which dynamic offset shows net savings on the given data.
// The sweep uses full-length (periodic-only) segments and Tolerance=MaxFloat64
// to isolate the effect of precision bits from reanchor frequency.
// Returns 0 if no benefit is found at any level.
func dynOffsetMinSigFigs(values []float64, baseOpts EncodeOptions) int {
	if baseOpts.ReanchorInterval <= 0 {
		baseOpts.ReanchorInterval = DefaultReanchorInterval
	}
	ri := baseOpts.ReanchorInterval
	for sf := 3; sf <= 9; sf++ {
		bits := SigFigsToBits(sf + 2) // tighter per-ratio precision, same as CLI
		defaultCenter := int32(1) << (bits - 1)

		var benefitFound bool
		segQSyms := make([]int32, 0, ri)
		prev := values[0]

		checkSegment := func() bool {
			if len(segQSyms) == 0 {
				return false
			}
			defCost := DynOffsetCost(segQSyms, defaultCenter)
			opt := OptimalKCenter(segQSyms, bits)
			if opt != defaultCenter && defCost-DynOffsetCost(segQSyms, opt) > dynOffsetThreshold {
				return true
			}
			return false
		}

		for i := 1; i < len(values); i++ {
			if i%ri == 0 {
				if checkSegment() {
					benefitFound = true
					break
				}
				segQSyms = segQSyms[:0]
				prev = values[i]
				continue
			}
			ratio, class := computeRatio(values[i], prev)
			if class == ClassNormal16 && ratio != 0 {
				defOff := QuantizeRatioOffset(ratio, bits)
				dequant := DequantizeRatioOffset(defOff, bits)
				segQSyms = append(segQSyms, defOff+defaultCenter)
				prev = prev * dequant
			} else if class == ClassBoundaryZero || class == ClassBoundaryInf {
				prev = values[i]
			}
		}
		if !benefitFound && checkSegment() {
			benefitFound = true
		}
		if benefitFound {
			return sf
		}
	}
	return 0
}
