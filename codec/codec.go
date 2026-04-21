// Package codec implements the ToroidZip ratio-first compression algorithm.
//
// Encoding overview:
//  1. Write a verbatim anchor value (float64).
//  2. For each subsequent value, compute the ratio r[n] = x[n] / x[n-1].
//  3. Classify the ratio (identity, normal, boundary-zero, boundary-inf).
//  4. Write the ratio stream. Boundary events are stored verbatim inline.
//  5. Every K values, write a new verbatim anchor to bound drift.
//
// Decoding overview:
//  1. Read anchor.
//  2. Reconstruct x[n] = x[n-1] * r[n] for all n.
//  3. Inject boundary events at their recorded positions.
package codec

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// DefaultReanchorInterval is the number of ratio steps between verbatim
// anchor writes. Lower values reduce drift but increase file size.
const DefaultReanchorInterval = 256

// EncodeOptions configures the encoder.
type EncodeOptions struct {
	// ReanchorInterval is the number of ratio steps between verbatim anchors.
	// 0 means use DefaultReanchorInterval.
	ReanchorInterval int

	// DriftMode selects the error-management strategy.
	// 0 means DefaultDriftMode (DriftReanchor).
	DriftMode DriftMode

	// EntropyMode selects the entropy coding layer.
	// 0 means EntropyRaw (Milestone 1 baseline, no compression).
	EntropyMode EntropyMode

	// PrecisionBits is the log-space quantisation depth for EntropyQuantized
	// and EntropyAdaptive. 0 means DefaultPrecisionBits.
	// In EntropyAdaptive (v7) the value is used as-is: it controls both the
	// quantization grid and which payload tier (u8/u16/u24/u32) is selected
	// based on the actual signed-offset magnitude at that bit depth.
	// Use SigFigsToBits(N) to derive B from a sig-figs accuracy target.
	PrecisionBits int

	// Tolerance is the per-ratio fast-path gate for EntropyAdaptive.
	// When the analytical worst-case error at PrecisionBits < Tolerance, the
	// fast path fires and all ClassNormal ratios are quantized at PrecisionBits
	// without an individual per-ratio check. Tier (u8/u16/u24/u32) is then set
	// purely by the signed-offset magnitude.
	// Tolerance = 0 (default): fast path never fires; each ratio checked individually.
	// Tolerance = math.MaxFloat64: fast path always fires ("quantize all normals").
	// To use SigFigsToBits(N) as accuracy driver, set Tolerance = math.MaxFloat64.
	// To use per-ratio error gating, set Tolerance = SigFigsToTolerance(N).
	Tolerance float64

	// AdaptiveReanchor enables drift-triggered reanchoring. When true, the encoder
	// emits a ClassReanchor whenever the reconstructed value would exceed
	// EndToEndTolerance, independently of ReanchorInterval position.
	// ReanchorInterval becomes a safety cap: reanchors still fire at fixed
	// intervals as a fallback. Primarily set by --sig-figs in CLI mode.
	AdaptiveReanchor bool

	// EndToEndTolerance is the maximum acceptable relative reconstruction error
	// for any single output value when AdaptiveReanchor is true. When zero,
	// end-to-end tolerance is not enforced (per-ratio Tolerance still applies).
	EndToEndTolerance float64
}

// Magic bytes identifying a ToroidZip stream.
var magic = [4]byte{'T', 'Z', 'R', 'Z'}

// Stream format version constants.
const (
	versionRaw         byte = 1 // EntropyRaw baseline
	versionLossless    byte = 2 // EntropyLossless
	versionQuantized   byte = 3 // EntropyQuantized v3 (absolute index; kept for decode)
	versionAdaptive    byte = 4 // EntropyAdaptive v4 (superseded; still decoded)
	versionAdaptiveV5  byte = 5 // EntropyAdaptive v5 (tiered: u16 / u32 / float64; superseded; still decoded)
	versionAdaptiveV6  byte = 6 // EntropyAdaptive v6 (tiered: u16 / int32-offset / float64)
	versionQuantizedV2  byte = 7 // EntropyQuantized v4 (offset-based tiers: u8/u16/u24/u32)
	versionAdaptiveV7   byte = 8 // EntropyAdaptive v7 (precision-relative tiers: u8/u16/u24/u32)
)

// Encode compresses values into the ToroidZip format and writes to w.
func Encode(values []float64, w io.Writer, opts EncodeOptions) error {
	if len(values) == 0 {
		return fmt.Errorf("toroidzip: encode: empty input")
	}
	if opts.ReanchorInterval <= 0 {
		opts.ReanchorInterval = DefaultReanchorInterval
	}
	switch opts.EntropyMode {
	case EntropyRaw:
		return encodeRaw(values, w, opts)
	case EntropyLossless:
		return encodeLossless(values, w, opts)
	case EntropyQuantized:
		return encodeQuantizedV2(values, w, opts)
	case EntropyAdaptive:
		return encodeAdaptive(values, w, opts)
	default:
		return fmt.Errorf("toroidzip: encode: unknown entropy mode %d", opts.EntropyMode)
	}
}

// encodeRaw writes the version-1 raw stream.
// Header: magic(4) + version=1(1) + driftMode(1) + reanchorInterval(4) + count(8) = 18 bytes.
// Body: anchor float64, then per-value: class_byte(1) + float64(8).
func encodeRaw(values []float64, w io.Writer, opts EncodeOptions) error {
	ri, dm := opts.ReanchorInterval, opts.DriftMode
	adaptiveReanchor := opts.AdaptiveReanchor && opts.EndToEndTolerance > 0
	endToEndTol := opts.EndToEndTolerance
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, versionRaw); err != nil {
		return err
	}
	if err := writeByte(w, byte(dm)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(ri)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}

	// prev tracks the effective previous value used to compute ratios.
	// Mode A: prev = original input (decoder drifts, bounded by reanchors).
	// Mode B: prev = Kahan-reconstructed value (encoder stays in sync with decoder).
	// Mode C: prev = quantized reconstruction (encoder stays in sync with decoder).
	prev := values[0]
	var kp kahanProd
	if dm == DriftCompensate {
		kp = newKahanProd(prev)
	}
	if err := writeFloat64(w, prev); err != nil {
		return err
	}

	for i := 1; i < len(values); i++ {
		if i%ri == 0 {
			if err := writeByte(w, byte(ClassReanchor)); err != nil {
				return err
			}
			if err := writeFloat64(w, values[i]); err != nil {
				return err
			}
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
			continue
		}

		ratio, class := computeRatio(values[i], prev)
		if dm == DriftQuantize && (class == ClassNormal || class == ClassIdentity) {
			ratio = float64(float32(ratio))
			class = Classify(ratio)
		}

		// Adaptive reanchor: if the reconstructed value would exceed T_end, emit
		// a verbatim ClassReanchor instead. For Mode A (DriftReanchor), prev is
		// the original value so per-step error is negligible; this fires mainly
		// under Mode C (DriftQuantize) where prev tracks the decoder's accumulation.
		if adaptiveReanchor && !IsBoundary(class) && class != ClassReanchor && values[i] != 0 {
			var decodedNext float64
			if class == ClassIdentity {
				decodedNext = prev
			} else {
				decodedNext = prev * ratio
			}
			if math.Abs(decodedNext-values[i])/math.Abs(values[i]) > endToEndTol {
				if err := writeByte(w, byte(ClassReanchor)); err != nil {
					return err
				}
				if err := writeFloat64(w, values[i]); err != nil {
					return err
				}
				prev = values[i]
				if dm == DriftCompensate {
					kp = newKahanProd(prev)
				}
				continue
			}
		}

		if err := writeByte(w, byte(class)); err != nil {
			return err
		}
		if err := writeFloat64(w, ratio); err != nil {
			return err
		}

		switch {
		case class == ClassBoundaryZero || class == ClassBoundaryInf:
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
		case dm == DriftCompensate:
			prev = kp.multiply(ratio)
		case dm == DriftQuantize:
			prev = prev * ratio
		default:
			prev = values[i]
		}
	}
	return nil
}

// gatherRans performs the encoder's first pass, collecting the class stream
// and packed payloads for the lossless/quantized formats.
//
// Payload layout (sequential bytes after the 8-byte anchor float64):
//
//	ClassIdentity:            nothing  — decoder keeps prev unchanged
//	ClassNormal (lossless):   8 bytes float64 (ratio)
//	ClassNormal (quantized):  1/2/4 bytes uint8/uint16/uint32 LE (log-space symbol)
//	                          tier = QuantPayloadTier(bits): 1→bits 1–8, 2→bits 9–16, 4→bits 17–30
//	ClassBoundary*/Reanchor:  8 bytes float64 (verbatim current value)
//
// When opts.AdaptiveReanchor is true and opts.EntropyMode is EntropyQuantized,
// a ClassReanchor is emitted whenever the dequantized reconstruction would
// exceed opts.EndToEndTolerance.
func gatherRans(values []float64, opts EncodeOptions) (classes []byte, payloads []byte) {
	ri, dm := opts.ReanchorInterval, opts.DriftMode
	bits := opts.PrecisionBits
	if bits <= 0 {
		bits = DefaultPrecisionBits
	}

	adaptiveReanchor := opts.AdaptiveReanchor && opts.EndToEndTolerance > 0 &&
		opts.EntropyMode == EntropyQuantized
	endToEndTol := opts.EndToEndTolerance

	payloads = make([]byte, 0, len(values)*4)
	payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[0]))
	classes = make([]byte, 0, len(values)-1)

	prev := values[0]
	var kp kahanProd
	if dm == DriftCompensate {
		kp = newKahanProd(prev)
	}

	for i := 1; i < len(values); i++ {
		if i%ri == 0 {
			classes = append(classes, byte(ClassReanchor))
			payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
			continue
		}

		ratio, class := computeRatio(values[i], prev)
		// DriftQuantize float32-rounds the ratio before storage (any entropy mode).
		if dm == DriftQuantize && (class == ClassNormal || class == ClassIdentity) {
			ratio = float64(float32(ratio))
			class = Classify(ratio)
		}

		// Adaptive reanchor pre-check for quantized ClassNormal: if the dequantized
		// reconstruction would exceed T_end, emit ClassReanchor instead.
		if adaptiveReanchor && class == ClassNormal && values[i] != 0 {
			sym := QuantizeRatio(ratio, bits)
			dequant := DequantizeRatio(sym, bits)
			if math.Abs(prev*dequant-values[i])/math.Abs(values[i]) > endToEndTol {
				classes = append(classes, byte(ClassReanchor))
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
				prev = values[i]
				if dm == DriftCompensate {
					kp = newKahanProd(prev)
				}
				continue
			}
		}

		classes = append(classes, byte(class))
		switch class {
		case ClassIdentity:
			// No payload. Both encoder and decoder leave prev unchanged.
		case ClassNormal:
			if opts.EntropyMode == EntropyQuantized {
				sym := QuantizeRatio(ratio, bits)
				switch QuantPayloadTier(bits) {
				case 1:
					payloads = append(payloads, byte(sym))
				case 2:
					payloads = binary.LittleEndian.AppendUint16(payloads, uint16(sym))
				default: // 4
					payloads = binary.LittleEndian.AppendUint32(payloads, sym)
				}
				ratio = DequantizeRatio(sym, bits) // encoder tracks dequantized ratio
			} else {
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(ratio))
			}
			if dm == DriftCompensate {
				prev = kp.multiply(ratio)
			} else {
				prev = prev * ratio
			}
		case ClassBoundaryZero, ClassBoundaryInf:
			// Store the actual current value verbatim.
			payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
			// ClassReanchor is handled at the top of the loop.
		}
	}
	return classes, payloads
}

// writeRansBody writes ransFreqs(20) + ransLen(4) + ransStream + payloads.
func writeRansBody(w io.Writer, classes []byte, freqs RansFreqs, payloads []byte) error {
	for _, f := range freqs {
		if err := writeUint32(w, f); err != nil {
			return err
		}
	}
	rs := RansEncode(classes, freqs)
	if err := writeUint32(w, uint32(len(rs))); err != nil {
		return err
	}
	if _, err := w.Write(rs); err != nil {
		return err
	}
	_, err := w.Write(payloads)
	return err
}

// encodeLossless writes the version-2 lossless stream.
// Header: magic(4) + version=2(1) + driftMode(1) + reanchorInterval(4) +
//
//	count(8) + ransFreqs(20) = 38 bytes.
//
// Body: [ransLen(4)][ransStream][anchor float64][per-event payloads].
func encodeLossless(values []float64, w io.Writer, opts EncodeOptions) error {
	classes, payloads := gatherRans(values, opts)
	freqs := RansCountFreqs(classes)
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, versionLossless); err != nil {
		return err
	}
	if err := writeByte(w, byte(opts.DriftMode)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(opts.ReanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}
	return writeRansBody(w, classes, freqs, payloads)
}

// encodeQuantized writes the version-3 quantized stream.
// Header: magic(4) + version=3(1) + driftMode(1) + reanchorInterval(4) +
//
//	count(8) + precisionBits(1) + ransFreqs(20) = 39 bytes.
//
// Body: [ransLen(4)][ransStream][anchor float64][per-event payloads].
func encodeQuantized(values []float64, w io.Writer, opts EncodeOptions) error {
	bits := opts.PrecisionBits
	if bits <= 0 {
		bits = DefaultPrecisionBits
	}
	classes, payloads := gatherRans(values, opts)
	freqs := RansCountFreqs(classes)
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, versionQuantized); err != nil {
		return err
	}
	if err := writeByte(w, byte(opts.DriftMode)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(opts.ReanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}
	if err := writeByte(w, byte(bits)); err != nil {
		return err
	}
	return writeRansBody(w, classes, freqs, payloads)
}

// ============================================================
// v4 quantized stream (versionQuantizedV2 = 7)
// Offset-based tier routing: u8 / u16 / u24 / u32 by |offset| magnitude.
// ============================================================

// gatherRansV4 performs the encoder first pass for the quantized v4 stream.
// For each ClassNormal ratio it computes a signed offset from log-space centre:
//
//	off = QuantizeRatioOffset(ratio, bits)
//	|off| ≤ 127      → ClassNormal8  + int8  (1 byte)
//	|off| ≤ 32767    → ClassNormal   + int16 (2 bytes)
//	|off| ≤ 8388607  → ClassNormal24 + int24 (3 bytes)
//	else             → ClassNormal32 + int32 (4 bytes)
//
// Boundary / identity / reanchor events are unchanged from gatherRans.
// Adaptive reanchoring (opts.AdaptiveReanchor) is preserved.
func gatherRansV4(values []float64, opts EncodeOptions) (classes []byte, payloads []byte) {
	ri, dm := opts.ReanchorInterval, opts.DriftMode
	bits := opts.PrecisionBits
	if bits <= 0 {
		bits = DefaultPrecisionBits
	}

	adaptiveReanchor := opts.AdaptiveReanchor && opts.EndToEndTolerance > 0 &&
		opts.EntropyMode == EntropyQuantized
	endToEndTol := opts.EndToEndTolerance

	payloads = make([]byte, 0, len(values)*2)
	payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[0]))
	classes = make([]byte, 0, len(values)-1)

	prev := values[0]
	var kp kahanProd
	if dm == DriftCompensate {
		kp = newKahanProd(prev)
	}

	for i := 1; i < len(values); i++ {
		if i%ri == 0 {
			classes = append(classes, byte(ClassReanchor))
			payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
			continue
		}

		ratio, class := computeRatio(values[i], prev)
		if dm == DriftQuantize && (class == ClassNormal || class == ClassIdentity) {
			ratio = float64(float32(ratio))
			class = Classify(ratio)
		}

		if adaptiveReanchor && class == ClassNormal && values[i] != 0 {
			off := QuantizeRatioOffset(ratio, bits)
			dequant := DequantizeRatioOffset(off, bits)
			if math.Abs(prev*dequant-values[i])/math.Abs(values[i]) > endToEndTol {
				classes = append(classes, byte(ClassReanchor))
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
				prev = values[i]
				if dm == DriftCompensate {
					kp = newKahanProd(prev)
				}
				continue
			}
		}

		switch class {
		case ClassIdentity:
			classes = append(classes, byte(ClassIdentity))
		case ClassNormal:
			off := QuantizeRatioOffset(ratio, bits)
			var absOff int32
			if off < 0 {
				absOff = -off
			} else {
				absOff = off
			}
			switch {
			case absOff <= 127:
				classes = append(classes, byte(ClassNormal8))
				payloads = append(payloads, byte(int8(off)))
			case absOff <= 32767:
				classes = append(classes, byte(ClassNormal))
				payloads = binary.LittleEndian.AppendUint16(payloads, uint16(int16(off)))
			case absOff <= 8388607:
				classes = append(classes, byte(ClassNormal24))
				payloads = append(payloads, byte(off), byte(off>>8), byte(off>>16))
			default:
				classes = append(classes, byte(ClassNormal32))
				payloads = binary.LittleEndian.AppendUint32(payloads, uint32(off))
			}
			ratio = DequantizeRatioOffset(off, bits)
			if dm == DriftCompensate {
				prev = kp.multiply(ratio)
			} else {
				prev = prev * ratio
			}
		case ClassBoundaryZero, ClassBoundaryInf:
			classes = append(classes, byte(class))
			payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
		}
	}
	return classes, payloads
}

// writeRansBody9 writes ransFreqs9(36) + ransLen(4) + ransStream + payloads.
func writeRansBody9(w io.Writer, classes []byte, freqs RansFreqs9, payloads []byte) error {
	for _, f := range freqs {
		if err := writeUint32(w, f); err != nil {
			return err
		}
	}
	rs := RansEncode9(classes, freqs)
	if err := writeUint32(w, uint32(len(rs))); err != nil {
		return err
	}
	if _, err := w.Write(rs); err != nil {
		return err
	}
	_, err := w.Write(payloads)
	return err
}

// encodeQuantizedV2 writes the version-7 quantized stream (offset-based tiers).
// Header: magic(4) + version=7(1) + driftMode(1) + reanchorInterval(4) +
//
//	count(8) + precisionBits(1) + ransFreqs9(36) = 55 bytes.
//
// Body: [ransLen(4)][ransStream][anchor float64][per-event payloads].
func encodeQuantizedV2(values []float64, w io.Writer, opts EncodeOptions) error {
	bits := opts.PrecisionBits
	if bits <= 0 {
		bits = DefaultPrecisionBits
	}
	classes, payloads := gatherRansV4(values, opts)
	freqs := RansCountFreqs9(classes)
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, versionQuantizedV2); err != nil {
		return err
	}
	if err := writeByte(w, byte(opts.DriftMode)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(opts.ReanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}
	if err := writeByte(w, byte(bits)); err != nil {
		return err
	}
	return writeRansBody9(w, classes, freqs, payloads)
}

// readRansFreqs9 reads a 9-element normalised frequency table from r.
func readRansFreqs9(r io.Reader) (RansFreqs9, error) {
	var freqs RansFreqs9
	for i := range freqs {
		f, err := readUint32(r)
		if err != nil {
			return freqs, fmt.Errorf("toroidzip: decode: reading rans9 freq[%d]: %w", i, err)
		}
		freqs[i] = f
	}
	return freqs, nil
}

// decodeQuantizedV2 reads a version-7 quantized stream (magic+version already consumed).
func decodeQuantizedV2(r io.Reader) ([]float64, error) {
	dm, count, err := readCommonRansHeader(r)
	if err != nil {
		return nil, err
	}
	bitsByte, err := readByte(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading precision bits: %w", err)
	}
	freqs, err := readRansFreqs9(r)
	if err != nil {
		return nil, err
	}
	return decodeRansV4(r, dm, count, freqs, int(bitsByte))
}

// decodeRansV4 is the shared body decoder for version-7 quantized streams.
// It reads offset-encoded payloads per class and reconstructs via DequantizeRatioOffset.
func decodeRansV4(r io.Reader, driftMode DriftMode, count uint64, freqs RansFreqs9, bits int) ([]float64, error) {
	ransLen, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading rans length: %w", err)
	}
	ransStream := make([]byte, ransLen)
	if ransLen > 0 {
		if _, err := io.ReadFull(r, ransStream); err != nil {
			return nil, fmt.Errorf("toroidzip: decode: reading rans stream: %w", err)
		}
	}

	classes, err := RansDecode9(ransStream, freqs, int(count)-1)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: rans decode: %w", err)
	}

	payRaw, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading payloads: %w", err)
	}
	pay := bytes.NewReader(payRaw)

	values := make([]float64, count)
	values[0], err = readFloat64(pay)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading anchor: %w", err)
	}

	var kp kahanProd
	if driftMode == DriftCompensate {
		kp = newKahanProd(values[0])
	}

	for i := 1; i < int(count); i++ {
		class := RatioClass(classes[i-1])
		switch class {
		case ClassIdentity:
			values[i] = values[i-1]
		case ClassNormal8:
			b, err := readByte(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading Normal8 at %d: %w", i, err)
			}
			off := int32(int8(b))
			ratio := DequantizeRatioOffset(off, bits)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}
		case ClassNormal:
			s, err := readUint16(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading Normal at %d: %w", i, err)
			}
			off := int32(int16(s))
			ratio := DequantizeRatioOffset(off, bits)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}
		case ClassNormal24:
			var b [3]byte
			if _, err := io.ReadFull(pay, b[:]); err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading Normal24 at %d: %w", i, err)
			}
			raw := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16
			if raw&(1<<23) != 0 {
				raw |= ^int32(0xFFFFFF) // sign-extend from bit 23
			}
			ratio := DequantizeRatioOffset(raw, bits)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}
		case ClassNormal32:
			u, err := readUint32(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading Normal32 at %d: %w", i, err)
			}
			off := int32(u)
			ratio := DequantizeRatioOffset(off, bits)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}
		case ClassBoundaryZero, ClassBoundaryInf, ClassReanchor:
			val, err := readFloat64(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading verbatim at %d: %w", i, err)
			}
			values[i] = val
			if driftMode == DriftCompensate {
				kp = newKahanProd(val)
			}
		default:
			return nil, fmt.Errorf("toroidzip: decode: unknown class %d at %d", class, i)
		}
	}
	return values, nil
}

// Decode decompresses a ToroidZip stream from r and returns the original values.
func Decode(r io.Reader) ([]float64, error) {
	var hdrMagic [4]byte
	if _, err := io.ReadFull(r, hdrMagic[:]); err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading magic: %w", err)
	}
	if hdrMagic != magic {
		return nil, fmt.Errorf("toroidzip: decode: invalid magic bytes")
	}
	ver, err := readByte(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading version: %w", err)
	}
	switch ver {
	case versionRaw:
		return decodeV1(r)
	case versionLossless:
		return decodeV2(r)
	case versionQuantized:
		return decodeV3(r)
	case versionAdaptive:
		return decodeV4(r)
	case versionAdaptiveV5:
		return decodeV5(r)
	case versionAdaptiveV6:
		return decodeV6(r)
	case versionQuantizedV2:
		return decodeQuantizedV2(r)
	case versionAdaptiveV7:
		return decodeV7(r)
	default:
		return nil, fmt.Errorf("toroidzip: decode: unsupported version %d", ver)
	}
}

// decodeV1 reads a version-1 raw stream (magic+version already consumed).
func decodeV1(r io.Reader) ([]float64, error) {
	driftModeByte, err := readByte(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading drift mode: %w", err)
	}
	driftMode := DriftMode(driftModeByte)

	if _, err = readUint32(r); err != nil { // reanchorInterval unused by decoder
		return nil, fmt.Errorf("toroidzip: decode: reading reanchor interval: %w", err)
	}

	count, err := readUint64(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading count: %w", err)
	}
	if count == 0 {
		return nil, fmt.Errorf("toroidzip: decode: zero count")
	}

	values := make([]float64, count)
	values[0], err = readFloat64(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading anchor: %w", err)
	}

	var kp kahanProd
	if driftMode == DriftCompensate {
		kp = newKahanProd(values[0])
	}

	for i := uint64(1); i < count; i++ {
		classByte, err := readByte(r)
		if err != nil {
			return nil, fmt.Errorf("toroidzip: decode: reading class at %d: %w", i, err)
		}
		val, err := readFloat64(r)
		if err != nil {
			return nil, fmt.Errorf("toroidzip: decode: reading value at %d: %w", i, err)
		}
		class := RatioClass(classByte)
		if class == ClassReanchor || IsBoundary(class) {
			values[i] = val
			if driftMode == DriftCompensate {
				kp = newKahanProd(val)
			}
		} else if driftMode == DriftCompensate {
			values[i] = kp.multiply(val)
		} else {
			values[i] = values[i-1] * val
		}
	}
	return values, nil
}

// readCommonRansHeader reads driftMode(1) + reanchorInterval(4) + count(8)
// from a v2/v3 stream (magic and version already consumed).
func readCommonRansHeader(r io.Reader) (driftMode DriftMode, count uint64, err error) {
	var dmb byte
	dmb, err = readByte(r)
	if err != nil {
		err = fmt.Errorf("toroidzip: decode: reading drift mode: %w", err)
		return
	}
	driftMode = DriftMode(dmb)
	if _, err = readUint32(r); err != nil {
		err = fmt.Errorf("toroidzip: decode: reading reanchor interval: %w", err)
		return
	}
	count, err = readUint64(r)
	if err != nil {
		err = fmt.Errorf("toroidzip: decode: reading count: %w", err)
		return
	}
	if count == 0 {
		err = fmt.Errorf("toroidzip: decode: zero count")
	}
	return
}

// readRansFreqs reads the 5-entry frequency table (20 bytes).
func readRansFreqs(r io.Reader) (RansFreqs, error) {
	var freqs RansFreqs
	for i := range freqs {
		f, err := readUint32(r)
		if err != nil {
			return freqs, fmt.Errorf("toroidzip: decode: reading rans freqs: %w", err)
		}
		freqs[i] = f
	}
	return freqs, nil
}

// decodeV2 reads a version-2 lossless stream (magic+version already consumed).
func decodeV2(r io.Reader) ([]float64, error) {
	dm, count, err := readCommonRansHeader(r)
	if err != nil {
		return nil, err
	}
	freqs, err := readRansFreqs(r)
	if err != nil {
		return nil, err
	}
	return decodeRans(r, dm, count, freqs, 0)
}

// decodeV3 reads a version-3 quantized stream (magic+version already consumed).
func decodeV3(r io.Reader) ([]float64, error) {
	dm, count, err := readCommonRansHeader(r)
	if err != nil {
		return nil, err
	}
	bitsByte, err := readByte(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading precision bits: %w", err)
	}
	freqs, err := readRansFreqs(r)
	if err != nil {
		return nil, err
	}
	return decodeRans(r, dm, count, freqs, int(bitsByte))
}

// decodeRans is the shared body decoder for v2 (bits=0) and v3 (bits>0).
// Payload conventions mirror gatherRans — see that function's doc comment.
func decodeRans(r io.Reader, driftMode DriftMode, count uint64, freqs RansFreqs, bits int) ([]float64, error) {
	// Read rANS stream.
	ransLen, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading rans length: %w", err)
	}
	ransStream := make([]byte, ransLen)
	if ransLen > 0 {
		if _, err := io.ReadFull(r, ransStream); err != nil {
			return nil, fmt.Errorf("toroidzip: decode: reading rans stream: %w", err)
		}
	}

	// Decode the class stream for indices [1, count).
	classes, err := RansDecode(ransStream, freqs, int(count)-1)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: rans decode: %w", err)
	}

	// Slurp remaining bytes as the packed payload section.
	payRaw, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading payloads: %w", err)
	}
	pay := bytes.NewReader(payRaw)

	values := make([]float64, count)
	values[0], err = readFloat64(pay)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading anchor: %w", err)
	}

	var kp kahanProd
	if driftMode == DriftCompensate {
		kp = newKahanProd(values[0])
	}

	for i := 1; i < int(count); i++ {
		class := RatioClass(classes[i-1])
		switch class {
		case ClassIdentity:
			// No payload; decoder keeps prev unchanged (same as encoder).
			values[i] = values[i-1]
		case ClassNormal:
			var ratio float64
			if bits > 0 {
				var sym uint32
				switch QuantPayloadTier(bits) {
				case 1:
					b, err := readByte(pay)
					if err != nil {
						return nil, fmt.Errorf("toroidzip: decode: reading quantized symbol at %d: %w", i, err)
					}
					sym = uint32(b)
				case 2:
					s, err := readUint16(pay)
					if err != nil {
						return nil, fmt.Errorf("toroidzip: decode: reading quantized symbol at %d: %w", i, err)
					}
					sym = uint32(s)
				default: // 4
					s, err := readUint32(pay)
					if err != nil {
						return nil, fmt.Errorf("toroidzip: decode: reading quantized symbol at %d: %w", i, err)
					}
					sym = s
				}
				ratio = DequantizeRatio(sym, bits)
			} else {
				ratio, err = readFloat64(pay)
				if err != nil {
					return nil, fmt.Errorf("toroidzip: decode: reading normal ratio at %d: %w", i, err)
				}
			}
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}
		case ClassBoundaryZero, ClassBoundaryInf, ClassReanchor:
			val, err := readFloat64(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading verbatim at %d: %w", i, err)
			}
			values[i] = val
			if driftMode == DriftCompensate {
				kp = newKahanProd(val)
			}
		default:
			return nil, fmt.Errorf("toroidzip: decode: unknown class %d at %d", class, i)
		}
	}
	return values, nil
}

// ============================================================
// v4 adaptive stream (EntropyAdaptive)
// ============================================================

// encodeAdaptive writes the current default adaptive stream (v7).
// v4 (6-symbol) encoder functions (gatherRans6, writeRansBody6) were
// removed in M8/8a — they were dead code (no callers). The v4 decode
// path (decodeV4, decodeRans6, readRansFreqs6) is kept for legacy streams.
func encodeAdaptive(values []float64, w io.Writer, opts EncodeOptions) error {
	return encodeAdaptiveV7(values, w, opts)
}

// gatherRans7v6 is the encoder first pass for the v6 EntropyAdaptive stream.
// In v6 all quantized ratios use 30-bit signed-offset encoding:
//
//   - fastPath (delta30 < tol): all ratios quantized at 30-bit; no per-ratio check
//   - int16 offset fits (abs(off30) ≤ 2^15-1): ClassNormal + int16 (2 bytes)
//   - int32 offset needed:                     ClassNormal32 + int32 (4 bytes)
//   - 30-bit insufficient or ratio==0:          ClassNormalExact + float64 (8 bytes)
//
// The ClassNormal u16 payload is now an int16 signed offset decoded via
// DequantizeRatioOffset, not an absolute uint16 symbol as in v5.
// This allows smooth-data ratios (ratio ≈ 1.0, small offset) to use 2 bytes
// at full 30-bit precision instead of 4 bytes.
func gatherRans7v6(values []float64, opts EncodeOptions) (classes []byte, payloads []byte) {
	ri, dm := opts.ReanchorInterval, opts.DriftMode
	tol := opts.Tolerance

	const bits30 = 30
	levels30 := uint32(1) << bits30
	delta30 := math.Pow(2, QuantMaxLog2R/float64(levels30)) - 1

	// fastPath: all 30-bit quantizations satisfy ε — skip per-ratio relErr check.
	fastPath := tol > 0 && delta30 < tol

	adaptiveReanchor := opts.AdaptiveReanchor && opts.EndToEndTolerance > 0
	endToEndTol := opts.EndToEndTolerance

	payloads = make([]byte, 0, len(values)*4)
	payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[0]))
	classes = make([]byte, 0, len(values)-1)

	prev := values[0]
	var kp kahanProd
	if dm == DriftCompensate {
		kp = newKahanProd(prev)
	}

	for i := 1; i < len(values); i++ {
		if i%ri == 0 {
			classes = append(classes, byte(ClassReanchor))
			payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
			continue
		}

		ratio, class := computeRatio(values[i], prev)
		if dm == DriftQuantize && (class == ClassNormal || class == ClassIdentity) {
			ratio = float64(float32(ratio))
			class = Classify(ratio)
		}

		switch class {
		case ClassIdentity:
			if adaptiveReanchor && values[i] != 0 &&
				math.Abs(prev-values[i])/math.Abs(values[i]) > endToEndTol {
				classes = append(classes, byte(ClassReanchor))
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
				prev = values[i]
				if dm == DriftCompensate {
					kp = newKahanProd(prev)
				}
				continue
			}
			classes = append(classes, byte(ClassIdentity))

		case ClassNormal:
			var chosenClass RatioClass
			var encodedRatio float64
			if ratio == 0 {
				chosenClass = ClassNormalExact
				encodedRatio = ratio
			} else {
				off30 := QuantizeRatioOffset(ratio, bits30)
				dequant30 := DequantizeRatioOffset(off30, bits30)
				withinTol := fastPath || delta30 < tol || math.Abs(dequant30/ratio-1.0) < tol
				if !withinTol {
					chosenClass = ClassNormalExact
					encodedRatio = ratio
				} else {
					encodedRatio = dequant30
					// Route to u16 tier if offset fits in int16, u32 tier otherwise.
					if off30 >= -(1<<15) && off30 <= (1<<15)-1 {
						chosenClass = ClassNormal
					} else {
						chosenClass = ClassNormal32
					}
				}
			}

			if adaptiveReanchor && values[i] != 0 &&
				math.Abs(prev*encodedRatio-values[i])/math.Abs(values[i]) > endToEndTol {
				classes = append(classes, byte(ClassReanchor))
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
				prev = values[i]
				if dm == DriftCompensate {
					kp = newKahanProd(prev)
				}
				continue
			}

			classes = append(classes, byte(chosenClass))
			switch chosenClass {
			case ClassNormal:
				// v6: int16 signed offset at 30-bit precision.
				payloads = binary.LittleEndian.AppendUint16(payloads, uint16(QuantizeRatioOffset(ratio, bits30)))
			case ClassNormal32:
				// v6: int32 signed offset at 30-bit precision.
				payloads = binary.LittleEndian.AppendUint32(payloads, uint32(QuantizeRatioOffset(ratio, bits30)))
			case ClassNormalExact:
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(ratio))
			}
			if dm == DriftCompensate {
				prev = kp.multiply(encodedRatio)
			} else {
				prev = prev * encodedRatio
			}

		case ClassBoundaryZero, ClassBoundaryInf:
			classes = append(classes, byte(class))
			payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
		}
	}
	return classes, payloads
}

// gatherRans7v7 is the encoder first pass for the v7 EntropyAdaptive stream.
// In v7 all quantized ratios use signed-offset encoding at opts.PrecisionBits:
//
//   - fastPath (deltaB < tol): all ratios quantized at configured bits; no per-ratio check
//   - int8  offset fits (abs(off) ≤ 127):      ClassNormal8  + int8  (1 byte)
//   - int16 offset fits (abs(off) ≤ 32767):    ClassNormal   + int16 (2 bytes)
//   - int24 offset fits (abs(off) ≤ 8388607):  ClassNormal24 + int24 (3 bytes, LE)
//   - int32 offset needed:                      ClassNormal32 + int32 (4 bytes)
//   - configured-bits insufficient or ratio==0: ClassNormalExact + float64 (8 bytes)
//
// Unlike v6 (hardcoded 30-bit), v7 uses opts.PrecisionBits for all offsets.
// Smooth data at N=4–5 (B=13–16) naturally lands in u8; large steps at N=6–7
// (B=20–23) land in u24 rather than paying u32 as in v5/v6.
func gatherRans7v7(values []float64, opts EncodeOptions) (classes []byte, payloads []byte) {
	ri, dm := opts.ReanchorInterval, opts.DriftMode
	tol := opts.Tolerance

	bits := opts.PrecisionBits
	if bits <= 0 {
		bits = DefaultPrecisionBits
	}
	levels := uint32(1) << bits
	deltaB := math.Pow(2, QuantMaxLog2R/float64(levels)) - 1

	// fastPath: all configured-precision quantizations satisfy ε — skip per-ratio relErr check.
	fastPath := tol > 0 && deltaB < tol

	adaptiveReanchor := opts.AdaptiveReanchor && opts.EndToEndTolerance > 0
	endToEndTol := opts.EndToEndTolerance

	payloads = make([]byte, 0, len(values)*2)
	payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[0]))
	classes = make([]byte, 0, len(values)-1)

	prev := values[0]
	var kp kahanProd
	if dm == DriftCompensate {
		kp = newKahanProd(prev)
	}

	for i := 1; i < len(values); i++ {
		if i%ri == 0 {
			classes = append(classes, byte(ClassReanchor))
			payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
			continue
		}

		ratio, class := computeRatio(values[i], prev)
		if dm == DriftQuantize && (class == ClassNormal || class == ClassIdentity) {
			ratio = float64(float32(ratio))
			class = Classify(ratio)
		}

		switch class {
		case ClassIdentity:
			if adaptiveReanchor && values[i] != 0 &&
				math.Abs(prev-values[i])/math.Abs(values[i]) > endToEndTol {
				classes = append(classes, byte(ClassReanchor))
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
				prev = values[i]
				if dm == DriftCompensate {
					kp = newKahanProd(prev)
				}
				continue
			}
			classes = append(classes, byte(ClassIdentity))

		case ClassNormal:
			var chosenClass RatioClass
			var encodedRatio float64
			var off int32
			if ratio == 0 {
				chosenClass = ClassNormalExact
				encodedRatio = ratio
			} else {
				off = QuantizeRatioOffset(ratio, bits)
				dequant := DequantizeRatioOffset(off, bits)
				withinTol := fastPath || deltaB < tol || math.Abs(dequant/ratio-1.0) < tol
				if !withinTol {
					chosenClass = ClassNormalExact
					encodedRatio = ratio
				} else {
					encodedRatio = dequant
					absOff := off
					if absOff < 0 {
						absOff = -absOff
					}
					switch {
					case absOff <= 127:
						chosenClass = ClassNormal8
					case absOff <= 32767:
						chosenClass = ClassNormal
					case absOff <= 8388607:
						chosenClass = ClassNormal24
					default:
						chosenClass = ClassNormal32
					}
				}
			}

			if adaptiveReanchor && values[i] != 0 &&
				math.Abs(prev*encodedRatio-values[i])/math.Abs(values[i]) > endToEndTol {
				classes = append(classes, byte(ClassReanchor))
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
				prev = values[i]
				if dm == DriftCompensate {
					kp = newKahanProd(prev)
				}
				continue
			}

			classes = append(classes, byte(chosenClass))
			switch chosenClass {
			case ClassNormal8:
				payloads = append(payloads, byte(int8(off)))
			case ClassNormal:
				payloads = binary.LittleEndian.AppendUint16(payloads, uint16(int16(off)))
			case ClassNormal24:
				payloads = append(payloads, byte(off), byte(off>>8), byte(off>>16))
			case ClassNormal32:
				payloads = binary.LittleEndian.AppendUint32(payloads, uint32(off))
			case ClassNormalExact:
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(ratio))
			}
			if dm == DriftCompensate {
				prev = kp.multiply(encodedRatio)
			} else {
				prev = prev * encodedRatio
			}

		case ClassBoundaryZero, ClassBoundaryInf:
			classes = append(classes, byte(class))
			payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
		}
	}
	return classes, payloads
}

// gatherRans7 is the encoder first pass for the v5 EntropyAdaptive stream.
// For each ClassNormal ratio it picks the smallest tier that satisfies ε:
//   - fastPath (delta16 < tol): all ClassNormal + uint16 (no per-ratio check)
//   - relErr < tol at 16 bits:  ClassNormal   + uint16  (2 bytes)
//   - relErr < tol at 30 bits:  ClassNormal32 + uint32  (4 bytes)
//   - relErr >= tol at 30 bits: ClassNormalExact + float64 (8 bytes)
//
// ratio==0 always takes the exact path (cannot be represented by any symbol).
//
// When opts.AdaptiveReanchor is true, a ClassReanchor is emitted whenever the
// quantized reconstruction of a value would exceed opts.EndToEndTolerance.
// opts.ReanchorInterval remains a safety cap that also triggers reanchors.
func gatherRans7(values []float64, opts EncodeOptions) (classes []byte, payloads []byte) {
	ri, dm := opts.ReanchorInterval, opts.DriftMode
	bits := opts.PrecisionBits
	if bits <= 0 || bits > 16 {
		bits = 16
	}
	tol := opts.Tolerance

	// Pre-compute analytical worst-case error at 16-bit and 30-bit precision.
	// delta = 2^(QuantMaxLog2R/levels) - 1  (half bucket-width in linear space).
	levels16 := uint32(1) << bits
	delta16 := math.Pow(2, QuantMaxLog2R/float64(levels16)) - 1
	const bits30 = 30
	levels30 := uint32(1) << bits30
	delta30 := math.Pow(2, QuantMaxLog2R/float64(levels30)) - 1

	// fastPath: skip all per-ratio checks when every 16-bit quantization is
	// guaranteed to satisfy ε analytically.
	fastPath := tol > 0 && delta16 < tol

	adaptiveReanchor := opts.AdaptiveReanchor && opts.EndToEndTolerance > 0
	endToEndTol := opts.EndToEndTolerance

	payloads = make([]byte, 0, len(values)*4)
	payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[0]))
	classes = make([]byte, 0, len(values)-1)

	prev := values[0]
	var kp kahanProd
	if dm == DriftCompensate {
		kp = newKahanProd(prev)
	}

	for i := 1; i < len(values); i++ {
		if i%ri == 0 {
			classes = append(classes, byte(ClassReanchor))
			payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
			continue
		}

		ratio, class := computeRatio(values[i], prev)
		if dm == DriftQuantize && (class == ClassNormal || class == ClassIdentity) {
			ratio = float64(float32(ratio))
			class = Classify(ratio)
		}

		switch class {
		case ClassIdentity:
			// Decoder keeps prev unchanged; decoded value for this position is prev.
			if adaptiveReanchor && values[i] != 0 &&
				math.Abs(prev-values[i])/math.Abs(values[i]) > endToEndTol {
				classes = append(classes, byte(ClassReanchor))
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
				prev = values[i]
				if dm == DriftCompensate {
					kp = newKahanProd(prev)
				}
				continue
			}
			classes = append(classes, byte(ClassIdentity))

		case ClassNormal:
			// Phase 1: pick the smallest tier that satisfies the per-ratio ε.
			sym16 := QuantizeRatio(ratio, bits)
			dequant16 := DequantizeRatio(sym16, bits)
			var sym30 uint32
			var encodedRatio float64
			var chosenClass RatioClass
			if ratio == 0 {
				encodedRatio = ratio
				chosenClass = ClassNormalExact
			} else if fastPath || math.Abs(dequant16/ratio-1.0) < tol {
				encodedRatio = dequant16
				chosenClass = ClassNormal
			} else {
				sym30 = QuantizeRatio(ratio, bits30)
				dequant30 := DequantizeRatio(sym30, bits30)
				if delta30 < tol || math.Abs(dequant30/ratio-1.0) < tol {
					encodedRatio = dequant30
					chosenClass = ClassNormal32
				} else {
					encodedRatio = ratio
					chosenClass = ClassNormalExact
				}
			}

			// Phase 2: adaptive end-to-end reanchor check.
			// If the quantized reconstruction would exceed T_end, emit a verbatim
			// ClassReanchor instead (resets accumulated drift to zero).
			if adaptiveReanchor && values[i] != 0 &&
				math.Abs(prev*encodedRatio-values[i])/math.Abs(values[i]) > endToEndTol {
				classes = append(classes, byte(ClassReanchor))
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
				prev = values[i]
				if dm == DriftCompensate {
					kp = newKahanProd(prev)
				}
				continue
			}

			// Phase 3: emit chosen tier.
			classes = append(classes, byte(chosenClass))
			switch chosenClass {
			case ClassNormal:
				payloads = binary.LittleEndian.AppendUint16(payloads, uint16(sym16))
			case ClassNormal32:
				payloads = binary.LittleEndian.AppendUint32(payloads, sym30)
			case ClassNormalExact:
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(ratio))
			}
			if dm == DriftCompensate {
				prev = kp.multiply(encodedRatio)
			} else {
				prev = prev * encodedRatio
			}

		case ClassBoundaryZero, ClassBoundaryInf:
			classes = append(classes, byte(class))
			payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(values[i]))
			prev = values[i]
			if dm == DriftCompensate {
				kp = newKahanProd(prev)
			}
		}
	}
	return classes, payloads
}

// writeRansBody7 writes ransFreqs7(28) + ransLen(4) + ransStream + payloads.
func writeRansBody7(w io.Writer, classes []byte, freqs RansFreqs7, payloads []byte) error {
	for _, f := range freqs {
		if err := writeUint32(w, f); err != nil {
			return err
		}
	}
	rs := RansEncode7(classes, freqs)
	if err := writeUint32(w, uint32(len(rs))); err != nil {
		return err
	}
	if _, err := w.Write(rs); err != nil {
		return err
	}
	_, err := w.Write(payloads)
	return err
}

// encodeAdaptiveV6 writes the version-6 adaptive stream.
// In v6 all ClassNormal ratios use 30-bit signed-offset encoding:
//   - ClassNormal (u16): int16 signed offset — small deviations from ratio=1.0 (2 bytes)
//   - ClassNormal32 (u32): int32 signed offset — larger deviations (4 bytes)
//   - ClassNormalExact (f64): 30-bit insufficient or ratio==0 (8 bytes)
//
// PrecisionBits is not used for tier routing in v6 (always 30-bit precision).
// Header: magic(4) + version=6(1) + driftMode(1) + reanchorInterval(4) +
//
//	count(8) + precisionBits=30(1) + ransFreqs7(28) = 47 bytes.
func encodeAdaptiveV6(values []float64, w io.Writer, opts EncodeOptions) error {
	classes, payloads := gatherRans7v6(values, opts)
	freqs := RansCountFreqs7(classes)
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, versionAdaptiveV6); err != nil {
		return err
	}
	if err := writeByte(w, byte(opts.DriftMode)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(opts.ReanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}
	// Always write 30 — the precision bits field is fixed at 30 for v6.
	if err := writeByte(w, 30); err != nil {
		return err
	}
	return writeRansBody7(w, classes, freqs, payloads)
}

// encodeAdaptiveV7 writes the version-8 adaptive stream.
// In v7 all ClassNormal ratios use signed-offset encoding at opts.PrecisionBits
// with u8/u16/u24/u32 tier routing based on |offset| magnitude.
// PrecisionBits is written to the header and used by the decoder for all tiers.
//
// Header: magic(4) + version=8(1) + driftMode(1) + reanchorInterval(4) +
//
//	count(8) + precisionBits(1) + ransFreqs9(36) = 55 bytes.
//
// Body: [ransLen(4)][ransStream][anchor float64][per-event payloads].
func encodeAdaptiveV7(values []float64, w io.Writer, opts EncodeOptions) error {
	bits := opts.PrecisionBits
	if bits <= 0 {
		bits = DefaultPrecisionBits
	}
	opts.PrecisionBits = bits
	classes, payloads := gatherRans7v7(values, opts)
	freqs := RansCountFreqs9(classes)
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, versionAdaptiveV7); err != nil {
		return err
	}
	if err := writeByte(w, byte(opts.DriftMode)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(opts.ReanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}
	if err := writeByte(w, byte(bits)); err != nil {
		return err
	}
	return writeRansBody9(w, classes, freqs, payloads)
}

// encodeAdaptiveV5 writes the version-5 adaptive stream.
// Superseded by v6. Kept for reference; no longer called by the encoder.
// Header: magic(4) + version=5(1) + driftMode(1) + reanchorInterval(4) +
//
//	count(8) + precisionBits(1) + ransFreqs7(28) = 47 bytes.
//
// Body: [ransLen(4)][ransStream][anchor float64][per-event payloads].
// ClassNormal payload: uint16. ClassNormal32 payload: uint32. ClassNormalExact: float64.
func encodeAdaptiveV5(values []float64, w io.Writer, opts EncodeOptions) error {
	bits := opts.PrecisionBits
	if bits <= 0 || bits > 16 {
		bits = 16
	}
	classes, payloads := gatherRans7(values, opts)
	freqs := RansCountFreqs7(classes)
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, versionAdaptiveV5); err != nil {
		return err
	}
	if err := writeByte(w, byte(opts.DriftMode)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(opts.ReanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}
	if err := writeByte(w, byte(bits)); err != nil {
		return err
	}
	return writeRansBody7(w, classes, freqs, payloads)
}

// readRansFreqs7 reads the 7-entry frequency table (28 bytes) for v5.
func readRansFreqs7(r io.Reader) (RansFreqs7, error) {
	var freqs RansFreqs7
	for i := range freqs {
		f, err := readUint32(r)
		if err != nil {
			return freqs, fmt.Errorf("toroidzip: decode: reading rans7 freqs: %w", err)
		}
		freqs[i] = f
	}
	return freqs, nil
}

// decodeV5 reads a version-5 adaptive stream (magic+version already consumed).
func decodeV5(r io.Reader) ([]float64, error) {
	dm, count, err := readCommonRansHeader(r)
	if err != nil {
		return nil, err
	}
	bitsByte, err := readByte(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading precision bits: %w", err)
	}
	freqs, err := readRansFreqs7(r)
	if err != nil {
		return nil, err
	}
	return decodeRans7(r, dm, count, freqs, int(bitsByte))
}

// decodeRans7 is the body decoder for v5 adaptive streams.
// ClassNormal:      read uint16 symbol, dequantize at bits.
// ClassNormal32:    read uint32 symbol, dequantize at 30 bits.
// ClassNormalExact: read float64 ratio verbatim.
// All other classes: same payload rules as decodeRans.
func decodeRans7(r io.Reader, driftMode DriftMode, count uint64, freqs RansFreqs7, bits int) ([]float64, error) {
	ransLen, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading rans length: %w", err)
	}
	ransStream := make([]byte, ransLen)
	if ransLen > 0 {
		if _, err := io.ReadFull(r, ransStream); err != nil {
			return nil, fmt.Errorf("toroidzip: decode: reading rans stream: %w", err)
		}
	}

	classes, err := RansDecode7(ransStream, freqs, int(count)-1)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: rans7 decode: %w", err)
	}

	payRaw, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading payloads: %w", err)
	}
	pay := bytes.NewReader(payRaw)

	values := make([]float64, count)
	values[0], err = readFloat64(pay)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading anchor: %w", err)
	}

	var kp kahanProd
	if driftMode == DriftCompensate {
		kp = newKahanProd(values[0])
	}

	for i := 1; i < int(count); i++ {
		class := RatioClass(classes[i-1])
		switch class {
		case ClassIdentity:
			values[i] = values[i-1]

		case ClassNormal:
			sym16, err := readUint16(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading i16 offset at %d: %w", i, err)
			}
			// v6: u16 payload is int16 signed offset at 30-bit precision.
			ratio := DequantizeRatioOffset(int32(int16(sym16)), 30)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassNormal32:
			sym32, err := readUint32(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading u32 symbol at %d: %w", i, err)
			}
			ratio := DequantizeRatio(sym32, 30)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassNormalExact:
			ratio, err := readFloat64(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading exact ratio at %d: %w", i, err)
			}
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassBoundaryZero, ClassBoundaryInf, ClassReanchor:
			val, err := readFloat64(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading verbatim at %d: %w", i, err)
			}
			values[i] = val
			if driftMode == DriftCompensate {
				kp = newKahanProd(val)
			}

		default:
			return nil, fmt.Errorf("toroidzip: decode: unknown class %d at %d", class, i)
		}
	}
	return values, nil
}

// decodeV6 reads a version-6 adaptive stream (magic+version already consumed).
func decodeV6(r io.Reader) ([]float64, error) {
	dm, count, err := readCommonRansHeader(r)
	if err != nil {
		return nil, err
	}
	bitsByte, err := readByte(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading precision bits: %w", err)
	}
	freqs, err := readRansFreqs7(r)
	if err != nil {
		return nil, err
	}
	return decodeRans7v6(r, dm, count, freqs, int(bitsByte))
}

// decodeRans7v6 is the body decoder for v6 adaptive streams.
// Identical to decodeRans7 except ClassNormal32 reads an int32 signed offset
// and reconstructs via DequantizeRatioOffset instead of DequantizeRatio.
func decodeRans7v6(r io.Reader, driftMode DriftMode, count uint64, freqs RansFreqs7, bits int) ([]float64, error) {
	ransLen, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading rans length: %w", err)
	}
	ransStream := make([]byte, ransLen)
	if ransLen > 0 {
		if _, err := io.ReadFull(r, ransStream); err != nil {
			return nil, fmt.Errorf("toroidzip: decode: reading rans stream: %w", err)
		}
	}

	classes, err := RansDecode7(ransStream, freqs, int(count)-1)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: rans7 decode: %w", err)
	}

	payRaw, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading payloads: %w", err)
	}
	pay := bytes.NewReader(payRaw)

	values := make([]float64, count)
	values[0], err = readFloat64(pay)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading anchor: %w", err)
	}

	var kp kahanProd
	if driftMode == DriftCompensate {
		kp = newKahanProd(values[0])
	}

	for i := 1; i < int(count); i++ {
		class := RatioClass(classes[i-1])
		switch class {
		case ClassIdentity:
			values[i] = values[i-1]

		case ClassNormal:
			sym16, err := readUint16(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading i16 offset at %d: %w", i, err)
			}
			// v6: u16 payload is int16 signed offset at 30-bit precision.
			ratio := DequantizeRatioOffset(int32(int16(sym16)), 30)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassNormal32:
			// v6: payload is int32 signed offset from log-space centre.
			raw, err := readUint32(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading i32 offset at %d: %w", i, err)
			}
			ratio := DequantizeRatioOffset(int32(raw), 30)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassNormalExact:
			ratio, err := readFloat64(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading exact ratio at %d: %w", i, err)
			}
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassBoundaryZero, ClassBoundaryInf, ClassReanchor:
			val, err := readFloat64(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading verbatim at %d: %w", i, err)
			}
			values[i] = val
			if driftMode == DriftCompensate {
				kp = newKahanProd(val)
			}

		default:
			return nil, fmt.Errorf("toroidzip: decode: unknown class %d at %d", class, i)
		}
	}
	return values, nil
}

// decodeV7 reads a version-8 adaptive stream (magic+version already consumed).
func decodeV7(r io.Reader) ([]float64, error) {
	dm, count, err := readCommonRansHeader(r)
	if err != nil {
		return nil, err
	}
	bitsByte, err := readByte(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading precision bits: %w", err)
	}
	freqs, err := readRansFreqs9(r)
	if err != nil {
		return nil, err
	}
	return decodeRans7v7(r, dm, count, freqs, int(bitsByte))
}

// decodeRans7v7 is the body decoder for v7 (version-8) adaptive streams.
// All ClassNormal* payloads are signed offsets at precision bits (from header).
// ClassNormalExact: float64 ratio verbatim (ratio==0 or outside quantizer range).
// ClassBoundaryZero/Inf/Reanchor: verbatim float64 value.
func decodeRans7v7(r io.Reader, driftMode DriftMode, count uint64, freqs RansFreqs9, bits int) ([]float64, error) {
	ransLen, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading rans length: %w", err)
	}
	ransStream := make([]byte, ransLen)
	if ransLen > 0 {
		if _, err := io.ReadFull(r, ransStream); err != nil {
			return nil, fmt.Errorf("toroidzip: decode: reading rans stream: %w", err)
		}
	}

	classes, err := RansDecode9(ransStream, freqs, int(count)-1)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: rans9 decode: %w", err)
	}

	payRaw, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading payloads: %w", err)
	}
	pay := bytes.NewReader(payRaw)

	values := make([]float64, count)
	values[0], err = readFloat64(pay)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading anchor: %w", err)
	}

	var kp kahanProd
	if driftMode == DriftCompensate {
		kp = newKahanProd(values[0])
	}

	for i := 1; i < int(count); i++ {
		class := RatioClass(classes[i-1])
		switch class {
		case ClassIdentity:
			values[i] = values[i-1]

		case ClassNormal8:
			b, err := readByte(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading Normal8 at %d: %w", i, err)
			}
			off := int32(int8(b))
			ratio := DequantizeRatioOffset(off, bits)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassNormal:
			s, err := readUint16(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading Normal at %d: %w", i, err)
			}
			off := int32(int16(s))
			ratio := DequantizeRatioOffset(off, bits)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassNormal24:
			var b [3]byte
			if _, err := io.ReadFull(pay, b[:]); err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading Normal24 at %d: %w", i, err)
			}
			raw := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16
			if raw&(1<<23) != 0 {
				raw |= ^int32(0xFFFFFF) // sign-extend from bit 23
			}
			ratio := DequantizeRatioOffset(raw, bits)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassNormal32:
			u, err := readUint32(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading Normal32 at %d: %w", i, err)
			}
			off := int32(u)
			ratio := DequantizeRatioOffset(off, bits)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassNormalExact:
			ratio, err := readFloat64(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading exact ratio at %d: %w", i, err)
			}
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassBoundaryZero, ClassBoundaryInf, ClassReanchor:
			val, err := readFloat64(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading verbatim at %d: %w", i, err)
			}
			values[i] = val
			if driftMode == DriftCompensate {
				kp = newKahanProd(val)
			}

		default:
			return nil, fmt.Errorf("toroidzip: decode: unknown class %d at %d", class, i)
		}
	}
	return values, nil
}

// readRansFreqs6 reads the 6-entry frequency table (24 bytes) for v4.
func readRansFreqs6(r io.Reader) (RansFreqs6, error) {
	var freqs RansFreqs6
	for i := range freqs {
		f, err := readUint32(r)
		if err != nil {
			return freqs, fmt.Errorf("toroidzip: decode: reading rans6 freqs: %w", err)
		}
		freqs[i] = f
	}
	return freqs, nil
}

// decodeV4 reads a version-4 adaptive stream (magic+version already consumed).
func decodeV4(r io.Reader) ([]float64, error) {
	dm, count, err := readCommonRansHeader(r)
	if err != nil {
		return nil, err
	}
	bitsByte, err := readByte(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading precision bits: %w", err)
	}
	freqs, err := readRansFreqs6(r)
	if err != nil {
		return nil, err
	}
	return decodeRans6(r, dm, count, freqs, int(bitsByte))
}

// decodeRans6 is the body decoder for v4 adaptive streams.
// ClassNormal: read uint16 symbol, dequantize.
// ClassNormalExact: read float64 ratio verbatim.
// All other classes: same payload rules as decodeRans.
func decodeRans6(r io.Reader, driftMode DriftMode, count uint64, freqs RansFreqs6, bits int) ([]float64, error) {
	ransLen, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading rans length: %w", err)
	}
	ransStream := make([]byte, ransLen)
	if ransLen > 0 {
		if _, err := io.ReadFull(r, ransStream); err != nil {
			return nil, fmt.Errorf("toroidzip: decode: reading rans stream: %w", err)
		}
	}

	classes, err := RansDecode6(ransStream, freqs, int(count)-1)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: rans6 decode: %w", err)
	}

	payRaw, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading payloads: %w", err)
	}
	pay := bytes.NewReader(payRaw)

	values := make([]float64, count)
	values[0], err = readFloat64(pay)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading anchor: %w", err)
	}

	var kp kahanProd
	if driftMode == DriftCompensate {
		kp = newKahanProd(values[0])
	}

	for i := 1; i < int(count); i++ {
		class := RatioClass(classes[i-1])
		switch class {
		case ClassIdentity:
			values[i] = values[i-1]

		case ClassNormal:
			sym16, err := readUint16(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading adaptive symbol at %d: %w", i, err)
			}
			ratio := DequantizeRatio(uint32(sym16), bits)
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassNormalExact:
			ratio, err := readFloat64(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading exact ratio at %d: %w", i, err)
			}
			if driftMode == DriftCompensate {
				values[i] = kp.multiply(ratio)
			} else {
				values[i] = values[i-1] * ratio
			}

		case ClassBoundaryZero, ClassBoundaryInf, ClassReanchor:
			val, err := readFloat64(pay)
			if err != nil {
				return nil, fmt.Errorf("toroidzip: decode: reading verbatim at %d: %w", i, err)
			}
			values[i] = val
			if driftMode == DriftCompensate {
				kp = newKahanProd(val)
			}

		default:
			return nil, fmt.Errorf("toroidzip: decode: unknown class %d at %d", class, i)
		}
	}
	return values, nil
}

// kahanProd tracks a cumulative product in log-space with Kahan compensation.
// This is Mode B's drift-reduction strategy: far more accurate than naive
// float64 chained multiplication over long sequences.
type kahanProd struct {
	logAbs float64 // log(|accumulated product|)
	comp   float64 // Kahan compensation term
	sign   float64 // +1 or -1
}

func newKahanProd(v float64) kahanProd {
	if v == 0 {
		return kahanProd{math.Inf(-1), 0, 1}
	}
	return kahanProd{math.Log(math.Abs(v)), 0, math.Copysign(1, v)}
}

// multiply applies one ratio step and returns the reconstructed value.
func (k *kahanProd) multiply(ratio float64) float64 {
	k.sign *= math.Copysign(1, ratio)
	// Kahan summation of log(|ratio|) into logAbs.
	y := math.Log(math.Abs(ratio)) - k.comp
	t := k.logAbs + y
	k.comp = (t - k.logAbs) - y
	k.logAbs = t
	return k.sign * math.Exp(k.logAbs)
}

// computeRatio returns the ratio r = current/prev and its class.
// Handles the case where prev is zero, near-zero, infinite, or NaN.
func computeRatio(current, prev float64) (float64, RatioClass) {
	if math.Abs(prev) < BoundaryZeroThreshold || math.IsInf(prev, 0) || math.IsNaN(prev) {
		return current, ClassBoundaryZero
	}
	ratio := current / prev
	return ratio, Classify(ratio)
}

// --- binary helpers ---

func writeByte(w io.Writer, b byte) error {
	_, err := w.Write([]byte{b})
	return err
}

func writeUint32(w io.Writer, v uint32) error {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], v)
	_, err := w.Write(buf[:])
	return err
}

func writeUint64(w io.Writer, v uint64) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], v)
	_, err := w.Write(buf[:])
	return err
}

func writeFloat64(w io.Writer, v float64) error {
	return writeUint64(w, math.Float64bits(v))
}

func readByte(r io.Reader) (byte, error) {
	var buf [1]byte
	_, err := io.ReadFull(r, buf[:])
	return buf[0], err
}

func readUint32(r io.Reader) (uint32, error) {
	var buf [4]byte
	_, err := io.ReadFull(r, buf[:])
	return binary.LittleEndian.Uint32(buf[:]), err
}

func readUint64(r io.Reader) (uint64, error) {
	var buf [8]byte
	_, err := io.ReadFull(r, buf[:])
	return binary.LittleEndian.Uint64(buf[:]), err
}

func readFloat64(r io.Reader) (float64, error) {
	bits, err := readUint64(r)
	return math.Float64frombits(bits), err
}

func readUint16(r io.Reader) (uint16, error) {
	var buf [2]byte
	_, err := io.ReadFull(r, buf[:])
	return binary.LittleEndian.Uint16(buf[:]), err
}
