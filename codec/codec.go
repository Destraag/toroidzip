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
	// fast path fires and all ClassNormal16 ratios are quantized at PrecisionBits
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
	versionQuantizedV2 byte = 7 // EntropyQuantized (offset-based tiers: u8/u16/u24/u32)
	versionAdaptiveV7  byte = 8 // EntropyAdaptive v7 (precision-relative tiers: u8/u16/u24/u32)
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
	case EntropyQuantized:
		return encodeQuantizedV2(values, w, opts)
	case EntropyAdaptive:
		return encodeAdaptive(values, w, opts)
	default:
		return fmt.Errorf("toroidzip: encode: unknown entropy mode %d", opts.EntropyMode)
	}
}

// ============================================================
// v4 quantized stream (versionQuantizedV2 = 7)
// Offset-based tier routing: u8 / u16 / u24 / u32 by |offset| magnitude.
// ============================================================

// gatherRansV4 performs the encoder first pass for the quantized v4 stream.
// For each ClassNormal16 ratio it computes a signed offset from log-space centre:
//
//	off = QuantizeRatioOffset(ratio, bits)
//	|off| ≤ 127      → ClassNormal8  + int8  (1 byte)
//	|off| ≤ 32767    → ClassNormal16 + int16 (2 bytes)
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
		if dm == DriftQuantize && (class == ClassNormal16 || class == ClassIdentity) {
			ratio = float64(float32(ratio))
			class = Classify(ratio)
		}

		if adaptiveReanchor && class == ClassNormal16 && values[i] != 0 {
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
		case ClassNormal16:
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
				classes = append(classes, byte(ClassNormal16))
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
		case ClassNormal16:
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
	case versionQuantizedV2:
		return decodeQuantizedV2(r)
	case versionAdaptiveV7:
		return decodeV7(r)
	default:
		return nil, fmt.Errorf("toroidzip: decode: unsupported version %d (streams v1–v6 are not supported; re-encode with current version)", ver)
	}
}

// readCommonRansHeader reads driftMode(1) + reanchorInterval(4) + count(8)
// from a v2/v3/v7/v8 stream (magic and version already consumed).
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

// ============================================================
// v4 adaptive stream (EntropyAdaptive)
// ============================================================

// encodeAdaptive writes the current default adaptive stream (v7).
func encodeAdaptive(values []float64, w io.Writer, opts EncodeOptions) error {
	return encodeAdaptiveV7(values, w, opts)
}

// gatherRans7v7 is the encoder first pass for the v7 EntropyAdaptive stream.
// In v7 all quantized ratios use signed-offset encoding at opts.PrecisionBits:
//
//   - fastPath (deltaB < tol): all ratios quantized at configured bits; no per-ratio check
//   - int8  offset fits (abs(off) ≤ 127):      ClassNormal8  + int8  (1 byte)
//   - int16 offset fits (abs(off) ≤ 32767):    ClassNormal16 + int16 (2 bytes)
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
		if dm == DriftQuantize && (class == ClassNormal16 || class == ClassIdentity) {
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

		case ClassNormal16:
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
						chosenClass = ClassNormal16
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
			case ClassNormal16:
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

// encodeAdaptiveV7 writes the version-8 adaptive stream.
// In v7 all ClassNormal16 ratios use signed-offset encoding at opts.PrecisionBits
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
// All ClassNormal16* payloads are signed offsets at precision bits (from header).
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

		case ClassNormal16:
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
