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
	// and EntropyAdaptive. 0 means DefaultPrecisionBits. In EntropyAdaptive
	// the value is capped at 16 (ClassNormal payloads are always uint16).
	PrecisionBits int

	// Tolerance is the maximum relative quantisation error for EntropyAdaptive.
	// A ClassNormal ratio is stored as a uint16 symbol when
	//   |DequantizeRatio(sym, bits) / ratio - 1| < Tolerance
	// and as a ClassNormalExact float64 payload otherwise.
	// Tolerance = 0 (default): all normals take the exact path — lossless-equivalent.
	// Tolerance = math.MaxFloat64: all normals are quantised — matches v3 at bits≤16.
	Tolerance float64
}

// Magic bytes identifying a ToroidZip stream.
var magic = [4]byte{'T', 'Z', 'R', 'Z'}

// Stream format version constants.
const (
	versionRaw        byte = 1 // EntropyRaw baseline
	versionLossless   byte = 2 // EntropyLossless
	versionQuantized  byte = 3 // EntropyQuantized
	versionAdaptive   byte = 4 // EntropyAdaptive v4 (superseded; still decoded)
	versionAdaptiveV5 byte = 5 // EntropyAdaptive v5 (tiered: u16 / u32 / float64)
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
		return encodeQuantized(values, w, opts)
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
func gatherRans(values []float64, opts EncodeOptions) (classes []byte, payloads []byte) {
	ri, dm := opts.ReanchorInterval, opts.DriftMode
	bits := opts.PrecisionBits
	if bits <= 0 {
		bits = DefaultPrecisionBits
	}

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

// gatherRans6 is the encoder first pass for the legacy v4 EntropyAdaptive stream.
// Kept for reference; encodeAdaptive now emits v5 via gatherRans7.
// For each ClassNormal ratio it performs the per-ratio ε decision:
//   - relative error of quantized symbol < opts.Tolerance → ClassNormal + uint16
//   - otherwise                                           → ClassNormalExact + float64
//
// ClassNormal symbols are always uint16 (bits capped at 16).
// All other classes follow the same payload rules as gatherRans.
func gatherRans6(values []float64, opts EncodeOptions) (classes []byte, payloads []byte) {
	ri, dm := opts.ReanchorInterval, opts.DriftMode
	bits := opts.PrecisionBits
	if bits <= 0 || bits > 16 {
		bits = 16
	}
	tol := opts.Tolerance

	// fastPath: if the analytical worst-case quantization error for any ratio at
	// this precision is already below tol, every ClassNormal ratio is guaranteed
	// to pass the per-ratio check. We skip the check entirely in that case.
	// delta = 2^(QuantMaxLog2R/2^bits) - 1  (half bucket-width in linear space).
	levels := uint32(1) << bits
	delta := math.Pow(2, QuantMaxLog2R/float64(levels)) - 1
	fastPath := tol > 0 && delta < tol

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
			classes = append(classes, byte(ClassIdentity))
			// no payload; prev stays unchanged

		case ClassNormal:
			sym := QuantizeRatio(ratio, bits)
			dequant := DequantizeRatio(sym, bits)
			// ratio==0 can never be quantized (decoder would reconstruct prev*dequant≠0).
			// Otherwise, in fastPath delta<tol guarantees relErr<tol without checking.
			exact := ratio == 0
			if !exact && !fastPath {
				relErr := math.Abs(dequant/ratio - 1.0)
				exact = relErr >= tol
			}
			if exact {
				// Exact fallback: full float64, no information loss.
				classes = append(classes, byte(ClassNormalExact))
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(ratio))
				// ratio unchanged — encoder and decoder both use the original float64
			} else {
				// Quantized path: compact uint16 symbol.
				classes = append(classes, byte(ClassNormal))
				payloads = binary.LittleEndian.AppendUint16(payloads, uint16(sym))
				ratio = dequant // encoder tracks dequantized ratio for drift
			}
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

// writeRansBody6 writes ransFreqs6(24) + ransLen(4) + ransStream + payloads.
func writeRansBody6(w io.Writer, classes []byte, freqs RansFreqs6, payloads []byte) error {
	for _, f := range freqs {
		if err := writeUint32(w, f); err != nil {
			return err
		}
	}
	rs := RansEncode6(classes, freqs)
	if err := writeUint32(w, uint32(len(rs))); err != nil {
		return err
	}
	if _, err := w.Write(rs); err != nil {
		return err
	}
	_, err := w.Write(payloads)
	return err
}

// encodeAdaptive writes the version-5 adaptive stream (tiered fallback).
// v4 (6-symbol) is kept only for decoding legacy streams.
func encodeAdaptive(values []float64, w io.Writer, opts EncodeOptions) error {
	return encodeAdaptiveV5(values, w, opts)
}

// gatherRans7 is the encoder first pass for the v5 EntropyAdaptive stream.
// For each ClassNormal ratio it picks the smallest tier that satisfies ε:
//   - fastPath (delta16 < tol): all ClassNormal + uint16 (no per-ratio check)
//   - relErr < tol at 16 bits:  ClassNormal   + uint16  (2 bytes)
//   - relErr < tol at 30 bits:  ClassNormal32 + uint32  (4 bytes)
//   - relErr >= tol at 30 bits: ClassNormalExact + float64 (8 bytes)
//
// ratio==0 always takes the exact path (cannot be represented by any symbol).
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
			classes = append(classes, byte(ClassIdentity))

		case ClassNormal:
			sym16 := QuantizeRatio(ratio, bits)
			dequant16 := DequantizeRatio(sym16, bits)

			var encodedRatio float64
			if ratio == 0 {
				// Zero cannot be quantized: exact fallback.
				classes = append(classes, byte(ClassNormalExact))
				payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(ratio))
				encodedRatio = ratio
			} else if fastPath || math.Abs(dequant16/ratio-1.0) < tol {
				// 16-bit path (fast or checked).
				classes = append(classes, byte(ClassNormal))
				payloads = binary.LittleEndian.AppendUint16(payloads, uint16(sym16))
				encodedRatio = dequant16
			} else {
				// 16-bit failed — try 30-bit.
				sym30 := QuantizeRatio(ratio, bits30)
				dequant30 := DequantizeRatio(sym30, bits30)
				if delta30 < tol || math.Abs(dequant30/ratio-1.0) < tol {
					// 30-bit path.
					classes = append(classes, byte(ClassNormal32))
					payloads = binary.LittleEndian.AppendUint32(payloads, sym30)
					encodedRatio = dequant30
				} else {
					// Exact fallback.
					classes = append(classes, byte(ClassNormalExact))
					payloads = binary.LittleEndian.AppendUint64(payloads, math.Float64bits(ratio))
					encodedRatio = ratio
				}
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

// encodeAdaptiveV5 writes the version-5 adaptive stream.
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
				return nil, fmt.Errorf("toroidzip: decode: reading u16 symbol at %d: %w", i, err)
			}
			ratio := DequantizeRatio(uint32(sym16), bits)
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
