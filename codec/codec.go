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
}

// Magic bytes identifying a ToroidZip stream.
var magic = [4]byte{'T', 'Z', 'R', 'Z'}

// version is the current stream format version.
const version byte = 1

// Encode compresses values into the ToroidZip format and writes to w.
func Encode(values []float64, w io.Writer, opts EncodeOptions) error {
	if len(values) == 0 {
		return fmt.Errorf("toroidzip: encode: empty input")
	}
	reanchorInterval := opts.ReanchorInterval
	if reanchorInterval <= 0 {
		reanchorInterval = DefaultReanchorInterval
	}
	driftMode := opts.DriftMode

	// Header: magic (4) + version (1) + driftMode (1) + reanchorInterval (4) + count (8) = 18 bytes
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, version); err != nil {
		return err
	}
	if err := writeByte(w, byte(driftMode)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(reanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}

	// Encode the value stream.
	// prev tracks the effective previous value used to compute ratios.
	// Mode A: prev = original input (decoder drifts, bounded by reanchors).
	// Mode B: prev = Kahan-reconstructed value (encoder stays in sync with decoder).
	// Mode C: prev = quantized reconstruction (encoder stays in sync with decoder).
	prev := values[0]
	var kp kahanProd
	if driftMode == DriftCompensate {
		kp = newKahanProd(prev)
	}
	if err := writeFloat64(w, prev); err != nil {
		return err
	}

	for i := 1; i < len(values); i++ {
		// Periodic re-anchor resets all modes to exact value.
		if i%reanchorInterval == 0 {
			if err := writeByte(w, byte(ClassReanchor)); err != nil {
				return err
			}
			if err := writeFloat64(w, values[i]); err != nil {
				return err
			}
			prev = values[i]
			if driftMode == DriftCompensate {
				kp = newKahanProd(prev)
			}
			continue
		}

		ratio, class := computeRatio(values[i], prev)

		// Mode C: quantize ratio to float32 precision for compressibility.
		if driftMode == DriftQuantize && (class == ClassNormal || class == ClassIdentity) {
			ratio = float64(float32(ratio))
			class = Classify(ratio)
		}

		if err := writeByte(w, byte(class)); err != nil {
			return err
		}
		if err := writeFloat64(w, ratio); err != nil {
			return err
		}

		// Advance prev according to mode.
		switch {
		case class == ClassBoundaryZero || class == ClassBoundaryInf:
			// Verbatim event: reset prev to the actual value.
			prev = values[i]
			if driftMode == DriftCompensate {
				kp = newKahanProd(prev)
			}
		case driftMode == DriftCompensate:
			prev = kp.multiply(ratio)
		case driftMode == DriftQuantize:
			prev = prev * ratio // ratio is already quantized
		default: // DriftReanchor
			prev = values[i]
		}
	}

	return nil
}

// Decode decompresses a ToroidZip stream from r and returns the original values.
func Decode(r io.Reader) ([]float64, error) {
	// Read header.
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
	if ver != version {
		return nil, fmt.Errorf("toroidzip: decode: unsupported version %d", ver)
	}

	driftModeByte, err := readByte(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading drift mode: %w", err)
	}
	driftMode := DriftMode(driftModeByte)

	reanchorInterval, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading reanchor interval: %w", err)
	}
	_ = reanchorInterval // stored for reference; reconstruction uses the class byte

	count, err := readUint64(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading count: %w", err)
	}
	if count == 0 {
		return nil, fmt.Errorf("toroidzip: decode: zero count")
	}

	values := make([]float64, count)

	// Read first anchor.
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
			// Verbatim value: re-anchor or boundary event stores the raw value directly.
			values[i] = val
			if driftMode == DriftCompensate {
				kp = newKahanProd(val)
			}
		} else if driftMode == DriftCompensate {
			// Mode B: Kahan log-space product — reduces cumulative drift.
			values[i] = kp.multiply(val)
		} else {
			// Mode A / Mode C: x[n] = x[n-1] * ratio.
			values[i] = values[i-1] * val
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
