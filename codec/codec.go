// Package codec implements the ToroidZip ratio-first compression algorithm.
//
// Encoding overview:
//  1. Write a verbatim anchor value (float64).
//  2. For each subsequent value, compute the ratio r[n] = x[n] / x[n-1].
//  3. Classify the ratio (identity, normal, pole-zero, pole-inf).
//  4. Write the ratio stream. Pole events are written to a sidecar.
//  5. Every K values, write a new verbatim anchor to bound drift.
//
// Decoding overview:
//  1. Read anchor.
//  2. Reconstruct x[n] = x[n-1] * r[n] for all n.
//  3. Inject pole events at their recorded positions.
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

// Magic bytes identifying a ToroidZip stream.
var magic = [4]byte{'T', 'Z', 'R', 'Z'}

// version is the current stream format version.
const version byte = 1

// Encode compresses values into the ToroidZip format and writes to w.
// reanchorInterval controls how often a verbatim anchor is written (0 = use default).
func Encode(values []float64, w io.Writer, reanchorInterval int) error {
	if len(values) == 0 {
		return fmt.Errorf("toroidzip: encode: empty input")
	}
	if reanchorInterval <= 0 {
		reanchorInterval = DefaultReanchorInterval
	}

	// Header: magic (4) + version (1) + reanchorInterval (4) + count (8) = 17 bytes
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, version); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(reanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}

	// Encode the value stream.
	prev := values[0]
	if err := writeFloat64(w, prev); err != nil {
		return err
	}

	for i := 1; i < len(values); i++ {
		// Periodic re-anchor to bound cumulative drift.
		if i%reanchorInterval == 0 {
			if err := writeByte(w, byte(ClassReanchor)); err != nil {
				return err
			}
			if err := writeFloat64(w, values[i]); err != nil {
				return err
			}
			prev = values[i]
			continue
		}

		ratio, class := computeRatio(values[i], prev)
		if err := writeByte(w, byte(class)); err != nil {
			return err
		}
		if err := writeFloat64(w, ratio); err != nil {
			return err
		}
		prev = values[i]
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

	reanchorInterval, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("toroidzip: decode: reading reanchor interval: %w", err)
	}
	_ = reanchorInterval // stored for future use; reconstruction uses the class byte

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
			// Verbatim value: re-anchor or pole event stores the raw value directly.
			values[i] = val
		} else {
			// Reconstruct: x[n] = x[n-1] * ratio.
			values[i] = values[i-1] * val
		}
	}

	return values, nil
}

// computeRatio returns the ratio r = current/prev and its class.
// Handles the case where prev is zero or the ratio is out of normal bounds.
func computeRatio(current, prev float64) (float64, RatioClass) {
	if prev == 0 || math.IsInf(prev, 0) || math.IsNaN(prev) {
		return current, ClassPoleZero
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
