// Package codec implements a rANS (range Asymmetric Numeral Systems) entropy coder for the RatioClass
// alphabet.
//
// Parameters:
//   - 5-symbol alphabet: ClassIdentity..ClassReanchor
//   - 12-bit frequency precision: M = 4096 (sum of all frequencies)
//   - 64-bit state; normalised range [L, 256·L) where L = 2^31
//   - Byte-level I/O (renormalise one byte at a time)
//
// Stream layout produced by RansEncode:
//
//	[8 bytes: initial decoder state, little-endian uint64]
//	[variable: renormalisation bytes, forward order]
//
// This is a pure codec: it knows nothing about the ToroidZip stream format.
// Integration with Encode/Decode happens in codec.go (Milestone 2, Piece 3).
package codec

import (
	"encoding/binary"
	"fmt"
	"math"
)

// rANS tuning constants.
const (
	ransScaleBits = 12
	ransM         = 1 << ransScaleBits // 4096 — total frequency table size
	ransNumSyms   = 5                  // ClassIdentity(0)..ClassReanchor(4)
)

// ransL is the lower bound of the normalised state range [ransL, ransL*256).
// Using an untyped constant so it freely converts to uint64 in expressions.
const ransL = 1 << 31

// RansFreqs is a normalised frequency table for the 5-symbol RatioClass
// alphabet. The invariants are:
//
//	sum(freqs) == ransM
//	freqs[i]   >= 1 for all i
type RansFreqs [ransNumSyms]uint32

// ransSym holds precomputed per-symbol information for encode and decode.
type ransSym struct {
	freq  uint32
	cumul uint32 // cumulative frequency before this symbol
}

// ransTables holds all precomputed lookup tables.
type ransTables struct {
	sym    [ransNumSyms]ransSym
	decode [ransM]byte // slot (x % ransM) → symbol index
}

// buildRansTables constructs encode/decode tables from a frequency array.
func buildRansTables(freqs RansFreqs) ransTables {
	var t ransTables
	var c uint32
	for i := 0; i < ransNumSyms; i++ {
		t.sym[i] = ransSym{freq: freqs[i], cumul: c}
		for s := c; s < c+freqs[i]; s++ {
			t.decode[s] = byte(i)
		}
		c += freqs[i]
	}
	return t
}

// RansCountFreqs counts RatioClass occurrences in classes and returns a
// frequency table normalised to sum exactly to ransM, with every symbol
// receiving at least 1.
func RansCountFreqs(classes []byte) RansFreqs {
	var raw [ransNumSyms]uint64
	for _, c := range classes {
		if int(c) < ransNumSyms {
			raw[c]++
		}
	}
	return normalizeRansFreqs(raw)
}

// normalizeRansFreqs scales raw counts to sum == ransM with each entry >= 1.
func normalizeRansFreqs(raw [ransNumSyms]uint64) RansFreqs {
	var total uint64
	for _, v := range raw {
		total += v
	}

	var freqs RansFreqs
	if total == 0 {
		// Degenerate: assign equal frequencies.
		each := uint32(ransM / ransNumSyms)
		for i := range freqs {
			freqs[i] = each
		}
		freqs[0] += uint32(ransM) - each*uint32(ransNumSyms)
		return freqs
	}

	// Scale proportionally, guarantee minimum 1, track the largest bucket.
	var sum uint32
	maxVal, maxIdx := uint32(0), 0
	for i, v := range raw {
		f := uint32(math.Round(float64(v) / float64(total) * ransM))
		if f < 1 {
			f = 1
		}
		freqs[i] = f
		sum += f
		if f > maxVal {
			maxVal, maxIdx = f, i
		}
	}

	// Adjust the largest bucket so the sum is exactly ransM.
	switch {
	case sum < ransM:
		freqs[maxIdx] += ransM - sum
	case sum > ransM:
		excess := sum - ransM
		if freqs[maxIdx] > excess {
			freqs[maxIdx] -= excess
		} else {
			// Distribute reduction across all buckets > 1.
			for i := range freqs {
				if freqs[i] > 1 && excess > 0 {
					freqs[i]--
					excess--
				}
			}
		}
	}

	return freqs
}

// RansEncode encodes a stream of RatioClass bytes using rANS.
//
// Returns nil for empty input. The caller is responsible for storing freqs
// alongside the output so RansDecode can reconstruct the tables.
func RansEncode(classes []byte, freqs RansFreqs) []byte {
	if len(classes) == 0 {
		return nil
	}
	t := buildRansTables(freqs)

	// Encode symbols in reverse order, collecting renormalisation bytes.
	// Reversing at the end allows the decoder to read bytes forward.
	norm := make([]byte, 0, len(classes)/2)
	x := uint64(ransL)

	for i := len(classes) - 1; i >= 0; i-- {
		s := t.sym[classes[i]]

		// Normalise: push low bytes until x is in the pre-encode range
		// [L·freq/M, L·b·freq/M) where b = 256.
		// xmax = (L/M)·b·freq = (L >> scaleBits)·256·freq.
		xmax := ((uint64(ransL) >> ransScaleBits) << 8) * uint64(s.freq)
		for x >= xmax {
			norm = append(norm, byte(x))
			x >>= 8
		}

		// Encode: map (x, symbol) → x'.
		x = (x/uint64(s.freq))<<ransScaleBits + uint64(s.cumul) + x%uint64(s.freq)
	}

	// Flush the final state (8 bytes, little-endian).
	var stateBuf [8]byte
	binary.LittleEndian.PutUint64(stateBuf[:], x)

	// Reverse the renormalisation bytes so the decoder reads them forward.
	for i, j := 0, len(norm)-1; i < j; i, j = i+1, j-1 {
		norm[i], norm[j] = norm[j], norm[i]
	}

	out := make([]byte, 8+len(norm))
	copy(out[:8], stateBuf[:])
	copy(out[8:], norm)
	return out
}

// RansDecode decodes count RatioClass bytes from data.
// data must start with the 8-byte state header produced by RansEncode,
// followed by the renormalisation byte stream.
func RansDecode(data []byte, freqs RansFreqs, count int) ([]byte, error) {
	if count == 0 {
		return nil, nil
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("rans: decode: stream too short (%d bytes, need ≥8)", len(data))
	}

	t := buildRansTables(freqs)

	x := binary.LittleEndian.Uint64(data[:8])
	pos := 8

	classes := make([]byte, count)
	for i := 0; i < count; i++ {
		// Decode: identify symbol from the low ransScaleBits bits of x.
		slot := uint32(x) & (ransM - 1)
		sym := t.decode[slot]
		s := t.sym[sym]
		classes[i] = sym

		// Update state: reverse the encode step.
		x = uint64(s.freq)*(x>>ransScaleBits) + uint64(slot) - uint64(s.cumul)

		// Renormalise: absorb bytes until x ≥ L.
		for x < uint64(ransL) {
			if pos < len(data) {
				x = (x << 8) | uint64(data[pos])
				pos++
			} else {
				x <<= 8 // stream exhausted; pad with zero
			}
		}
	}

	return classes, nil
}

// ============================================================
// 6-symbol rANS — v4 adaptive stream (EntropyAdaptive)
// Adds ClassNormalExact (5) to the existing 5-symbol alphabet.
// API mirrors the 5-symbol functions above; internals are identical
// in structure — only the alphabet size differs.
// NOTE: v4 stream is superseded by v5 (7-symbol). The 6-symbol functions
// remain to support decoding of existing v4-encoded streams.
// ============================================================

// ransNumSyms6 is the alphabet size for the v4 adaptive stream.
const ransNumSyms6 = 6

// RansFreqs6 is a normalised frequency table for the 6-symbol v4 alphabet.
// Invariants: sum(freqs) == ransM, freqs[i] >= 1 for all i.
type RansFreqs6 [ransNumSyms6]uint32

// ransTables6 holds precomputed encode/decode tables for the 6-symbol alphabet.
type ransTables6 struct {
	sym    [ransNumSyms6]ransSym
	decode [ransM]byte
}

// buildRansTables6 constructs encode/decode tables from a 6-symbol frequency array.
func buildRansTables6(freqs RansFreqs6) ransTables6 {
	var t ransTables6
	var c uint32
	for i := 0; i < ransNumSyms6; i++ {
		t.sym[i] = ransSym{freq: freqs[i], cumul: c}
		for s := c; s < c+freqs[i]; s++ {
			t.decode[s] = byte(i)
		}
		c += freqs[i]
	}
	return t
}

// normalizeRansFreqs6 scales raw 6-symbol counts to sum == ransM, each >= 1.
func normalizeRansFreqs6(raw [ransNumSyms6]uint64) RansFreqs6 {
	var total uint64
	for _, v := range raw {
		total += v
	}

	var freqs RansFreqs6
	if total == 0 {
		each := uint32(ransM / ransNumSyms6)
		for i := range freqs {
			freqs[i] = each
		}
		freqs[0] += uint32(ransM) - each*uint32(ransNumSyms6)
		return freqs
	}

	var sum uint32
	maxVal, maxIdx := uint32(0), 0
	for i, v := range raw {
		f := uint32(math.Round(float64(v) / float64(total) * ransM))
		if f < 1 {
			f = 1
		}
		freqs[i] = f
		sum += f
		if f > maxVal {
			maxVal, maxIdx = f, i
		}
	}

	switch {
	case sum < ransM:
		freqs[maxIdx] += ransM - sum
	case sum > ransM:
		excess := sum - ransM
		if freqs[maxIdx] > excess {
			freqs[maxIdx] -= excess
		} else {
			for i := range freqs {
				if freqs[i] > 1 && excess > 0 {
					freqs[i]--
					excess--
				}
			}
		}
	}

	return freqs
}

// RansCountFreqs6 counts RatioClass occurrences in classes (6-symbol alphabet)
// and returns a frequency table normalised to sum exactly to ransM.
func RansCountFreqs6(classes []byte) RansFreqs6 {
	var raw [ransNumSyms6]uint64
	for _, c := range classes {
		if int(c) < ransNumSyms6 {
			raw[c]++
		}
	}
	return normalizeRansFreqs6(raw)
}

// RansEncode6 encodes a stream of RatioClass bytes using the 6-symbol rANS alphabet.
// Returns nil for empty input.
func RansEncode6(classes []byte, freqs RansFreqs6) []byte {
	if len(classes) == 0 {
		return nil
	}
	t := buildRansTables6(freqs)

	norm := make([]byte, 0, len(classes)/2)
	x := uint64(ransL)

	for i := len(classes) - 1; i >= 0; i-- {
		s := t.sym[classes[i]]
		xmax := ((uint64(ransL) >> ransScaleBits) << 8) * uint64(s.freq)
		for x >= xmax {
			norm = append(norm, byte(x))
			x >>= 8
		}
		x = (x/uint64(s.freq))<<ransScaleBits + uint64(s.cumul) + x%uint64(s.freq)
	}

	var stateBuf [8]byte
	binary.LittleEndian.PutUint64(stateBuf[:], x)

	for i, j := 0, len(norm)-1; i < j; i, j = i+1, j-1 {
		norm[i], norm[j] = norm[j], norm[i]
	}

	out := make([]byte, 8+len(norm))
	copy(out[:8], stateBuf[:])
	copy(out[8:], norm)
	return out
}

// RansDecode6 decodes count RatioClass bytes from data using the 6-symbol alphabet.
// data must start with the 8-byte state header produced by RansEncode6.
func RansDecode6(data []byte, freqs RansFreqs6, count int) ([]byte, error) {
	if count == 0 {
		return nil, nil
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("rans: decode6: stream too short (%d bytes, need ≥8)", len(data))
	}

	t := buildRansTables6(freqs)
	x := binary.LittleEndian.Uint64(data[:8])
	pos := 8

	classes := make([]byte, count)
	for i := 0; i < count; i++ {
		slot := uint32(x) & (ransM - 1)
		sym := t.decode[slot]
		s := t.sym[sym]
		classes[i] = sym

		x = uint64(s.freq)*(x>>ransScaleBits) + uint64(slot) - uint64(s.cumul)

		for x < uint64(ransL) {
			if pos < len(data) {
				x = (x << 8) | uint64(data[pos])
				pos++
			} else {
				x <<= 8
			}
		}
	}

	return classes, nil
}

// ============================================================
// 7-symbol rANS — v5 adaptive stream (EntropyAdaptive)
// Adds ClassNormal32 (6) to the 6-symbol alphabet.
// API mirrors the 6-symbol functions above; internals are identical
// in structure — only the alphabet size differs.
// ============================================================

// ransNumSyms7 is the alphabet size for the v5 adaptive stream.
const ransNumSyms7 = 7

// RansFreqs7 is a normalised frequency table for the 7-symbol v5 alphabet.
// Invariants: sum(freqs) == ransM, freqs[i] >= 1 for all i.
type RansFreqs7 [ransNumSyms7]uint32

// ransTables7 holds precomputed encode/decode tables for the 7-symbol alphabet.
type ransTables7 struct {
	sym    [ransNumSyms7]ransSym
	decode [ransM]byte
}

// buildRansTables7 constructs encode/decode tables from a 7-symbol frequency array.
func buildRansTables7(freqs RansFreqs7) ransTables7 {
	var t ransTables7
	var c uint32
	for i := 0; i < ransNumSyms7; i++ {
		t.sym[i] = ransSym{freq: freqs[i], cumul: c}
		for s := c; s < c+freqs[i]; s++ {
			t.decode[s] = byte(i)
		}
		c += freqs[i]
	}
	return t
}

// normalizeRansFreqs7 scales raw 7-symbol counts to sum == ransM, each >= 1.
func normalizeRansFreqs7(raw [ransNumSyms7]uint64) RansFreqs7 {
	var total uint64
	for _, v := range raw {
		total += v
	}

	var freqs RansFreqs7
	if total == 0 {
		each := uint32(ransM / ransNumSyms7)
		for i := range freqs {
			freqs[i] = each
		}
		freqs[0] += uint32(ransM) - each*uint32(ransNumSyms7)
		return freqs
	}

	var sum uint32
	maxVal, maxIdx := uint32(0), 0
	for i, v := range raw {
		f := uint32(math.Round(float64(v) / float64(total) * ransM))
		if f < 1 {
			f = 1
		}
		freqs[i] = f
		sum += f
		if f > maxVal {
			maxVal, maxIdx = f, i
		}
	}

	switch {
	case sum < ransM:
		freqs[maxIdx] += ransM - sum
	case sum > ransM:
		excess := sum - ransM
		if freqs[maxIdx] > excess {
			freqs[maxIdx] -= excess
		} else {
			for i := range freqs {
				if freqs[i] > 1 && excess > 0 {
					freqs[i]--
					excess--
				}
			}
		}
	}

	return freqs
}

// RansCountFreqs7 counts RatioClass occurrences in classes (7-symbol alphabet)
// and returns a frequency table normalised to sum exactly to ransM.
func RansCountFreqs7(classes []byte) RansFreqs7 {
	var raw [ransNumSyms7]uint64
	for _, c := range classes {
		if int(c) < ransNumSyms7 {
			raw[c]++
		}
	}
	return normalizeRansFreqs7(raw)
}

// RansEncode7 encodes a stream of RatioClass bytes using the 7-symbol rANS alphabet.
// Returns nil for empty input.
func RansEncode7(classes []byte, freqs RansFreqs7) []byte {
	if len(classes) == 0 {
		return nil
	}
	t := buildRansTables7(freqs)

	norm := make([]byte, 0, len(classes)/2)
	x := uint64(ransL)

	for i := len(classes) - 1; i >= 0; i-- {
		s := t.sym[classes[i]]
		xmax := ((uint64(ransL) >> ransScaleBits) << 8) * uint64(s.freq)
		for x >= xmax {
			norm = append(norm, byte(x))
			x >>= 8
		}
		x = (x/uint64(s.freq))<<ransScaleBits + uint64(s.cumul) + x%uint64(s.freq)
	}

	var stateBuf [8]byte
	binary.LittleEndian.PutUint64(stateBuf[:], x)

	for i, j := 0, len(norm)-1; i < j; i, j = i+1, j-1 {
		norm[i], norm[j] = norm[j], norm[i]
	}

	out := make([]byte, 8+len(norm))
	copy(out[:8], stateBuf[:])
	copy(out[8:], norm)
	return out
}

// RansDecode7 decodes count RatioClass bytes from data using the 7-symbol alphabet.
// data must start with the 8-byte state header produced by RansEncode7.
func RansDecode7(data []byte, freqs RansFreqs7, count int) ([]byte, error) {
	if count == 0 {
		return nil, nil
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("rans: decode7: stream too short (%d bytes, need ≥8)", len(data))
	}

	t := buildRansTables7(freqs)
	x := binary.LittleEndian.Uint64(data[:8])
	pos := 8

	classes := make([]byte, count)
	for i := 0; i < count; i++ {
		slot := uint32(x) & (ransM - 1)
		sym := t.decode[slot]
		s := t.sym[sym]
		classes[i] = sym

		x = uint64(s.freq)*(x>>ransScaleBits) + uint64(slot) - uint64(s.cumul)

		for x < uint64(ransL) {
			if pos < len(data) {
				x = (x << 8) | uint64(data[pos])
				pos++
			} else {
				x <<= 8
			}
		}
	}

	return classes, nil
}

// ============================================================
// 9-symbol rANS alphabet (v4-quantized and v7-adaptive streams)
// Adds ClassNormal8 (7) and ClassNormal24 (8) to the 7-symbol alphabet.
// API mirrors the 7-symbol functions above; internals are identical
// in structure — only the alphabet size differs.
// ============================================================

// ransNumSyms9 is the alphabet size for the v4-quantized and v7-adaptive streams.
const ransNumSyms9 = 9

// RansFreqs9 is a normalised frequency table for the 9-symbol alphabet.
// Invariants: sum(freqs) == ransM, freqs[i] >= 1 for all i.
type RansFreqs9 [ransNumSyms9]uint32

// ransTables9 holds precomputed encode/decode tables for the 9-symbol alphabet.
type ransTables9 struct {
	sym    [ransNumSyms9]ransSym
	decode [ransM]byte
}

// buildRansTables9 constructs encode/decode tables from a 9-symbol frequency array.
func buildRansTables9(freqs RansFreqs9) ransTables9 {
	var t ransTables9
	var c uint32
	for i := 0; i < ransNumSyms9; i++ {
		t.sym[i] = ransSym{freq: freqs[i], cumul: c}
		for s := c; s < c+freqs[i]; s++ {
			t.decode[s] = byte(i)
		}
		c += freqs[i]
	}
	return t
}

// normalizeRansFreqs9 scales raw 9-symbol counts to sum == ransM, each >= 1.
func normalizeRansFreqs9(raw [ransNumSyms9]uint64) RansFreqs9 {
	var total uint64
	for _, v := range raw {
		total += v
	}

	var freqs RansFreqs9
	if total == 0 {
		each := uint32(ransM / ransNumSyms9)
		for i := range freqs {
			freqs[i] = each
		}
		freqs[0] += uint32(ransM) - each*uint32(ransNumSyms9)
		return freqs
	}

	var sum uint32
	maxVal, maxIdx := uint32(0), 0
	for i, v := range raw {
		f := uint32(math.Round(float64(v) / float64(total) * ransM))
		if f < 1 {
			f = 1
		}
		freqs[i] = f
		sum += f
		if f > maxVal {
			maxVal, maxIdx = f, i
		}
	}

	switch {
	case sum < ransM:
		freqs[maxIdx] += ransM - sum
	case sum > ransM:
		excess := sum - ransM
		if freqs[maxIdx] > excess {
			freqs[maxIdx] -= excess
		} else {
			for i := range freqs {
				if freqs[i] > 1 && excess > 0 {
					freqs[i]--
					excess--
				}
			}
		}
	}

	return freqs
}

// RansCountFreqs9 counts RatioClass occurrences in classes (9-symbol alphabet)
// and returns a frequency table normalised to sum exactly to ransM.
func RansCountFreqs9(classes []byte) RansFreqs9 {
	var raw [ransNumSyms9]uint64
	for _, c := range classes {
		if int(c) < ransNumSyms9 {
			raw[c]++
		}
	}
	return normalizeRansFreqs9(raw)
}

// RansEncode9 encodes a stream of RatioClass bytes using the 9-symbol alphabet.
// Returns nil for empty input.
func RansEncode9(classes []byte, freqs RansFreqs9) []byte {
	if len(classes) == 0 {
		return nil
	}
	t := buildRansTables9(freqs)

	norm := make([]byte, 0, len(classes)/2)
	x := uint64(ransL)

	for i := len(classes) - 1; i >= 0; i-- {
		s := t.sym[classes[i]]
		xmax := ((uint64(ransL) >> ransScaleBits) << 8) * uint64(s.freq)
		for x >= xmax {
			norm = append(norm, byte(x))
			x >>= 8
		}
		x = (x/uint64(s.freq))<<ransScaleBits + uint64(s.cumul) + x%uint64(s.freq)
	}

	var stateBuf [8]byte
	binary.LittleEndian.PutUint64(stateBuf[:], x)

	for i, j := 0, len(norm)-1; i < j; i, j = i+1, j-1 {
		norm[i], norm[j] = norm[j], norm[i]
	}

	out := make([]byte, 8+len(norm))
	copy(out[:8], stateBuf[:])
	copy(out[8:], norm)
	return out
}

// RansDecode9 decodes count RatioClass bytes from data using the 9-symbol alphabet.
// data must start with the 8-byte state header produced by RansEncode9.
func RansDecode9(data []byte, freqs RansFreqs9, count int) ([]byte, error) {
	if count == 0 {
		return nil, nil
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("rans: decode9: stream too short (%d bytes, need ≥8)", len(data))
	}

	t := buildRansTables9(freqs)
	x := binary.LittleEndian.Uint64(data[:8])
	pos := 8

	classes := make([]byte, count)
	for i := 0; i < count; i++ {
		slot := uint32(x) & (ransM - 1)
		sym := t.decode[slot]
		s := t.sym[sym]
		classes[i] = sym

		x = uint64(s.freq)*(x>>ransScaleBits) + uint64(slot) - uint64(s.cumul)

		for x < uint64(ransL) {
			if pos < len(data) {
				x = (x << 8) | uint64(data[pos])
				pos++
			} else {
				x <<= 8
			}
		}
	}

	return classes, nil
}
