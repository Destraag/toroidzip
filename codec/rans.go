// Package codec implements rANS (range Asymmetric Numeral Systems) entropy coding
// for the RatioClass alphabet used by ToroidZip streams.
//
// Only the 9-symbol alphabet (ClassIdentity..ClassNormal24) is supported;
// legacy 5/6/7-symbol families have been removed.
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
)

// ransL is the lower bound of the normalised state range [ransL, ransL*256).
// Using an untyped constant so it freely converts to uint64 in expressions.
const ransL = 1 << 31

// ransSym holds precomputed per-symbol information for encode and decode.
type ransSym struct {
	freq  uint32
	cumul uint32 // cumulative frequency before this symbol
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

// ============================================================
// 10-symbol rANS alphabet (v9 adaptive stream with dynamic offset)
// Adds ClassReanchorDynamic (9) to the 9-symbol alphabet.
// ============================================================

// ransNumSyms10 is the alphabet size for the v9 adaptive stream.
const ransNumSyms10 = 10

// RansFreqs10 is a normalised frequency table for the 10-symbol alphabet.
// Invariants: sum(freqs) == ransM, freqs[i] >= 1 for all i.
type RansFreqs10 [ransNumSyms10]uint32

// ransTables10 holds precomputed encode/decode tables for the 10-symbol alphabet.
type ransTables10 struct {
	sym    [ransNumSyms10]ransSym
	decode [ransM]byte
}

// buildRansTables10 constructs encode/decode tables from a 10-symbol frequency array.
func buildRansTables10(freqs RansFreqs10) ransTables10 {
	var t ransTables10
	var c uint32
	for i := 0; i < ransNumSyms10; i++ {
		t.sym[i] = ransSym{freq: freqs[i], cumul: c}
		for s := c; s < c+freqs[i]; s++ {
			t.decode[s] = byte(i)
		}
		c += freqs[i]
	}
	return t
}

// normalizeRansFreqs10 scales raw 10-symbol counts to sum == ransM, each >= 1.
func normalizeRansFreqs10(raw [ransNumSyms10]uint64) RansFreqs10 {
	var total uint64
	for _, v := range raw {
		total += v
	}

	var freqs RansFreqs10
	if total == 0 {
		each := uint32(ransM / ransNumSyms10)
		for i := range freqs {
			freqs[i] = each
		}
		freqs[0] += uint32(ransM) - each*uint32(ransNumSyms10)
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

// RansCountFreqs10 counts RatioClass occurrences in classes (10-symbol alphabet)
// and returns a frequency table normalised to sum exactly to ransM.
func RansCountFreqs10(classes []byte) RansFreqs10 {
	var raw [ransNumSyms10]uint64
	for _, c := range classes {
		if int(c) < ransNumSyms10 {
			raw[c]++
		}
	}
	return normalizeRansFreqs10(raw)
}

// RansEncode10 encodes a stream of RatioClass bytes using the 10-symbol alphabet.
// Returns nil for empty input.
func RansEncode10(classes []byte, freqs RansFreqs10) []byte {
	if len(classes) == 0 {
		return nil
	}
	t := buildRansTables10(freqs)

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

// RansDecode10 decodes count RatioClass bytes from data using the 10-symbol alphabet.
// data must start with the 8-byte state header produced by RansEncode10.
func RansDecode10(data []byte, freqs RansFreqs10, count int) ([]byte, error) {
	if count == 0 {
		return nil, nil
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("rans: decode10: stream too short (%d bytes, need ≥8)", len(data))
	}

	t := buildRansTables10(freqs)
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
