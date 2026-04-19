// rANS (range Asymmetric Numeral Systems) entropy coder for the RatioClass
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
