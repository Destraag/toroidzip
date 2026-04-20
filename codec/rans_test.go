package codec_test

import (
	"testing"

	"github.com/Destraag/toroidzip/codec"
)

// --- RansFreqs normalization ---

func TestRansFreqsSum(t *testing.T) {
	cases := []struct {
		name    string
		classes []byte
	}{
		{"smooth data (mostly identity)", makeIdentityStream(10000, 0.05)},
		{"all identity", makeIdentityStream(1000, 0)},
		{"all same non-identity", makeConstStream(1000, byte(codec.ClassNormal))},
		{"single symbol", []byte{byte(codec.ClassBoundaryInf)}},
		{"all five classes", []byte{0, 1, 2, 3, 4}},
		{"empty (degenerate)", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			freqs := codec.RansCountFreqs(tc.classes)
			var sum uint32
			for i, f := range freqs {
				if f < 1 {
					t.Errorf("freqs[%d] = %d, want >= 1", i, f)
				}
				sum += f
			}
			if sum != 4096 {
				t.Errorf("sum = %d, want 4096", sum)
			}
		})
	}
}

// --- RansEncode / RansDecode round-trips ---

func TestRansRoundTripSimple(t *testing.T) {
	classes := []byte{0, 1, 0, 0, 2, 0, 1, 4, 0, 3}
	ransRoundTrip(t, classes)
}

func TestRansRoundTripAllIdentity(t *testing.T) {
	classes := makeIdentityStream(10000, 0)
	ransRoundTrip(t, classes)
}

func TestRansRoundTripMixed(t *testing.T) {
	// Realistic smooth-data distribution: ~90% identity, ~9% normal, ~1% events.
	classes := makeIdentityStream(10000, 0.1)
	ransRoundTrip(t, classes)
}

func TestRansRoundTripSingleSymbol(t *testing.T) {
	ransRoundTrip(t, []byte{byte(codec.ClassBoundaryZero)})
}

func TestRansRoundTripAllFiveClasses(t *testing.T) {
	classes := []byte{0, 1, 2, 3, 4, 4, 3, 2, 1, 0}
	ransRoundTrip(t, classes)
}

func TestRansEmptyInput(t *testing.T) {
	freqs := codec.RansCountFreqs(nil)
	encoded := codec.RansEncode(nil, freqs)
	if encoded != nil {
		t.Errorf("expected nil for empty input, got %d bytes", len(encoded))
	}
	decoded, err := codec.RansDecode(encoded, freqs, 0)
	if err != nil || len(decoded) != 0 {
		t.Errorf("expected nil/empty decode for count=0, got err=%v decoded=%v", err, decoded)
	}
}

func TestRansDecodeTruncated(t *testing.T) {
	_, err := codec.RansDecode([]byte{1, 2, 3}, codec.RansFreqs{}, 1)
	if err == nil {
		t.Error("expected error on truncated stream, got nil")
	}
}

// --- 6-symbol rANS (v4 adaptive alphabet) ---

func TestRansFreqs6Sum(t *testing.T) {
	cases := []struct {
		name    string
		classes []byte
	}{
		{"all six classes", []byte{0, 1, 2, 3, 4, 5}},
		{"mostly identity + exact", makeStream6(10000, 0.05, 0.02)},
		{"only ClassNormalExact", makeConstStream(100, byte(codec.ClassNormalExact))},
		{"empty (degenerate)", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			freqs := codec.RansCountFreqs6(tc.classes)
			var sum uint32
			for i, f := range freqs {
				if f < 1 {
					t.Errorf("freqs6[%d] = %d, want >= 1", i, f)
				}
				sum += f
			}
			if sum != 4096 {
				t.Errorf("sum = %d, want 4096", sum)
			}
		})
	}
}

func TestRansRoundTrip6AllSixClasses(t *testing.T) {
	// Each class appears at least twice to exercise all decode slots.
	classes := []byte{0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0}
	ransRoundTrip6(t, classes)
}

func TestRansRoundTrip6Realistic(t *testing.T) {
	// Realistic v4 distribution: ~88% identity, ~7% normal (quantized),
	// ~3% exact (ClassNormalExact), ~2% boundary/reanchor events.
	classes := makeStream6(10000, 0.10, 0.03)
	ransRoundTrip6(t, classes)
}

func TestRansRoundTrip6OnlyExact(t *testing.T) {
	// All symbols are ClassNormalExact (worst case for v4 — no quantization benefit).
	classes := makeConstStream(1000, byte(codec.ClassNormalExact))
	ransRoundTrip6(t, classes)
}

func TestRansRoundTrip6Single(t *testing.T) {
	ransRoundTrip6(t, []byte{byte(codec.ClassNormalExact)})
}

func TestRansDecode6Truncated(t *testing.T) {
	_, err := codec.RansDecode6([]byte{1, 2, 3}, codec.RansFreqs6{}, 1)
	if err == nil {
		t.Error("expected error on truncated stream, got nil")
	}
}

func TestRansEncode6EmptyInput(t *testing.T) {
	freqs := codec.RansCountFreqs6(nil)
	encoded := codec.RansEncode6(nil, freqs)
	if encoded != nil {
		t.Errorf("expected nil for empty input, got %d bytes", len(encoded))
	}
	decoded, err := codec.RansDecode6(encoded, freqs, 0)
	if err != nil || len(decoded) != 0 {
		t.Errorf("unexpected result: err=%v decoded=%v", err, decoded)
	}
}

// TestRans6CompatWith5Symbol verifies that a stream containing only symbols
// 0-4 (no ClassNormalExact) round-trips identically through both the 5-symbol
// and 6-symbol codecs. This validates that the 6-symbol codec is a strict
// superset and the extra slot doesn't perturb existing symbol mapping.
func TestRans6CompatWith5Symbol(t *testing.T) {
	classes := makeIdentityStream(5000, 0.08)
	// 5-symbol path
	freqs5 := codec.RansCountFreqs(classes)
	enc5 := codec.RansEncode(classes, freqs5)
	dec5, err := codec.RansDecode(enc5, freqs5, len(classes))
	if err != nil {
		t.Fatalf("5-sym decode: %v", err)
	}
	// 6-symbol path (same classes, no symbol 5)
	freqs6 := codec.RansCountFreqs6(classes)
	enc6 := codec.RansEncode6(classes, freqs6)
	dec6, err := codec.RansDecode6(enc6, freqs6, len(classes))
	if err != nil {
		t.Fatalf("6-sym decode: %v", err)
	}
	// Both must reproduce the original.
	for i := range classes {
		if dec5[i] != classes[i] {
			t.Fatalf("5-sym mismatch at %d: got %d want %d", i, dec5[i], classes[i])
		}
		if dec6[i] != classes[i] {
			t.Fatalf("6-sym mismatch at %d: got %d want %d", i, dec6[i], classes[i])
		}
	}
}

// --- Benchmarks ---

func BenchmarkRansEncode(b *testing.B) {
	classes := makeIdentityStream(100_000, 0.05)
	freqs := codec.RansCountFreqs(classes)
	b.ResetTimer()
	b.SetBytes(int64(len(classes)))
	for i := 0; i < b.N; i++ {
		_ = codec.RansEncode(classes, freqs)
	}
}

func BenchmarkRansDecode(b *testing.B) {
	classes := makeIdentityStream(100_000, 0.05)
	freqs := codec.RansCountFreqs(classes)
	encoded := codec.RansEncode(classes, freqs)
	b.ResetTimer()
	b.SetBytes(int64(len(encoded)))
	for i := 0; i < b.N; i++ {
		_, _ = codec.RansDecode(encoded, freqs, len(classes))
	}
}

// --- helpers ---

// ransRoundTrip encodes and decodes classes, asserting byte-exact equality.
func ransRoundTrip(t *testing.T, classes []byte) {
	t.Helper()
	freqs := codec.RansCountFreqs(classes)
	encoded := codec.RansEncode(classes, freqs)
	decoded, err := codec.RansDecode(encoded, freqs, len(classes))
	if err != nil {
		t.Fatalf("RansDecode: %v", err)
	}
	if len(decoded) != len(classes) {
		t.Fatalf("length mismatch: got %d want %d", len(decoded), len(classes))
	}
	for i := range classes {
		if decoded[i] != classes[i] {
			t.Errorf("index %d: got %d want %d", i, decoded[i], classes[i])
		}
	}
}

// makeIdentityStream generates a class byte stream where pNonIdentity fraction
// of symbols are ClassNormal (index 1), rest are ClassIdentity (index 0).
func makeIdentityStream(n int, pNonIdentity float64) []byte {
	classes := make([]byte, n)
	step := 1
	if pNonIdentity > 0 {
		step = int(1 / pNonIdentity)
	}
	for i := range classes {
		if step > 0 && i%step == 0 {
			classes[i] = byte(codec.ClassNormal)
		}
	}
	return classes
}

// makeConstStream generates a stream of n copies of sym.
func makeConstStream(n int, sym byte) []byte {
	classes := make([]byte, n)
	for i := range classes {
		classes[i] = sym
	}
	return classes
}

// makeStream6 generates a 6-symbol class stream.
// pNonIdentity fraction are ClassNormal (1); pExact fraction are ClassNormalExact (5);
// the rest are ClassIdentity (0).
func makeStream6(n int, pNonIdentity, pExact float64) []byte {
	classes := make([]byte, n)
	normalStep, exactStep := 0, 0
	if pNonIdentity > 0 {
		normalStep = int(1 / pNonIdentity)
	}
	if pExact > 0 {
		exactStep = int(1 / pExact)
	}
	for i := range classes {
		switch {
		case exactStep > 0 && i%exactStep == 0:
			classes[i] = byte(codec.ClassNormalExact)
		case normalStep > 0 && i%normalStep == 0:
			classes[i] = byte(codec.ClassNormal)
		}
	}
	return classes
}

// ransRoundTrip6 encodes and decodes classes via the 6-symbol codec,
// asserting byte-exact equality.
func ransRoundTrip6(t *testing.T, classes []byte) {
	t.Helper()
	freqs := codec.RansCountFreqs6(classes)
	encoded := codec.RansEncode6(classes, freqs)
	decoded, err := codec.RansDecode6(encoded, freqs, len(classes))
	if err != nil {
		t.Fatalf("RansDecode6: %v", err)
	}
	if len(decoded) != len(classes) {
		t.Fatalf("length mismatch: got %d want %d", len(decoded), len(classes))
	}
	for i := range classes {
		if decoded[i] != classes[i] {
			t.Errorf("index %d: got %d want %d", i, decoded[i], classes[i])
		}
	}
}

// ransRoundTrip7 encodes and decodes classes via the 7-symbol codec,
// asserting byte-exact equality.
func ransRoundTrip7(t *testing.T, classes []byte) {
	t.Helper()
	freqs := codec.RansCountFreqs7(classes)
	encoded := codec.RansEncode7(classes, freqs)
	decoded, err := codec.RansDecode7(encoded, freqs, len(classes))
	if err != nil {
		t.Fatalf("RansDecode7: %v", err)
	}
	if len(decoded) != len(classes) {
		t.Fatalf("length mismatch: got %d want %d", len(decoded), len(classes))
	}
	for i := range classes {
		if decoded[i] != classes[i] {
			t.Errorf("index %d: got %d want %d", i, decoded[i], classes[i])
		}
	}
}

// TestRansEncode7EmptyInput checks that encoding nil/empty returns nil.
func TestRansEncode7EmptyInput(t *testing.T) {
	var freqs codec.RansFreqs7
	for i := range freqs {
		freqs[i] = 1
	}
	freqs[0] += uint32(4096 - 7)
	encoded := codec.RansEncode7(nil, freqs)
	if encoded != nil {
		t.Errorf("expected nil for empty input, got %d bytes", len(encoded))
	}
}

// TestRans7RoundTripIdentityOnly verifies all-ClassIdentity streams.
func TestRans7RoundTripIdentityOnly(t *testing.T) {
	classes := make([]byte, 5000)
	ransRoundTrip7(t, classes)
}

// TestRans7RoundTripMixedSymbols verifies round-trip with all 7 symbols present.
func TestRans7RoundTripMixedSymbols(t *testing.T) {
	classes := make([]byte, 7000)
	for i := range classes {
		classes[i] = byte(i % 7)
	}
	ransRoundTrip7(t, classes)
}

// TestRans7RoundTripNormal32Heavy verifies a stream dominated by ClassNormal32.
func TestRans7RoundTripNormal32Heavy(t *testing.T) {
	classes := make([]byte, 5000)
	for i := range classes {
		switch i % 10 {
		case 0:
			classes[i] = byte(codec.ClassIdentity)
		case 1:
			classes[i] = byte(codec.ClassNormalExact)
		case 2:
			classes[i] = byte(codec.ClassNormal32)
		default:
			classes[i] = byte(codec.ClassNormal)
		}
	}
	ransRoundTrip7(t, classes)
}

// TestRans7CompatWith6Symbol verifies that a stream containing only symbols
// 0-5 (no ClassNormal32) round-trips identically through both the 6-symbol
// and 7-symbol codecs. The 7-symbol codec must be a strict superset.
func TestRans7CompatWith6Symbol(t *testing.T) {
	// Build a stream with symbols 0–5 only (no symbol 6 = ClassNormal32).
	classes := makeIdentityStream(5000, 0.08)
	// Sprinkle some ClassNormalExact (5) — valid in both 6 and 7 symbol codecs.
	for i := range classes {
		if i%50 == 0 {
			classes[i] = byte(codec.ClassNormalExact)
		}
	}

	freqs6 := codec.RansCountFreqs6(classes)
	enc6 := codec.RansEncode6(classes, freqs6)
	dec6, err := codec.RansDecode6(enc6, freqs6, len(classes))
	if err != nil {
		t.Fatalf("6-sym decode: %v", err)
	}

	freqs7 := codec.RansCountFreqs7(classes)
	enc7 := codec.RansEncode7(classes, freqs7)
	dec7, err := codec.RansDecode7(enc7, freqs7, len(classes))
	if err != nil {
		t.Fatalf("7-sym decode: %v", err)
	}

	for i := range classes {
		if dec6[i] != classes[i] {
			t.Fatalf("6-sym mismatch at %d: got %d want %d", i, dec6[i], classes[i])
		}
		if dec7[i] != classes[i] {
			t.Fatalf("7-sym mismatch at %d: got %d want %d", i, dec7[i], classes[i])
		}
	}
}
