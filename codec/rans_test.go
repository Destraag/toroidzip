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
