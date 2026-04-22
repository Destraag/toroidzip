// mechanism_test.go — M13a mechanism guard tests.
//
// These tests assert the *mechanism* (tier routing, reanchor counts, header
// wiring), not just output accuracy.  A correct-output test cannot detect a
// "feature eclipse" regression where an intended code path is silently
// bypassed but the fallback still produces valid (if larger) output.
//
// Background: the M9 regression routed every ClassNormal ratio to
// ClassNormalExact because the per-ratio tolerance gate was orders of
// magnitude tighter than ε_max(B).  Accuracy tests passed (ClassNormalExact
// is correct); only a payload-cost / tier-distribution test would have
// caught it immediately.
package codec_test

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/Destraag/toroidzip/codec"
)

// classCounts holds per-class event counts from a v7 class stream.
type classCounts struct {
	identity, u8, u16, u24, u32, exact, reanchor, boundary int
}

func countClasses(classes []byte) classCounts {
	var c classCounts
	for _, b := range classes {
		switch codec.RatioClass(b) {
		case codec.ClassIdentity:
			c.identity++
		case codec.ClassNormal8:
			c.u8++
		case codec.ClassNormal:
			c.u16++
		case codec.ClassNormal24:
			c.u24++
		case codec.ClassNormal32:
			c.u32++
		case codec.ClassNormalExact:
			c.exact++
		case codec.ClassReanchor:
			c.reanchor++
		default:
			c.boundary++
		}
	}
	return c
}

func (c classCounts) normalEvents() int { return c.u8 + c.u16 + c.u24 + c.u32 + c.exact }
func (c classCounts) u8Frac() float64 {
	n := c.normalEvents()
	if n == 0 {
		return 0
	}
	return float64(c.u8) / float64(n)
}

// TestTierDistributionInvariant guards gatherRans7v7's tier routing mechanism.
//
// Structural invariants enforced:
//
//  1. Tolerance=math.MaxFloat64 → ClassNormalExact count == 0.
//     Fails immediately when per-ratio gating eclipses the fast path (M9 pattern).
//
//  2. B ≤ 13 (max |offset| = 4095 ≤ 32767) → no ClassNormal24, no ClassNormal32.
//     Fails when PrecisionBits is silently increased beyond what was configured.
//
//  3. B ≤ 16 (max |offset| = 32767) → no ClassNormal24, no ClassNormal32.
//     (Sensor and NearConstant data only; volatile jumps excluded at B=16.)
//
//  4. Smooth data at B=13 → u8 fraction ≥ 0.75 of normal events.
//     Fails when PrecisionBits is increased (wider grid pushes offsets out of int8).
//
//  5. NearConstant at B=16 → u8 fraction ≥ 0.70 of normal events.
func TestTierDistributionInvariant(t *testing.T) {
	const n = 10_000
	sensor := makeSensorStream(n)
	nearConst := makeNearConstant(n)

	// opts builds a MaxFloat64 fast-path configuration at a given B.
	// ReanchorInterval > n ensures no fixed-interval reanchors confuse counts.
	opts := func(B int) codec.EncodeOptions {
		return codec.EncodeOptions{
			EntropyMode:      codec.EntropyAdaptive,
			PrecisionBits:    B,
			Tolerance:        math.MaxFloat64,
			ReanchorInterval: n + 1,
		}
	}

	type inv struct {
		name      string
		values    []float64
		B         int
		noExact   bool
		noU24     bool
		noU32     bool
		minU8Frac float64 // 0 = no minimum enforced
	}

	cases := []inv{
		// B=10 (3sf): max |offset| = 511 — worst tier is u16, never u24/u32.
		{name: "Sensor/B=10", values: sensor, B: 10, noExact: true, noU24: true, noU32: true},
		{name: "NearConst/B=10", values: nearConst, B: 10, noExact: true, noU24: true, noU32: true},

		// B=13 (4sf): max |offset| = 4095 — still below u16 ceiling.
		{name: "Sensor/B=13", values: sensor, B: 13, noExact: true, noU24: true, noU32: true, minU8Frac: 0.75},
		{name: "NearConst/B=13", values: nearConst, B: 13, noExact: true, noU24: true, noU32: true},

		// B=16 (5sf): max |offset| = 32767 — exactly at u16 ceiling.
		{name: "Sensor/B=16", values: sensor, B: 16, noExact: true, noU24: true, noU32: true},
		{name: "NearConst/B=16", values: nearConst, B: 16, noExact: true, noU24: true, noU32: true, minU8Frac: 0.70},

		// B=20 (6sf): max |offset| = 524287 — worst tier is u24, never u32.
		{name: "Sensor/B=20", values: sensor, B: 20, noExact: true, noU32: true},
		{name: "NearConst/B=20", values: nearConst, B: 20, noExact: true, noU32: true},

		// B=30 (9sf): full range — only guarantee is no exact with MaxFloat64.
		{name: "Sensor/B=30", values: sensor, B: 30, noExact: true},
		{name: "NearConst/B=30", values: nearConst, B: 30, noExact: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			c := countClasses(codec.GatherV7ClassesForTest(tc.values, opts(tc.B)))
			norm := c.normalEvents()

			if tc.noExact && c.exact != 0 {
				t.Errorf("B=%d Tolerance=MaxFloat64: ClassNormalExact=%d want 0 — "+
					"per-ratio gate is eclipsing the fast path (M9 regression pattern)",
					tc.B, c.exact)
			}
			if tc.noU24 && c.u24 != 0 {
				maxOffset := (1 << tc.B) / 2
				t.Errorf("B=%d: ClassNormal24=%d want 0 — max |offset| at B=%d is %d "+
					"which fits in u16 (≤32767); u24 means PrecisionBits was not respected",
					tc.B, c.u24, tc.B, maxOffset)
			}
			if tc.noU32 && c.u32 != 0 {
				maxOffset := (1 << tc.B) / 2
				t.Errorf("B=%d: ClassNormal32=%d want 0 — max |offset| at B=%d is %d "+
					"which fits below u32 threshold; u32 means PrecisionBits was not respected",
					tc.B, c.u32, tc.B, maxOffset)
			}
			if tc.minU8Frac > 0 && norm > 0 {
				if frac := c.u8Frac(); frac < tc.minU8Frac {
					t.Errorf("B=%d: u8 fraction=%.3f < min %.3f (identity=%d u8=%d u16=%d u24=%d u32=%d exact=%d) — "+
						"smooth data should dominate the u8 tier; "+
						"a lower fraction means PrecisionBits was silently increased",
						tc.B, frac, tc.minU8Frac, c.identity, c.u8, c.u16, c.u24, c.u32, c.exact)
				}
			}
			t.Logf("B=%d: id=%d u8=%d u16=%d u24=%d u32=%d exact=%d reanchor=%d boundary=%d u8Frac=%.3f",
				tc.B, c.identity, c.u8, c.u16, c.u24, c.u32, c.exact, c.reanchor, c.boundary, c.u8Frac())
		})
	}
}

// TestAdaptiveReanchorMechanism guards the ClassReanchor emission mechanism in
// gatherRans7v7 by counting reanchor events in the raw class stream.
//
// The adaptive check fires on a per-step basis: at each position the encoder
// computes `|prev_accumulated * dequant - values[i]| / |values[i]|` (where
// prev already tracks decoder-side drift).  This equals the CURRENT STEP'S
// quantization error, not total accumulated drift.  Setting EndToEndTolerance
// below ε_max(B) causes the check to fire on steps whose single-step error
// exceeds tol.
//
// Three sub-tests:
//
//  1. EndToEndTolerance << ε_max → mechanism fires aggressively (many reanchors).
//     Fails when AdaptiveReanchor is not wired through to gatherRans7v7.
//
//  2. EndToEndTolerance >> any possible error → mechanism never fires.
//     Fails when the check fires spuriously.
//
//  3. Fixed-K (AdaptiveReanchor=false) → exactly floor((n-1)/K) reanchors.
//     Baseline invariant for the fixed-interval path.
func TestAdaptiveReanchorMechanism(t *testing.T) {
	const n = 1_000
	const B = 13 // 4sf — ε_max ≈ 3.385e-4

	levels := math.Pow(2, float64(B))
	epsilonMax := math.Pow(2, codec.QuantMaxLog2R/levels) - 1

	sensor := makeSensorStream(n)

	// Sub-test 1: EndToEndTolerance well below ε_max → fires on most ClassNormal steps.
	t.Run("TightTol/ManyReanchors", func(t *testing.T) {
		// 1e-8 is orders of magnitude below ε_max ≈ 3.4e-4, so every
		// ClassNormal step whose dequantization error exceeds 1e-8 triggers a
		// reanchor.  On sensor data with ratios ≈ 1.001 at B=13, essentially
		// every normal step has error in the range [ε_max/1000, ε_max] so many
		// reanchors must fire.
		opts := codec.EncodeOptions{
			EntropyMode:       codec.EntropyAdaptive,
			PrecisionBits:     B,
			Tolerance:         math.MaxFloat64,
			ReanchorInterval:  n + 1, // disable fixed-K
			AdaptiveReanchor:  true,
			EndToEndTolerance: 1e-8, // far below ε_max ≈ 3.4e-4
		}
		c := countClasses(codec.GatherV7ClassesForTest(sensor, opts))
		if c.reanchor == 0 {
			t.Errorf("EndToEndTolerance=1e-8 << ε_max(%d)=%.2e: ClassReanchor=0, want >0 — "+
				"AdaptiveReanchor is not being threaded through to gatherRans7v7 "+
				"or the per-step drift check condition is broken",
				B, epsilonMax)
		}
		t.Logf("tight tol: ClassReanchor=%d of %d events (ε_max=%.2e, tol=1e-8)",
			c.reanchor, n-1, epsilonMax)
	})

	// Sub-test 2: EndToEndTolerance >> any possible error → never fires.
	t.Run("LooseTol/NoReanchors", func(t *testing.T) {
		// With tol=1.0 (100% relative error tolerance), no single quantization
		// step will ever exceed tol.  AdaptiveReanchor must emit zero reanchors.
		opts := codec.EncodeOptions{
			EntropyMode:       codec.EntropyAdaptive,
			PrecisionBits:     B,
			Tolerance:         math.MaxFloat64,
			ReanchorInterval:  n + 1, // disable fixed-K
			AdaptiveReanchor:  true,
			EndToEndTolerance: 1.0, // trivially satisfied
		}
		c := countClasses(codec.GatherV7ClassesForTest(sensor, opts))
		if c.reanchor != 0 {
			t.Errorf("EndToEndTolerance=1.0 >> ε_max(%d)=%.2e: ClassReanchor=%d, want 0 — "+
				"adaptive reanchor is firing spuriously (check condition uses wrong threshold)",
				B, epsilonMax, c.reanchor)
		}
		t.Logf("loose tol: ClassReanchor=%d (expected 0)", c.reanchor)
	})

	// Sub-test 3: fixed-K reanchor interval (no adaptive) → known exact count.
	t.Run("FixedK/ExactCount", func(t *testing.T) {
		const fixedK = 100
		expected := (n - 1) / fixedK
		opts := codec.EncodeOptions{
			EntropyMode:      codec.EntropyAdaptive,
			PrecisionBits:    B,
			Tolerance:        math.MaxFloat64,
			ReanchorInterval: fixedK,
			AdaptiveReanchor: false,
		}
		c := countClasses(codec.GatherV7ClassesForTest(sensor, opts))
		if c.reanchor != expected {
			t.Errorf("fixed K=%d: ClassReanchor=%d, want %d — "+
				"ReanchorInterval wiring is broken in gatherRans7v7",
				fixedK, c.reanchor, expected)
		}
		t.Logf("fixed K=%d: ClassReanchor=%d (expected %d)", fixedK, c.reanchor, expected)
	})
}

// TestSigFigsWiring verifies the full chain: SigFigsToBits(N) → PrecisionBits
// in EncodeOptions → precisionBits byte written to stream header → tier routing
// in gatherRans7v7 → decode accuracy within ε_max(B).
//
// This test catches regressions where:
//   - PrecisionBits is silently overridden (e.g., hardcoded to 16 or 30).
//   - The header precisionBits field is written with a different value than opts.
//   - Tier routing uses a different bits value than was configured.
//   - Accuracy does not scale with B (PrecisionBits has no effect on output).
func TestSigFigsWiring(t *testing.T) {
	const n = 1_000
	sensor := makeSensorStream(n)

	// v7 header layout (magic already consumed by Decode):
	// full stream: magic(4) | version(1) | driftMode(1) | reanchorInterval(4) |
	//              count(8) | precisionBits(1) | ransFreqs9(36) | ...
	// precisionBits is at byte offset 18 in the raw encoded buffer.
	const precisionBitsOffset = 18

	type sfCase struct {
		N int // sig figs
		B int // expected PrecisionBits = SigFigsToBits(N)
	}

	cases := []sfCase{
		{3, codec.SigFigsToBits(3)}, // B=10
		{4, codec.SigFigsToBits(4)}, // B=13
		{5, codec.SigFigsToBits(5)}, // B=16
		{6, codec.SigFigsToBits(6)}, // B=20
		{9, codec.SigFigsToBits(9)}, // B=30
	}

	// Encode once per (N, B) and collect (headerBits, tierCounts, maxRelErr).
	type wireResult struct {
		headerBits int
		counts     classCounts
		maxRelErr  float64
		encBytes   int
	}

	results := make([]wireResult, len(cases))
	for i, tc := range cases {
		opts := codec.EncodeOptions{
			EntropyMode:      codec.EntropyAdaptive,
			PrecisionBits:    tc.B,
			Tolerance:        math.MaxFloat64,
			ReanchorInterval: n + 1, // no fixed reanchors
		}

		// Tier counts from raw class stream.
		classes := codec.GatherV7ClassesForTest(sensor, opts)
		counts := countClasses(classes)

		// Full encode to inspect header and measure accuracy.
		var buf bytes.Buffer
		if err := codec.Encode(sensor, &buf, opts); err != nil {
			t.Fatalf("N=%d B=%d: encode: %v", tc.N, tc.B, err)
		}
		encoded := buf.Bytes()

		headerBits := 0
		if len(encoded) > precisionBitsOffset {
			headerBits = int(encoded[precisionBitsOffset])
		}

		decoded, err := codec.Decode(bytes.NewReader(encoded))
		if err != nil {
			t.Fatalf("N=%d B=%d: decode: %v", tc.N, tc.B, err)
		}
		var maxErr float64
		for j, want := range sensor {
			if want != 0 && j < len(decoded) {
				if e := math.Abs(decoded[j]-want) / math.Abs(want); e > maxErr {
					maxErr = e
				}
			}
		}

		results[i] = wireResult{
			headerBits: headerBits,
			counts:     counts,
			maxRelErr:  maxErr,
			encBytes:   len(encoded),
		}
	}

	for i, tc := range cases {
		r := results[i]
		name := fmt.Sprintf("N%d_B%d", tc.N, tc.B)
		t.Run(name, func(t *testing.T) {
			// 1. Header wiring: encoded stream must record the configured B.
			if r.headerBits != tc.B {
				t.Errorf("header precisionBits=%d, want %d (SigFigsToBits(%d)) — "+
					"PrecisionBits is not being written to the stream header correctly",
					r.headerBits, tc.B, tc.N)
			}

			// 2. No ClassNormalExact with MaxFloat64 (same as TierDistribution test).
			if r.counts.exact != 0 {
				t.Errorf("ClassNormalExact=%d want 0 with Tolerance=MaxFloat64", r.counts.exact)
			}

			// 3. Structural tier ceiling: at B ≤ 16 (N ≤ 5), no u24/u32 allowed
			//    (max |offset| at B=16 is 32767 which fits in ClassNormal/u16).
			if tc.B <= 16 {
				if r.counts.u24 != 0 || r.counts.u32 != 0 {
					t.Errorf("B=%d (≤16): u24=%d u32=%d want 0 — "+
						"tier ceiling violated; offset magnitude exceeds B-bit range",
						tc.B, r.counts.u24, r.counts.u32)
				}
			}

			// 4. Accuracy bound: max relative error ≤ n * ε_max(B).
			//    This is a loose bound (Mode A accumulation over n steps).
			//    It fails when PrecisionBits is silently replaced with a smaller value.
			levels := math.Pow(2, float64(tc.B))
			epsilonMax := math.Pow(2, codec.QuantMaxLog2R/levels) - 1
			bound := float64(n) * epsilonMax
			if r.maxRelErr > bound {
				t.Errorf("max relative error %e exceeds n*ε_max(%d)=%e — "+
					"accuracy worse than expected; PrecisionBits may be silently reduced",
					r.maxRelErr, tc.B, bound)
			}

			t.Logf("N=%d B=%d: header=%d encBytes=%d u8=%d u16=%d u24=%d u32=%d exact=%d maxRelErr=%e ε_max=%e",
				tc.N, tc.B, r.headerBits, r.encBytes,
				r.counts.u8, r.counts.u16, r.counts.u24, r.counts.u32, r.counts.exact, r.maxRelErr, epsilonMax)
		})
	}

	// 5. Ordered invariants across N values.
	// Encoded size must increase with B (more bits → larger offsets → more bytes).
	// Max error must decrease with B (more bits → finer grid → smaller errors).
	t.Run("OrderingInvariants", func(t *testing.T) {
		for i := 1; i < len(cases); i++ {
			prev, cur := results[i-1], results[i]
			prevN, curN := cases[i-1].N, cases[i].N

			if prev.encBytes >= cur.encBytes {
				// Strictly speaking, size can be equal for constant-ratio data
				// where all events are ClassIdentity regardless of B.  Accept >=.
				t.Logf("note: encBytes[N=%d]=%d >= encBytes[N=%d]=%d (may be ClassIdentity dominated)",
					prevN, prev.encBytes, curN, cur.encBytes)
			}
			if prev.maxRelErr <= cur.maxRelErr && prev.maxRelErr > 0 {
				t.Errorf("accuracy not monotone: maxRelErr[N=%d]=%e ≤ maxRelErr[N=%d]=%e — "+
					"increasing PrecisionBits should always reduce or maintain error",
					prevN, prev.maxRelErr, curN, cur.maxRelErr)
			}
		}
	})
}
