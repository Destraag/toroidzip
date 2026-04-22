// main_test.go is an integration test for the toroidzip CLI binary.
// It builds the binary, exercises the --sig-figs and --bytes encode paths,
// and verifies that the math.MaxFloat64 Tolerance wiring is effective:
// output must be meaningfully smaller than raw and decode accuracy must be
// within the expected sig-figs bound.
package main_test

import (
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func buildBinary(t *testing.T, dir string) string {
	t.Helper()
	ext := ""
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}
	out := filepath.Join(dir, "toroidzip"+ext)
	cmd := exec.Command("go", "build", "-o", out, ".")
	cmd.Dir = "." // cmd/toroidzip/
	if data, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build toroidzip: %v\n%s", err, data)
	}
	return out
}

func writeF64(t *testing.T, path string, values []float64) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, values); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func readF64(t *testing.T, path string) []float64 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	n := len(data) / 8
	values := make([]float64, n)
	for i := range values {
		bits := binary.LittleEndian.Uint64(data[i*8 : i*8+8])
		values[i] = math.Float64frombits(bits)
	}
	return values
}

func smoothSeries(n int, start, factor float64) []float64 {
	v := make([]float64, n)
	cur := start
	for i := range v {
		v[i] = cur
		cur *= factor
	}
	return v
}

// TestCLISigFigsWiring builds the CLI binary and verifies that --sig-figs 4
// produces a compressed stream smaller than raw and decodes within the 4
// sig-figs end-to-end accuracy guarantee.
// This guards the math.MaxFloat64 Tolerance wiring in main.go lines ~168/195.
func TestCLISigFigsWiring(t *testing.T) {
	tmp := t.TempDir()
	bin := buildBinary(t, tmp)

	values := smoothSeries(500, 100.0, 1.001)
	inputPath := filepath.Join(tmp, "input.f64")
	encPath := filepath.Join(tmp, "out.tzrz")
	decPath := filepath.Join(tmp, "decoded.f64")

	writeF64(t, inputPath, values)
	rawSize := int64(len(values)) * 8

	// Encode with --sig-figs 4.
	cmd := exec.Command(bin, "encode", "--sig-figs", "4", inputPath, encPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("encode --sig-figs 4: %v\n%s", err, out)
	}

	fi, err := os.Stat(encPath)
	if err != nil {
		t.Fatalf("stat encoded: %v", err)
	}
	// With math.MaxFloat64 tolerance, smooth data at 4 sig-figs should compress
	// well (mostly u8/u16 per ratio). With the old SigFigsToTolerance value,
	// every ratio became ClassNormalExact (8 bytes each) and the stream would
	// equal or exceed raw size.
	if fi.Size() >= rawSize/2 {
		t.Errorf("--sig-figs 4 compressed size %d >= rawSize/2 %d — likely Tolerance wiring broken",
			fi.Size(), rawSize/2)
	}

	// Decode and verify accuracy.
	cmd = exec.Command(bin, "decode", encPath, decPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("decode: %v\n%s", err, out)
	}

	got := readF64(t, decPath)
	if len(got) != len(values) {
		t.Fatalf("decoded len %d want %d", len(got), len(values))
	}
	// 4 sig-figs end-to-end: max relative error ≤ 5×10⁻⁴ (conservative).
	tol := 5e-4
	for i, v := range values {
		if v == 0 {
			continue
		}
		if e := math.Abs(got[i]-v) / math.Abs(v); e > tol {
			t.Errorf("value[%d]: rel err %e > %e", i, e, tol)
			break
		}
	}
}

// TestCLIBytesWiring verifies that --bytes 2 (u16 tier) also produces a
// meaningfully compressed output — same wiring path as --sig-figs.
func TestCLIBytesWiring(t *testing.T) {
	tmp := t.TempDir()
	bin := buildBinary(t, tmp)

	values := smoothSeries(500, 100.0, 1.001)
	inputPath := filepath.Join(tmp, "input.f64")
	encPath := filepath.Join(tmp, "out.tzrz")
	decPath := filepath.Join(tmp, "decoded.f64")

	writeF64(t, inputPath, values)
	rawSize := int64(len(values)) * 8

	cmd := exec.Command(bin, "encode", "--bytes", "2", inputPath, encPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("encode --bytes 2: %v\n%s", err, out)
	}

	fi, err := os.Stat(encPath)
	if err != nil {
		t.Fatalf("stat encoded: %v", err)
	}
	if fi.Size() >= rawSize/2 {
		t.Errorf("--bytes 2 compressed size %d >= rawSize/2 %d — likely Tolerance wiring broken",
			fi.Size(), rawSize/2)
	}

	cmd = exec.Command(bin, "decode", encPath, decPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("decode: %v\n%s", err, out)
	}

	got := readF64(t, decPath)
	if len(got) != len(values) {
		t.Fatalf("decoded len %d want %d", len(got), len(values))
	}
	// --bytes 2 ≈ 4 sig figs accuracy.
	tol := 5e-4
	for i, v := range values {
		if v == 0 {
			continue
		}
		if e := math.Abs(got[i]-v) / math.Abs(v); e > tol {
			t.Errorf("value[%d]: rel err %e > %e", i, e, tol)
			break
		}
	}
}
