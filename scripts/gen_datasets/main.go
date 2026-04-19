// gen_datasets writes six synthetic float64 datasets to a target directory as
// raw little-endian binary files (extension .f64).  Each file contains n
// IEEE-754 float64 values with no header.
//
// Usage:
//
//	go run ./scripts/gen_datasets [--n N] [--out DIR]
//
// Defaults: n=50000, out=./bench_data
//
// The datasets mirror the generators in codec/bench_test.go exactly so that
// M4 external-baseline results are directly comparable to the M3 harness.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

func main() {
	n := flag.Int("n", 50_000, "number of float64 values per dataset")
	out := flag.String("out", "bench_data", "output directory")
	flag.Parse()

	if err := os.MkdirAll(*out, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "gen_datasets: mkdir %s: %v\n", *out, err)
		os.Exit(1)
	}

	datasets := []struct {
		name string
		gen  func(int) []float64
	}{
		{"sensor", makeSensorStream},
		{"financial", makeFinancialWalk},
		{"multiscale", makeScientificMultiScale},
		{"volatile", makeVolatileSeries},
		{"nearconstant", makeNearConstant},
		{"neuralweight", makeNeuralWeightProxy},
	}

	for _, ds := range datasets {
		values := ds.gen(*n)
		path := filepath.Join(*out, ds.name+".f64")
		if err := writeF64(path, values); err != nil {
			fmt.Fprintf(os.Stderr, "gen_datasets: write %s: %v\n", path, err)
			os.Exit(1)
		}
		fmt.Printf("wrote %s (%d values, %d bytes)\n", path, len(values), len(values)*8)
	}
}

// writeF64 writes values as little-endian IEEE-754 float64 binary.
func writeF64(path string, values []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	buf := make([]byte, 8)
	for _, v := range values {
		binary.LittleEndian.PutUint64(buf, math.Float64bits(v))
		if _, err := f.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

// ─── Dataset generators (mirrors codec/bench_test.go exactly) ──────────────

type lcg struct{ state uint64 }

func newLCG(seed uint64) lcg { return lcg{seed | 1} }

func (l *lcg) float() float64 {
	l.state = l.state*6364136223846793005 + 1442695040888963407
	return float64(int64(l.state>>11)) / float64(1<<53)
}

func makeSensorStream(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0xDEAD_BEEF)
	v := 100.0
	for i := range out {
		drift := 0.2 * math.Sin(2*math.Pi*float64(i)/float64(n/3))
		noise := rng.float() * 0.002
		v *= math.Exp(drift/float64(n)*6 + noise)
		out[i] = v
	}
	return out
}

func makeFinancialWalk(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0xF17A4CE0)
	v := 100.0
	for i := range out {
		u1 := math.Abs(rng.float()) + 1e-9
		if u1 > 1 {
			u1 = 1 - 1e-9
		}
		u2 := rng.float() + 0.5001
		normal := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		ret := normal * 0.005
		jump := rng.float()
		if jump < -0.49 {
			ret += 0.15
		} else if jump > 0.49 {
			ret -= 0.15
		}
		v *= math.Exp(ret)
		if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			v = 100.0
		}
		out[i] = v
	}
	return out
}

func makeScientificMultiScale(n int) []float64 {
	out := make([]float64, n)
	v := 1e-6
	for i := range out {
		phase := 2 * math.Pi * float64(i) / float64(n)
		logTarget := 6 * math.Sin(phase)
		logCurrent := math.Log10(v)
		v *= math.Pow(10, (logTarget-logCurrent)*0.01)
		out[i] = v
	}
	return out
}

func makeVolatileSeries(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0xBEEF_CAFE)
	v := 1.0
	for i := range out {
		r := rng.float() + 0.5001
		if r < 0.01 {
			v *= 10
		} else if r < 0.02 {
			v *= 0.1
		} else {
			v *= 1.0 + (rng.float())*0.0002
		}
		if v <= 0 {
			v = 1e-6
		}
		out[i] = v
	}
	return out
}

func makeNearConstant(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0xC0FFEE42)
	v := 273.15
	for i := range out {
		v *= 1 + rng.float()*0.0001
		out[i] = v
	}
	return out
}

func makeNeuralWeightProxy(n int) []float64 {
	out := make([]float64, n)
	rng := newLCG(0x1337_C0DE)
	centres := [4]float64{0.001, 0.05, -0.03, 0.12}
	for i := range out {
		layer := (i / (n / 4)) % 4
		u1 := math.Abs(rng.float())*0.99 + 0.01
		u2 := rng.float() + 0.5001
		normal := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		v := centres[layer] + normal*0.01
		if v == 0 || math.IsNaN(v) {
			v = 1e-9
		}
		out[i] = v
	}
	return out
}
