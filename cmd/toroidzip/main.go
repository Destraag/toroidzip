// Command toroidzip is the CLI for the ToroidZip ratio-first float codec.
//
// Usage:
//
//	toroidzip encode <input.f64> <output.tzrz>
//	toroidzip decode <input.tzrz> <output.f64>
//
// Input/output files are raw IEEE 754 little-endian float64 sequences.
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/dbrjo/toroidzip/codec"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	var err error
	switch os.Args[1] {
	case "encode":
		err = runEncode(os.Args[2:])
	case "decode":
		err = runDecode(os.Args[2:])
	default:
		usage()
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func runEncode(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("encode requires <input.f64> <output.tzrz>")
	}
	values, err := readFloat64File(args[0])
	if err != nil {
		return fmt.Errorf("reading input: %w", err)
	}
	out, err := os.Create(args[1])
	if err != nil {
		return fmt.Errorf("creating output: %w", err)
	}
	defer out.Close()

	if err := codec.Encode(values, out, 0); err != nil {
		return fmt.Errorf("encoding: %w", err)
	}
	fmt.Printf("encoded %d values -> %s\n", len(values), args[1])
	return nil
}

func runDecode(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("decode requires <input.tzrz> <output.f64>")
	}
	in, err := os.Open(args[0])
	if err != nil {
		return fmt.Errorf("opening input: %w", err)
	}
	defer in.Close()

	values, err := codec.Decode(in)
	if err != nil {
		return fmt.Errorf("decoding: %w", err)
	}

	if err := writeFloat64File(args[1], values); err != nil {
		return fmt.Errorf("writing output: %w", err)
	}
	fmt.Printf("decoded %d values -> %s\n", len(values), args[1])
	return nil
}

func readFloat64File(path string) ([]float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data)%8 != 0 {
		return nil, fmt.Errorf("file size %d is not a multiple of 8 (not a float64 sequence)", len(data))
	}
	values := make([]float64, len(data)/8)
	for i := range values {
		bits := binary.LittleEndian.Uint64(data[i*8 : i*8+8])
		values[i] = math.Float64frombits(bits)
	}
	return values, nil
}

func writeFloat64File(path string, values []float64) error {
	data := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(data[i*8:i*8+8], math.Float64bits(v))
	}
	return os.WriteFile(path, data, 0644)
}

func usage() {
	fmt.Fprintln(os.Stderr, `toroidzip — ratio-first float64 codec

Usage:
  toroidzip encode <input.f64> <output.tzrz>   compress raw float64 file
  toroidzip decode <input.tzrz> <output.f64>   decompress to raw float64`)
}
