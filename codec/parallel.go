package codec

// parallel.go implements the streaming parallel encode pipeline (M17a/17c).
//
// Pipeline (all goroutines run concurrently from value 1 onward):
//
//	segmenter ──[segCh, depth 4×N]──► workers ──[resCh, depth N]──► assembler
//
// Architecture (M17c revision):
//
//   - Segmenter: periodic boundaries only (i%ReanchorInterval==0). O(1) per
//     value, no ratio computation. All encoding decisions belong to workers.
//
//   - Workers: own ALL encoding decisions — adaptive reanchor (gatherRans7v7
//     handles it inline), kCenter for v9 (segmentKCenter helper), encoding.
//     Workers operate independently on their sub-slice.
//
//   - Assembler: stitches ordered blobs. Prepends inter-segment ClassReanchor
//     or ClassReanchorDynamic. Inline ClassReanchor events within a blob's
//     class stream flow through unchanged.
//
// Restrictions:
//   - DriftCompensate and DriftQuantize are not supported in parallel mode.
//   - DynamicOffset + AdaptiveReanchor + EndToEndTolerance > 0 falls back to
//     serial: gatherRans7v8Segment does not support inline reanchoring, and
//     the combination produces very short segments with negligible savings.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"runtime"
	"sync"
)

// parallelSegmentBufferDepthFactor controls the segmenter's lookahead.
// Buffer depth = parallelSegmentBufferDepthFactor × N.
// The segmenter races ahead filling the buffer; it blocks only when full.
const parallelSegmentBufferDepthFactor = 4

// segmentDesc describes one independently-encodable segment.
// start is the anchor index (inclusive); end is exclusive.
// seqIdx is the 0-based ordinal used for ordered assembly.
// Encoding parameters (kCenter, isDynamic) are computed by the worker, not here.
type segmentDesc struct {
	start  int
	end    int
	seqIdx int
}

// segmentBlob holds the encoded output of one worker-processed segment.
type segmentBlob struct {
	seqIdx    int
	classes   []byte
	payloads  []byte
	kCenter   int32 // passed through from segmentDesc for assembler stitching
	isDynamic bool  // passed through from segmentDesc
}

// EncodeParallel compresses values using a parallel pipeline and writes to w.
// n is the worker count; n<=0 means runtime.NumCPU().
// When n==1 the call is forwarded to the single-threaded Encode path so that
// --parallel 1 produces byte-identical output to v1.0.0.
func EncodeParallel(values []float64, w io.Writer, opts EncodeOptions, n int) error {
	if len(values) == 0 {
		return fmt.Errorf("toroidzip: encode: empty input")
	}
	if opts.ReanchorInterval <= 0 {
		opts.ReanchorInterval = DefaultReanchorInterval
	}
	if opts.DriftMode != DriftReanchor {
		return fmt.Errorf("toroidzip: parallel encode: DriftCompensate and DriftQuantize are not supported in parallel mode; use --drift-mode reanchor (default)")
	}
	if n <= 0 {
		n = runtime.NumCPU()
	}
	if n == 1 {
		// Single-threaded path — byte-identical to v1.0.0.
		serial := opts
		serial.Parallelism = 0
		return Encode(values, w, serial)
	}

	// DynamicOffset + AdaptiveReanchor: gatherRans7v8Segment does not emit inline
	// ClassReanchor events, so EndToEndTolerance would be violated on long periodic
	// segments. The combination also yields negligible savings (short effective
	// segments defeat kCenter optimisation). Fall back to the serial path.
	if opts.DynamicOffset && opts.AdaptiveReanchor && opts.EndToEndTolerance > 0 {
		serial := opts
		serial.Parallelism = 0
		return Encode(values, w, serial)
	}

	bufDepth := parallelSegmentBufferDepthFactor * n
	segCh := make(chan segmentDesc, bufDepth)
	resCh := make(chan segmentBlob, n)

	go runSegmenter(values, opts, segCh)

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			runWorker(values, opts, segCh, resCh)
		}()
	}
	go func() {
		wg.Wait()
		close(resCh)
	}()

	return runAssembler(values, w, opts, resCh)
}

// runSegmenter emits periodic segment boundaries onto ch.
// It performs NO ratio computation — workers own all encoding decisions.
// The sole job is to cut the value stream at ReanchorInterval boundaries
// so workers can operate independently in parallel.
func runSegmenter(values []float64, opts EncodeOptions, ch chan<- segmentDesc) {
	defer close(ch)
	ri := opts.ReanchorInterval
	seqIdx := 0
	segStart := 0
	for i := 1; i < len(values); i++ {
		if i%ri == 0 {
			ch <- segmentDesc{start: segStart, end: i, seqIdx: seqIdx}
			seqIdx++
			segStart = i
		}
	}
	ch <- segmentDesc{start: segStart, end: len(values), seqIdx: seqIdx}
}

// segmentKCenter computes the optimal kCenter and isDynamic flag for a v9
// (DynamicOffset) worker. It pre-scans the sub-slice to collect Q symbols,
// then calls OptimalKCenter and checks the cost threshold.
//
// Segment 0 (seqIdx==0) always returns defaultCenter/false — the v9 decoder
// uses defaultCenter unconditionally for the first anchor.
//
// Design note — why we do NOT split segments when kCenter shifts:
//
// kCenter optimisation amortises a 4-byte storage cost over the segment's Q
// symbols. A kCenter shift mid-segment would cut the segment in two, reducing
// the symbol count available for amortisation in both halves. The break-even
// point is already ~12 symbols (anchor + kCenter cost); halving the segment
// makes it worse, not better.
//
// Additionally, detecting an impending kCenter shift requires a forward scan
// of the next segment's symbols — O(2n) per segment and incompatible with the
// streaming worker design. The adaptive reanchor path (error-triggered, owned
// by gatherRans7v7) already fires on the same data conditions (sustained drift)
// that would motivate a kCenter split. There is no observable gap between the
// two signals on real data, so a separate kCenter-change trigger adds complexity
// with no measurable benefit.
func segmentKCenter(sub []float64, opts EncodeOptions, seqIdx int) (kCenter int32, isDynamic bool) {
	bits := opts.PrecisionBits
	if bits <= 0 {
		bits = DefaultPrecisionBits
	}
	defaultCenter := int32(1) << (bits - 1)
	if seqIdx == 0 {
		return defaultCenter, false
	}
	tol := opts.Tolerance
	levels := uint32(1) << bits
	deltaB := math.Pow(2, QuantMaxLog2R/float64(levels)) - 1
	fastPath := tol > 0 && deltaB < tol

	qSyms := make([]int32, 0, len(sub))
	prev := sub[0]
	for i := 1; i < len(sub); i++ {
		ratio, class := computeRatio(sub[i], prev)
		switch class {
		case ClassNormal16:
			if ratio == 0 {
				prev = prev * ratio
				continue
			}
			defOff := QuantizeRatioOffset(ratio, bits)
			dequant := DequantizeRatioOffset(defOff, bits)
			if fastPath || deltaB < tol || math.Abs(dequant/ratio-1.0) < tol {
				qSyms = append(qSyms, defOff+defaultCenter)
			}
			prev = prev * dequant
		case ClassIdentity:
			// prev unchanged
		default: // ClassBoundaryZero, ClassBoundaryInf
			prev = sub[i]
		}
	}
	if len(qSyms) == 0 {
		return defaultCenter, false
	}
	opt := OptimalKCenter(qSyms, bits)
	if opt == defaultCenter || DynOffsetCost(qSyms, defaultCenter)-DynOffsetCost(qSyms, opt) <= dynOffsetThreshold {
		return defaultCenter, false
	}
	return opt, true
}

// runWorker reads segmentDesc from segCh and encodes each segment.
// Workers own ALL encoding decisions:
//   - EntropyAdaptive: gatherRans7v7 handles adaptive reanchor inline.
//   - DynamicOffset: segmentKCenter computes optimal kCenter before encoding.
func runWorker(values []float64, opts EncodeOptions, segCh <-chan segmentDesc, resCh chan<- segmentBlob) {
	for seg := range segCh {
		sub := values[seg.start:seg.end]
		var classes []byte
		var payloads []byte
		var kCenter int32
		var isDynamic bool
		switch opts.EntropyMode {
		case EntropyQuantized:
			classes, payloads = gatherRansV4(sub, opts)
		case EntropyAdaptive:
			if opts.DynamicOffset {
				kCenter, isDynamic = segmentKCenter(sub, opts, seg.seqIdx)
				classes, payloads = gatherRans7v8Segment(sub, opts, kCenter)
			} else {
				classes, payloads = gatherRans7v7(sub, opts)
			}
		}
		resCh <- segmentBlob{
			seqIdx:    seg.seqIdx,
			classes:   classes,
			payloads:  payloads,
			kCenter:   kCenter,
			isDynamic: isDynamic,
		}
	}
}

// runAssembler collects blobs from resCh (potentially out of order),
// reorders them by seqIdx, merges class and payload streams, and writes
// the final encoded stream to w.
//
// Assembly rules:
//   - Segment 0: classes and payloads written as-is (anchor at payload[0:8]).
//   - Segment k>0, !isDynamic: prepend ClassReanchor; payloads written as-is.
//   - Segment k>0, isDynamic:  prepend ClassReanchorDynamic; payload layout is
//     [anchor_8b][kCenter_4b][event_payloads], so kCenter is spliced in after
//     the first 8 bytes of the blob's payload.
func runAssembler(values []float64, w io.Writer, opts EncodeOptions, resCh <-chan segmentBlob) error {
	// Collect all blobs.
	var blobs []segmentBlob
	for blob := range resCh {
		blobs = append(blobs, blob)
	}

	// Sort by seqIdx to restore segment order.
	// Simple insertion sort is fine — number of segments ≤ len(values)/ri.
	for i := 1; i < len(blobs); i++ {
		for j := i; j > 0 && blobs[j].seqIdx < blobs[j-1].seqIdx; j-- {
			blobs[j], blobs[j-1] = blobs[j-1], blobs[j]
		}
	}

	// Merge class and payload streams.
	totalClasses := 0
	for _, b := range blobs {
		totalClasses += len(b.classes)
	}
	// Non-first segments each contribute one reanchor class event (1 byte each).
	totalClasses += len(blobs) - 1

	mergedClasses := make([]byte, 0, totalClasses)
	var mergedPayloads bytes.Buffer

	dynOffset := opts.DynamicOffset && opts.EntropyMode == EntropyAdaptive

	for k, blob := range blobs {
		if k > 0 {
			if dynOffset && blob.isDynamic {
				// ClassReanchorDynamic payload: anchor(8) + kCenter(4) + event payloads.
				// blob.payloads starts with the anchor (8 bytes) from gatherRans7v8Segment;
				// splice kCenter in after it.
				mergedClasses = append(mergedClasses, byte(ClassReanchorDynamic))
				mergedPayloads.Write(blob.payloads[:8]) // anchor float64
				var kc [4]byte
				binary.LittleEndian.PutUint32(kc[:], uint32(blob.kCenter))
				mergedPayloads.Write(kc[:])             // kCenter int32
				mergedPayloads.Write(blob.payloads[8:]) // event payloads
			} else {
				mergedClasses = append(mergedClasses, byte(ClassReanchor))
				mergedPayloads.Write(blob.payloads)
			}
		} else {
			mergedPayloads.Write(blob.payloads)
		}
		mergedClasses = append(mergedClasses, blob.classes...)
	}

	// Write the stream using the appropriate header writer.
	bits := opts.PrecisionBits
	if bits <= 0 {
		bits = DefaultPrecisionBits
	}

	switch opts.EntropyMode {
	case EntropyQuantized:
		freqs := RansCountFreqs9(mergedClasses)
		return writeQuantizedV2Stream(values, w, opts, bits, mergedClasses, freqs, mergedPayloads.Bytes())
	case EntropyAdaptive:
		if dynOffset {
			freqs10 := RansCountFreqs10(mergedClasses)
			return writeAdaptiveV8Stream(values, w, opts, bits, mergedClasses, freqs10, mergedPayloads.Bytes())
		}
		freqs := RansCountFreqs9(mergedClasses)
		return writeAdaptiveV7Stream(values, w, opts, bits, mergedClasses, freqs, mergedPayloads.Bytes())
	default:
		return fmt.Errorf("toroidzip: parallel assemble: unknown entropy mode %d", opts.EntropyMode)
	}
}

// writeAdaptiveV8Stream writes the v9 header + rANS body for a pre-assembled
// class and payload stream. Used by the parallel assembler for DynamicOffset streams.
// Header: magic(4)+version=9(1)+driftMode(1)+reanchorInterval(4)+count(8)+precisionBits(1)+ransFreqs10(40) = 59 bytes.
func writeAdaptiveV8Stream(values []float64, w io.Writer, opts EncodeOptions, bits int, classes []byte, freqs RansFreqs10, payloads []byte) error {
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, versionAdaptiveV8); err != nil {
		return err
	}
	if err := writeByte(w, byte(opts.DriftMode)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(opts.ReanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}
	if err := writeByte(w, byte(bits)); err != nil {
		return err
	}
	return writeRansBody10(w, classes, freqs, payloads)
}

// writeQuantizedV2Stream writes the v7 header + rANS body for a pre-assembled
// class and payload stream. Shared by single-threaded and parallel paths.
func writeQuantizedV2Stream(values []float64, w io.Writer, opts EncodeOptions, bits int, classes []byte, freqs RansFreqs9, payloads []byte) error {
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, versionQuantizedV2); err != nil {
		return err
	}
	if err := writeByte(w, byte(opts.DriftMode)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(opts.ReanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}
	if err := writeByte(w, byte(bits)); err != nil {
		return err
	}
	return writeRansBody9(w, classes, freqs, payloads)
}

// writeAdaptiveV7Stream writes the v8 header + rANS body for a pre-assembled
// class and payload stream. Shared by single-threaded and parallel paths.
func writeAdaptiveV7Stream(values []float64, w io.Writer, opts EncodeOptions, bits int, classes []byte, freqs RansFreqs9, payloads []byte) error {
	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := writeByte(w, versionAdaptiveV7); err != nil {
		return err
	}
	if err := writeByte(w, byte(opts.DriftMode)); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(opts.ReanchorInterval)); err != nil {
		return err
	}
	if err := writeUint64(w, uint64(len(values))); err != nil {
		return err
	}
	if err := writeByte(w, byte(bits)); err != nil {
		return err
	}
	return writeRansBody9(w, classes, freqs, payloads)
}
