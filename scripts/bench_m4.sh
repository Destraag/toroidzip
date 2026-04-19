#!/usr/bin/env bash
# M4 External Baseline Benchmark
# Compares ToroidZip (best mode per M3) against external codecs on the 6
# synthetic datasets used in the M3 harness.
#
# External tools (skipped with a warning if not found):
#   zstd    — https://github.com/facebook/zstd  (brew/apt: zstd)
#   brotli  — https://github.com/google/brotli  (brew/apt: brotli)
#   fpzip   — https://github.com/LLNL/fpzip     (build from source)
#
# Usage:
#   chmod +x scripts/bench_m4.sh
#   ./scripts/bench_m4.sh [--n N] [--keep]
#
# Flags:
#   --n N          number of float64 values per dataset (default 50000)
#   --sig-figs N   test only this precision (default: test 3, 6, and 9 sig figs)
#   --keep         keep the bench_data/ temp directory after the run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N=50000
KEEP=0
PRECISIONS=(3 6 9)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --sig-figs) PRECISIONS=("$2"); shift 2 ;;
    --keep) KEEP=1; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

BENCH_DIR="$REPO_ROOT/bench_data"
UNCOMPRESSED_BYTES=$(( N * 8 ))

# ── helpers ──────────────────────────────────────────────────────────────────

require_tool() {
  local tool="$1" install_hint="$2"
  if ! command -v "$tool" &>/dev/null; then
    echo "  [SKIP] $tool not found. Install: $install_hint" >&2
    return 1
  fi
  return 0
}

file_bytes() { wc -c < "$1" | tr -d ' '; }

ratio() {
  local enc="$1"
  awk "BEGIN { printf \"%.4f\", $enc / $UNCOMPRESSED_BYTES }"
}

time_compress() {
  # Returns wall-clock seconds (3 runs, take median via sort)
  local cmd="$1" out="$2"
  local times=()
  for _ in 1 2 3; do
    local t0 t1
    t0=$(date +%s%3N)
    eval "$cmd" >/dev/null 2>/dev/null
    t1=$(date +%s%3N)
    times+=( $(( t1 - t0 )) )
  done
  # Sort and pick middle value (median of 3)
  local sorted
  sorted=$(printf '%s\n' "${times[@]}" | sort -n)
  echo "$sorted" | awk 'NR==2'
}

mb_per_sec() {
  local ms="$1"
  if [[ "$ms" -eq 0 ]]; then echo "N/A"; return; fi
  awk "BEGIN { printf \"%.1f\", ($UNCOMPRESSED_BYTES / 1048576) / ($ms / 1000) }"
}

# ── preamble ─────────────────────────────────────────────────────────────────

echo ""
echo "## M4 External Baseline Benchmark — n=$N values ($(( UNCOMPRESSED_BYTES / 1024 )) KB uncompressed)"
echo ""
printf "ToroidZip modes tested: %s(Reanchor; use --sig-figs N to test one precision)\n" \
  "$(printf "Q-%ssf " "${PRECISIONS[@]}")"
echo "Ratio = encoded_bytes / uncompressed_bytes (lower is better)."
echo "External codecs are lossless; ToroidZip rows are lossy at the stated precision."
echo ""

# ── build ToroidZip CLI ───────────────────────────────────────────────────────

echo "Building ToroidZip CLI..."
(cd "$REPO_ROOT" && go build -o "$REPO_ROOT/scripts/.tz_bin" ./cmd/toroidzip) || {
  echo "ERROR: failed to build toroidzip CLI" >&2; exit 1
}
TZ="$REPO_ROOT/scripts/.tz_bin"

# ── generate datasets ─────────────────────────────────────────────────────────

echo "Generating datasets (n=$N)..."
(cd "$REPO_ROOT" && go run ./scripts/gen_datasets --n "$N" --out "$BENCH_DIR") || {
  echo "ERROR: failed to generate datasets" >&2; exit 1
}
echo ""

# ── detect available external tools ──────────────────────────────────────────

HAS_ZSTD=0; HAS_BROTLI=0; HAS_FPZIP=0

require_tool zstd   "choco install zstd  |  brew install zstd  |  apt install zstd"    && HAS_ZSTD=1   || true
require_tool brotli "choco install brotli | brew install brotli | apt install brotli"  && HAS_BROTLI=1 || true
require_tool fpzip  "build from https://github.com/LLNL/fpzip (no package available)"  && HAS_FPZIP=1  || true

echo ""

# ── print table header ────────────────────────────────────────────────────────

DATASETS=( sensor financial multiscale volatile nearconstant neuralweight )

pad_r() { printf "%-${2}s" "$1"; }
pad_l() { printf "%${2}s" "$1"; }

print_header() {
  printf "| %-14s | %-14s | %-12s | %9s | %10s |\n" \
    "Dataset" "Codec" "Level/Mode" "Ratio" "Enc MB/s"
  printf "|%s|%s|%s|%s|%s|\n" \
    "---------------:" "---------------:" "-------------:" "----------:" "-----------:"
}

print_row() {
  local dataset="$1" codec="$2" level="$3" enc_bytes="$4" enc_ms="$5"
  local r mbs
  r=$(ratio "$enc_bytes")
  mbs=$(mb_per_sec "$enc_ms")
  printf "| %-14s | %-14s | %-12s | %9s | %10s |\n" \
    "$dataset" "$codec" "$level" "$r" "$mbs"
}

echo "### Results"
echo ""
print_header

# ── benchmark loop ────────────────────────────────────────────────────────────

for ds in "${DATASETS[@]}"; do
  f="$BENCH_DIR/${ds}.f64"

  # Uncompressed baseline (no codec, just file size)
  printf "| %-14s | %-14s | %-12s | %9s | %10s |\n" \
    "$ds" "uncompressed" "-" "1.0000" "-"

  # ToroidZip at each requested precision (Reanchor mode)
  for sf in "${PRECISIONS[@]}"; do
    tz_out="$BENCH_DIR/${ds}.q${sf}.tzrz"
    ms=$(time_compress "$TZ encode --entropy-mode quantized --sig-figs $sf $f $tz_out" "$tz_out")
    enc=$(file_bytes "$tz_out")
    print_row "$ds" "toroidzip" "Q-${sf}sf/Reanchor" "$enc" "$ms"
  done

  # zstd level 3 (default) and level 19 (max)
  if [[ $HAS_ZSTD -eq 1 ]]; then
    for lvl in 3 19; do
      zst_out="$BENCH_DIR/${ds}.zst${lvl}"
      ms=$(time_compress "zstd -q -$lvl --force $f -o $zst_out" "$zst_out")
      enc=$(file_bytes "$zst_out")
      print_row "$ds" "zstd" "level $lvl" "$enc" "$ms"
    done
  fi

  # brotli quality 6 (fast) and 11 (max)
  if [[ $HAS_BROTLI -eq 1 ]]; then
    for q in 6 11; do
      br_out="$BENCH_DIR/${ds}.br${q}"
      ms=$(time_compress "brotli -q $q --force -o $br_out $f" "$br_out")
      enc=$(file_bytes "$br_out")
      print_row "$ds" "brotli" "q=$q" "$enc" "$ms"
    done
  fi

  # fpzip (lossless float codec; fixed settings)
  if [[ $HAS_FPZIP -eq 1 ]]; then
    fpz_out="$BENCH_DIR/${ds}.fpz"
    # fpzip -t float -d 1 -n <count> -i <in> -o <out>
    ms=$(time_compress "fpzip -t double -d 1 -n $N -i $f -o $fpz_out" "$fpz_out")
    enc=$(file_bytes "$fpz_out")
    print_row "$ds" "fpzip" "lossless" "$enc" "$ms"
  fi
done

echo ""

# ── tier comparison summary ───────────────────────────────────────────────────

echo "### Tier comparison"
echo ""
echo "External codecs are lossless (no data loss). ToroidZip rows are lossy"
echo "at the stated precision. Compare within a tier, not across tiers."
echo ""

HAS_ANY_LOSSLESS=0
[[ $HAS_ZSTD -eq 1 || $HAS_BROTLI -eq 1 || $HAS_FPZIP -eq 1 ]] && HAS_ANY_LOSSLESS=1

# Lossless tier
if [[ $HAS_ANY_LOSSLESS -eq 1 ]]; then
  echo "#### Lossless tier (no data loss)"
  echo ""
  printf "| %-14s | %-14s | %-12s | %9s |\n" "Dataset" "Best codec" "Level/Mode" "Ratio"
  printf "|%s|%s|%s|%s|\n" "---------------:" "---------------:" "-------------:" "----------:"
  for ds in "${DATASETS[@]}"; do
    declare -A ll_sizes
    [[ $HAS_ZSTD   -eq 1 ]] && {
      ll_sizes["zstd:level 3"]=$(file_bytes "$BENCH_DIR/${ds}.zst3")
      ll_sizes["zstd:level 19"]=$(file_bytes "$BENCH_DIR/${ds}.zst19")
    }
    [[ $HAS_BROTLI -eq 1 ]] && {
      ll_sizes["brotli:q=6"]=$(file_bytes "$BENCH_DIR/${ds}.br6")
      ll_sizes["brotli:q=11"]=$(file_bytes "$BENCH_DIR/${ds}.br11")
    }
    [[ $HAS_FPZIP  -eq 1 ]] && { ll_sizes["fpzip:lossless"]=$(file_bytes "$BENCH_DIR/${ds}.fpz"); }
    best_codec=""; best_level=""; best_bytes=$UNCOMPRESSED_BYTES
    for key in "${!ll_sizes[@]}"; do
      b="${ll_sizes[$key]}"
      if [[ "$b" -lt "$best_bytes" ]]; then
        best_bytes="$b"; best_codec="${key%%:*}"; best_level="${key##*:}"
      fi
    done
    printf "| %-14s | %-14s | %-12s | %9s |\n" "$ds" "$best_codec" "$best_level" "$(ratio "$best_bytes")"
    unset ll_sizes
  done
  echo ""
fi

# ToroidZip quantized tiers — one section per tested precision
for sf in "${PRECISIONS[@]}"; do
  echo "#### Q-${sf}sf quantized tier (~${sf} significant figures, lossy)"
  echo ""
  if [[ $HAS_ANY_LOSSLESS -eq 1 ]]; then
    printf "| %-14s | %9s | %18s |\n" "Dataset" "TZ ratio" "vs best lossless"
    printf "|%s|%s|%s|\n" "---------------:" "----------:" "-------------------:"
  else
    printf "| %-14s | %9s |\n" "Dataset" "TZ ratio"
    printf "|%s|%s|\n" "---------------:" "----------:"
  fi
  for ds in "${DATASETS[@]}"; do
    tz_bytes=$(file_bytes "$BENCH_DIR/${ds}.q${sf}.tzrz")
    tz_r=$(ratio "$tz_bytes")
    if [[ $HAS_ANY_LOSSLESS -eq 1 ]]; then
      declare -A ll2
      [[ $HAS_ZSTD   -eq 1 ]] && {
        ll2["a"]=$(file_bytes "$BENCH_DIR/${ds}.zst3")
        ll2["b"]=$(file_bytes "$BENCH_DIR/${ds}.zst19")
      }
      [[ $HAS_BROTLI -eq 1 ]] && {
        ll2["c"]=$(file_bytes "$BENCH_DIR/${ds}.br6")
        ll2["d"]=$(file_bytes "$BENCH_DIR/${ds}.br11")
      }
      [[ $HAS_FPZIP  -eq 1 ]] && { ll2["e"]=$(file_bytes "$BENCH_DIR/${ds}.fpz"); }
      best_ll=$UNCOMPRESSED_BYTES
      for key in "${!ll2[@]}"; do
        b="${ll2[$key]}"; [[ "$b" -lt "$best_ll" ]] && best_ll="$b"
      done
      unset ll2
      vs=$(awk "BEGIN { printf \"%+.1f%%\", ($tz_bytes - $best_ll) / $best_ll * 100 }")
      printf "| %-14s | %9s | %18s |\n" "$ds" "$tz_r" "$vs"
    else
      printf "| %-14s | %9s |\n" "$ds" "$tz_r"
    fi
  done
  echo ""
done

echo ""

# ── cleanup ───────────────────────────────────────────────────────────────────

rm -f "$REPO_ROOT/scripts/.tz_bin"

if [[ $KEEP -eq 0 ]]; then
  rm -rf "$BENCH_DIR"
  echo "(bench_data/ removed; use --keep to retain)"
fi

echo ""
echo "Done."
