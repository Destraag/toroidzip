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
#   --sig-figs N   test only this quantized precision (default: test 3, 6, and 9 sig figs)
#   --effort fast|best  lossless codec effort: fast=low level, best=max level (default: both)
#   --adaptive     also run adaptive at tolerances 1e-4 and 1e-6 (PrecisionBits=16)
#   --adaptive-v7  also run adaptive v7 at precision sf=4,5,6,9 with tol=1e-3 (fastPath)
#   --keep         keep the bench_data/ temp directory after the run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N=50000
KEEP=0
PRECISIONS=(3 6 9)
EFFORT=both   # fast | best | both
RUN_ADAPTIVE=0
ADAPTIVE_TOLS=(1e-6 1e-4)
ADAPTIVE_BITS=16
RUN_ADAPTIVE_V7=0
# A7 sig-fig levels: each sf uses bits=SigFigsToBits(sf) and tol=SigFigsToTolerance(sf).
# Hardcoded lookup (avoids needing a separate CLI call):
#   sf=4 → bits=13  sf=5 → bits=16  sf=6 → bits=20  sf=9 → bits=30
ADAPTIVE_V7_SF=(3 4 5 6 9)
declare -A ADAPTIVE_V7_BITS=([3]=10 [4]=13 [5]=16 [6]=20 [9]=30)
# All A7 modes use tol=1e308 (math.MaxFloat64-equivalent) so fastPath fires,
# meaning every ratio is quantized at the configured bit depth. This makes
# A7-Nsf directly comparable to Q-Nsf at the same precision level (A7-3sf ≡ Q-3sf).
declare -A ADAPTIVE_V7_TOL=([3]=1e308 [4]=1e308 [5]=1e308 [6]=1e308 [9]=1e308)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --sig-figs) PRECISIONS=("$2"); shift 2 ;;
    --effort) EFFORT="$2"; shift 2 ;;
    --adaptive) RUN_ADAPTIVE=1; shift ;;
    --adaptive-v7) RUN_ADAPTIVE_V7=1; shift ;;
    --keep) KEEP=1; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# Resolve lossless levels from --effort
case "$EFFORT" in
  fast) ZSTD_LEVELS=(3);  BROTLI_QUALS=(6)  ;;
  best) ZSTD_LEVELS=(19); BROTLI_QUALS=(11) ;;
  both) ZSTD_LEVELS=(3 19); BROTLI_QUALS=(6 11) ;;
  *) echo "--effort must be fast, best, or both (got '$EFFORT')" >&2; exit 1 ;;
esac

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
[[ $RUN_ADAPTIVE    -eq 1 ]] && echo "  + Adaptive (v6-compat) ε=${ADAPTIVE_TOLS[*]} ${ADAPTIVE_BITS}b (--adaptive)"
[[ $RUN_ADAPTIVE_V7 -eq 1 ]] && echo "  + Adaptive v7 sf=${ADAPTIVE_V7_SF[*]} precision-relative (--adaptive-v7)"
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

require_tool zstd   "brew install zstd | apt install zstd | Windows: download zstd.exe from https://github.com/facebook/zstd/releases and place on PATH"    && HAS_ZSTD=1   || true
require_tool brotli "choco install brotli | brew install brotli | apt install brotli"  && HAS_BROTLI=1 || true
require_tool fpzip  "no binary release; build from source: https://github.com/LLNL/fpzip (requires cmake + C++ compiler)"  && HAS_FPZIP=1  || true

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

  # ToroidZip quantized at each requested sig-figs level.
  # --sig-figs now implies adaptive mode, so we map sf→bits directly.
  sf_to_bits() {
    case "$1" in
      3) echo 10 ;; 4) echo 13 ;; 5) echo 16 ;;
      6) echo 20 ;; 7) echo 23 ;; 8) echo 26 ;; 9) echo 30 ;;
      *) echo 16 ;;
    esac
  }
  for sf in "${PRECISIONS[@]}"; do
    tz_out="$BENCH_DIR/${ds}.q${sf}.tzrz"
    bits=$(sf_to_bits "$sf")
    ms=$(time_compress "$TZ encode --entropy-mode quantized --precision $bits $f $tz_out" "$tz_out")
    enc=$(file_bytes "$tz_out")
    print_row "$ds" "toroidzip" "Q-${sf}sf/Reanchor" "$enc" "$ms"
  done

  # ToroidZip adaptive (v6-compat) — one row per tolerance
  if [[ $RUN_ADAPTIVE -eq 1 ]]; then
    for ADAPTIVE_TOL in "${ADAPTIVE_TOLS[@]}"; do
      tz_adap="$BENCH_DIR/${ds}.adaptive${ADAPTIVE_TOL}.tzrz"
      ms=$(time_compress "$TZ encode --entropy-mode adaptive --tolerance ${ADAPTIVE_TOL} --precision ${ADAPTIVE_BITS} $f $tz_adap" "$tz_adap")
      enc=$(file_bytes "$tz_adap")
      print_row "$ds" "toroidzip" "Adap-ε${ADAPTIVE_TOL}" "$enc" "$ms"
    done
  fi

  # ToroidZip adaptive v7 — one row per sig-figs level
  if [[ $RUN_ADAPTIVE_V7 -eq 1 ]]; then
    for sf in "${ADAPTIVE_V7_SF[@]}"; do
      tz_a7="$BENCH_DIR/${ds}.a7sf${sf}.tzrz"
      b=${ADAPTIVE_V7_BITS[$sf]}
      tol=${ADAPTIVE_V7_TOL[$sf]}
      ms=$(time_compress "$TZ encode --entropy-mode adaptive --precision $b --tolerance $tol $f $tz_a7" "$tz_a7")
      enc=$(file_bytes "$tz_a7")
      print_row "$ds" "toroidzip" "A7-${sf}sf(B=${b})" "$enc" "$ms"
    done
  fi

  # zstd
  if [[ $HAS_ZSTD -eq 1 ]]; then
    for lvl in "${ZSTD_LEVELS[@]}"; do
      zst_out="$BENCH_DIR/${ds}.zst${lvl}"
      ms=$(time_compress "zstd -q -$lvl --force $f -o $zst_out" "$zst_out")
      enc=$(file_bytes "$zst_out")
      print_row "$ds" "zstd" "level $lvl" "$enc" "$ms"
    done
  fi

  # brotli
  if [[ $HAS_BROTLI -eq 1 ]]; then
    for q in "${BROTLI_QUALS[@]}"; do
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
    [[ $HAS_ZSTD   -eq 1 ]] && for lvl in "${ZSTD_LEVELS[@]}"; do
      ll_sizes["zstd:level $lvl"]=$(file_bytes "$BENCH_DIR/${ds}.zst${lvl}")
    done
    [[ $HAS_BROTLI -eq 1 ]] && for q in "${BROTLI_QUALS[@]}"; do
      ll_sizes["brotli:q=$q"]=$(file_bytes "$BENCH_DIR/${ds}.br${q}")
    done
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
      [[ $HAS_ZSTD   -eq 1 ]] && for lvl in "${ZSTD_LEVELS[@]}"; do
        ll2["zstd${lvl}"]=$(file_bytes "$BENCH_DIR/${ds}.zst${lvl}")
      done
      [[ $HAS_BROTLI -eq 1 ]] && for q in "${BROTLI_QUALS[@]}"; do
        ll2["br${q}"]=$(file_bytes "$BENCH_DIR/${ds}.br${q}")
      done
      [[ $HAS_FPZIP  -eq 1 ]] && { ll2["fpzip"]=$(file_bytes "$BENCH_DIR/${ds}.fpz"); }
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

# ── overall ranking ───────────────────────────────────────────────────────────
# Rank every (codec, level) pair by average ratio across all datasets.
# TZ rows are labeled lossy; lossless rows labeled lossless.

echo "### Overall ranking (average ratio across all datasets, lower is better)"
echo ""
echo "Note: TZ rows are lossy; lossless rows preserve all data. Not a fair"
echo "head-to-head, but useful to see where each setting lands on the ratio axis."
echo ""
printf "| %4s | %-18s | %-12s | %-8s | %12s |\n" "Rank" "Codec" "Level/Mode" "Type" "Avg ratio"
printf "|%s|%s|%s|%s|%s|\n" "-----:" "-------------------:" "-------------:" "---------:" "-------------:"

# Collect all (tag, label, type, total_bytes) entries
declare -A rank_total rank_label rank_type

for ds in "${DATASETS[@]}"; do
  for sf in "${PRECISIONS[@]}"; do
    tag="tz_q${sf}"
    rank_label["$tag"]="toroidzip"
    rank_type["$tag"]="lossy"
    rank_total["$tag"]=$(( ${rank_total["$tag"]:-0} + $(file_bytes "$BENCH_DIR/${ds}.q${sf}.tzrz") ))
  done
  if [[ $RUN_ADAPTIVE -eq 1 ]]; then
    for ADAPTIVE_TOL in "${ADAPTIVE_TOLS[@]}"; do
      tag="tz_adaptive_${ADAPTIVE_TOL}"
      rank_label["$tag"]="toroidzip"
      rank_type["$tag"]="lossy"
      rank_total["$tag"]=$(( ${rank_total["$tag"]:-0} + $(file_bytes "$BENCH_DIR/${ds}.adaptive${ADAPTIVE_TOL}.tzrz") ))
    done
  fi
  if [[ $RUN_ADAPTIVE_V7 -eq 1 ]]; then
    for sf in "${ADAPTIVE_V7_SF[@]}"; do
      tag="tz_a7_sf${sf}"
      rank_label["$tag"]="toroidzip"
      rank_type["$tag"]="lossy"
      rank_total["$tag"]=$(( ${rank_total["$tag"]:-0} + $(file_bytes "$BENCH_DIR/${ds}.a7sf${sf}.tzrz") ))
    done
  fi
  [[ $HAS_ZSTD -eq 1 ]] && for lvl in "${ZSTD_LEVELS[@]}"; do
    tag="zstd_${lvl}"
    rank_label["$tag"]="zstd"
    rank_type["$tag"]="lossless"
    rank_total["$tag"]=$(( ${rank_total["$tag"]:-0} + $(file_bytes "$BENCH_DIR/${ds}.zst${lvl}") ))
  done
  [[ $HAS_BROTLI -eq 1 ]] && for q in "${BROTLI_QUALS[@]}"; do
    tag="brotli_${q}"
    rank_label["$tag"]="brotli"
    rank_type["$tag"]="lossless"
    rank_total["$tag"]=$(( ${rank_total["$tag"]:-0} + $(file_bytes "$BENCH_DIR/${ds}.br${q}") ))
  done
  [[ $HAS_FPZIP -eq 1 ]] && {
    rank_label["fpzip"]="fpzip"
    rank_type["fpzip"]="lossless"
    rank_total["fpzip"]=$(( ${rank_total["fpzip"]:-0} + $(file_bytes "$BENCH_DIR/${ds}.fpz") ))
  }
done

TOTAL_UNCOMPRESSED=$(( ${#DATASETS[@]} * UNCOMPRESSED_BYTES ))

# Sort by total bytes, print ranked
rank=1
while IFS= read -r line; do
  total="${line%%:*}"
  tag="${line##*:}"
  codec="${rank_label[$tag]}"
  level_tag="${tag#*_}"
  case "$tag" in
    tz_q*) level="Q-${level_tag#q}sf/Reanchor" ;;
    tz_adaptive_*) level="Adap-ε${tag#tz_adaptive_}/${ADAPTIVE_BITS}b" ;;
    tz_a7_sf*) level="A7-${tag#tz_a7_sf}sf" ;;
    zstd_*)     level="level ${level_tag}" ;;
    brotli_*)   level="q=${level_tag}" ;;
    fpzip)      level="lossless" ;;
    *)          level="$level_tag" ;;
  esac
  typ="${rank_type[$tag]}"
  avg=$(awk "BEGIN { printf \"%.4f\", $total / $TOTAL_UNCOMPRESSED }")
  printf "| %4d | %-18s | %-12s | %-8s | %12s |\n" "$rank" "$codec" "$level" "$typ" "$avg"
  (( rank++ ))
done < <(
  for tag in "${!rank_total[@]}"; do
    echo "${rank_total[$tag]}:${tag}"
  done | sort -t: -k1 -n
)
unset rank_total rank_label rank_type

echo ""

# ── cleanup ───────────────────────────────────────────────────────────────────

rm -f "$REPO_ROOT/scripts/.tz_bin"

if [[ $KEEP -eq 0 ]]; then
  rm -rf "$BENCH_DIR"
  echo "(bench_data/ removed; use --keep to retain)"
fi

echo ""
echo "Done."
