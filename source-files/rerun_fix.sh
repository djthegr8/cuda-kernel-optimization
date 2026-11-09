#!/usr/bin/env bash
# Standalone: redoes only the parts broken by the awk bug in the first run
#   1) p3/timing/summary.csv  (rebuilt from the saved raw logs)
#   2) ablation/p3_timing.csv (blocked variant, BLOCK_COUNTERS sweep)
#   3) n_sweep/*              (false-positive rate and timing vs. N)
# Run from source-files/, after the main run finished (needs results/, traces/, *.out).
set -u
cd "$(dirname "$(readlink -f "$0")")"

OUT=results
REPS=${REPS:-5}
ABL_REPS=${ABL_REPS:-3}
N=$(cat "$OUT/p3/N.txt" 2>/dev/null || echo 20000000)
T=$(readlink -f traces)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
[ -x "$CUDA_HOME/bin/nvcc" ] || CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
NVCC=$CUDA_HOME/bin/nvcc
ARCH=${ARCH:-sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}
echo "N=$N  arch=$ARCH  nvcc=$NVCC"

# ---------- 1) p3/timing/summary.csv from raw logs ----------
echo "variant,run,phase,keys,kernel_ms" > "$OUT/p3/timing/summary.csv"
for v in vanilla blocked; do
  for r in $(seq 1 "$REPS"); do
    [ -f "$OUT/p3/timing/$v.run$r.log" ] || continue
    awk -v v=$v -v r=$r '
      /^insert:/ {print v","r",insert,"$2","$6}
      /^search:/ {print v","r",search"(++s)","$2","$6}
      /^delete:/ {print v","r",delete,"$2","$6}' "$OUT/p3/timing/$v.run$r.log" >> "$OUT/p3/timing/summary.csv"
  done
done

# ---------- 2) P3 ablation: BLOCK_COUNTERS of the blocked variant ----------
mkdir -p "$OUT/ablation/bin"
echo "BLOCK_COUNTERS,run,phase,keys,kernel_ms" > "$OUT/ablation/p3_timing.csv"
for bc in 16 32 64 128 256; do
  exe="$OUT/ablation/bin/p3-blocked-bc$bc"
  "$NVCC" -O2 -std=c++17 -arch=$ARCH -DBLOCK_COUNTERS=${bc}u problem3-blocked.cu -o "$exe" || continue
  for r in $(seq 1 "$ABL_REPS"); do
    log=$("$exe" "$N" "$T/insert.bin" "$T/search.bin" "$T/delete.bin" 2>&1)
    echo "$log" | awk -v b="$bc" -v r="$r" '
      /^insert:/ {print b","r",insert,"$2","$6}
      /^search:/ {print b","r",search"(++s)","$2","$6}
      /^delete:/ {print b","r",delete,"$2","$6}' >> "$OUT/ablation/p3_timing.csv"
    [ "$r" = 1 ] && echo "$log" | grep '^\[' > "$OUT/ablation/p3_fp_bc$bc.txt"
  done
done
rm -rf "$OUT/ablation/bin"

# ---------- 3) N sweep ----------
mkdir -p "$OUT/n_sweep"
echo "variant,N,run,phase,keys,kernel_ms" > "$OUT/n_sweep/timing.csv"
echo "variant,N,phase,searches,false_negatives,false_positives,fp_rate_all,fp_rate_absent_only" > "$OUT/n_sweep/false_positives.csv"
for n in 1000000 5000000 10000000 20000000; do
  for v in vanilla blocked; do
    for r in $(seq 1 "$ABL_REPS"); do
      log=$(./problem3-$v.out "$n" "$T/insert.bin" "$T/search.bin" "$T/delete.bin" 2>&1)
      echo "$log" | awk -v v="$v" -v n="$n" -v r="$r" '
        /^insert:/ {print v","n","r",insert,"$2","$6}
        /^search:/ {print v","n","r",search"(++s)","$2","$6}
        /^delete:/ {print v","n","r",delete,"$2","$6}' >> "$OUT/n_sweep/timing.csv"
      if [ "$r" = 1 ]; then
        echo "$log" > "$OUT/n_sweep/${v}_N$n.log"
        echo "$log" | awk -v v="$v" -v n="$n" '
          /searches:/ {ph=$0; sub(/^\[/,"",ph); sub(/\].*/,"",ph); gsub(/,/,"");
                       s=$(NF-6); fn=$(NF-3); fp=$NF; cur=ph; S=s; FN=fn; FP=fp}
          /false positive rate/ {sub(/^\[[^]]*\] /,""); a=$0
                       match(a,/: [0-9.]+/); ra=substr(a,RSTART+2,RLENGTH-2)
                       rest=substr(a,RSTART+RLENGTH); match(rest,/: [0-9.]+/); rb=substr(rest,RSTART+2,RLENGTH-2)
                       print v","n","cur","S","FN","FP","ra","rb}' >> "$OUT/n_sweep/false_positives.csv"
      fi
    done
  done
done

# ---------- checks ----------
echo "---- line counts (expected 41, 61, 97, 17) ----"
for f in "$OUT/p3/timing/summary.csv" "$OUT/ablation/p3_timing.csv" \
         "$OUT/n_sweep/timing.csv" "$OUT/n_sweep/false_positives.csv"; do
  echo "$(wc -l < "$f") lines  $f"
done
echo "---- P2 correctness ----"
cat "$OUT/p2/correctness.csv"
date > "$OUT/DONE_fix"
echo "Finished. Zip the 'source-files/results' folder."
