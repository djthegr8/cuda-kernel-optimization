#!/usr/bin/env bash
# Collects everything needed for the report (Problems 2 and 3) into ./results.
# Run from the directory that holds the sources, disp.txt and grid.txt:
#   ./run_benchmarks.sh
# Optional env vars: SKIP_BUILD=1 (reuse existing binaries), SKIP_P3=1 (skip P3 timing+ncu), SKIP_ABL_P2=1, SKIP_P2=1 (keep existing results/p2), REPS_CPU (runs of the serial v0, default 1), CUDA_HOME (default /usr/local/cuda-12.6), ABL_REPS (default 3), N (Bloom filter counters, default 20000000), REPS (default 5),
#                    ARCH (e.g. sm_86; default: auto-detected), SKIP_NCU=1
set -u
cd "$(dirname "$(readlink -f "$0")")"

N=${N:-20000000}
REPS=${REPS:-5}
OUT=results
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
[ -x "$CUDA_HOME/bin/nvcc" ] || CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
NVCC=$CUDA_HOME/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$OUT"/{p2,p3/timing,p3/ncu}

# ---------- build (architecture auto-detected) ----------
if [ -z "${ARCH:-}" ]; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  ARCH=sm_${cc}
fi
echo "Using -arch=$ARCH"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv > "$OUT/gpu.txt"
"$NVCC" --version > "$OUT/nvcc_version.txt"
if [ -z "${SKIP_BUILD:-}" ]; then
make -B all NVCC="$NVCC" \
  CUDAFLAGS="-O2 -std=c++17 -arch=$ARCH" \
  CUDAFLAGS_P2="-O2 -std=c++17 -arch=$ARCH --fmad=false" \
  CUDAFLAGS_P23="-O2 -std=c++14 -arch=$ARCH --fmad=false -Xcompiler -ffp-contract=off" \
  2>&1 | tee "$OUT/build.log"
fi

# ---------- Problem 2 ----------
# For every version: REPS runs, total time per run, and the number of points.
if [ -z "${SKIP_P2:-}" ]; then
echo "version,run,points,total_time_s" > "$OUT/p2/summary.csv"
for v in v0 v1 v2 v3; do
  reps=$REPS; [ "$v" = v0 ] && reps=${REPS_CPU:-1}  # serial CPU version: 1 run by default
  for r in $(seq 1 "$reps"); do
    ./problem2-$v.out > "$OUT/p2/$v.run$r.log" 2> "$OUT/p2/$v.run$r.err"
    pts=$(awk '/result pnts/ {print $3}' "$OUT/p2/$v.run$r.log")
    t=$(awk '/Total time/ {print $4}' "$OUT/p2/$v.run$r.log")
    echo "$v,$r,$pts,$t" >> "$OUT/p2/summary.csv"
    if [ "$r" = 1 ]; then  # output is deterministic in content; check it once
      sort "results-$v.txt" | md5sum | cut -d' ' -f1 > "$OUT/p2/$v.sorted_md5"
      wc -l < "results-$v.txt" > "$OUT/p2/$v.lines"
    fi
  done
done
{
  echo "version,lines,sorted_md5,matches_v0"
  for v in v0 v1 v2 v3; do
    m=no; [ "$(cat $OUT/p2/$v.sorted_md5)" = "$(cat $OUT/p2/v0.sorted_md5)" ] && m=yes
    echo "$v,$(cat $OUT/p2/$v.lines),$(cat $OUT/p2/$v.sorted_md5),$m"
  done
} > "$OUT/p2/correctness.csv"
fi

# ---------- Problem 3 ----------
# Traces: reuse ./traces if present, otherwise generate (1M inserts/searches, 300k deletes).
if [ ! -f traces/insert.bin ]; then
  python3 generates_traces.py --n-insert 1000000 --n-search 1000000 \
      --n-delete 300000 --out traces
fi
[ -d "$OUT/p3/traces" ] || cp -r traces "$OUT/p3/traces"
T=$(readlink -f traces)
echo "$N" > "$OUT/p3/N.txt"

if [ -z "${SKIP_P3:-}" ]; then
echo "variant,run,phase,keys,kernel_ms" > "$OUT/p3/timing/summary.csv"
for v in vanilla blocked; do
  for r in $(seq 1 "$REPS"); do
    log="$OUT/p3/timing/$v.run$r.log"
    ./problem3-$v.out "$N" "$T/insert.bin" "$T/search.bin" "$T/delete.bin" > "$log" 2>&1
    awk -v v="$v" -v r="$r" '
      /^insert:/ {print v","r",insert,"$2","$6}
      /^search:/ {print v","r",search"(++s)","$2","$6}
      /^delete:/ {print v","r",delete,"$2","$6}' "$log" >> "$OUT/p3/timing/summary.csv"
  done
  # false-positive lines (deterministic; taken from the first run)
  grep '^\[' "$OUT/p3/timing/$v.run1.log" > "$OUT/p3/timing/$v.false_positives.txt"
done

# Nsight Compute: first insert and first search launch only (delete is ignored).
if [ -z "${SKIP_NCU:-}" ] && command -v ncu >/dev/null; then
  for v in vanilla blocked; do
    ncu --set full \
        --metrics dram__bytes.sum,gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
        -k "regex:bloom_(insert|search)" -c 2 -f -o "$OUT/p3/ncu/$v" \
        ./problem3-$v.out "$N" "$T/insert.bin" "$T/search.bin" "$T/delete.bin" \
        > "$OUT/p3/ncu/$v.stdout" 2> "$OUT/p3/ncu/$v.stderr" \
      || { echo "ncu failed for $v (see $OUT/p3/ncu/$v.stderr)"; continue; }
    ncu -i "$OUT/p3/ncu/$v.ncu-rep" --page details --csv > "$OUT/p3/ncu/${v}_details.csv"
    ncu -i "$OUT/p3/ncu/$v.ncu-rep" --page raw     --csv > "$OUT/p3/ncu/${v}_raw.csv"
  done
else
  echo "Skipping ncu (not found or SKIP_NCU set)"
fi
fi

# ---------- Ablations (compile-time macros; ABL_REPS runs each) ----------
ABL_REPS=${ABL_REPS:-3}
mkdir -p "$OUT/ablation/bin"
F2="-O2 -std=c++17 -arch=$ARCH --fmad=false"
ref=$(cat "$OUT/p2/v0.sorted_md5")

# P2: launch block size for v1; block size x pruning for v2.
if [ -z "${SKIP_ABL_P2:-}" ]; then
echo "version,BLOCK,PRUNE,run,points,total_time_s,matches_v0" > "$OUT/ablation/p2.csv"
for v in v1 v2; do
  for b in 64 128 256 512 1024; do
    for pr in 0 1; do
      [ "$v" = v1 ] && [ "$pr" = 1 ] && continue
      exe="$OUT/ablation/bin/p2-$v-b$b-p$pr"
      "$NVCC" $F2 -DBLOCK=$b -DPRUNE=$pr problem2-$v.cu -o "$exe" || continue
      for r in $(seq 1 "$ABL_REPS"); do
        log=$("$exe" 2>/dev/null)
        pts=$(echo "$log" | awk '/result pnts/ {print $3}')
        t=$(echo "$log" | awk '/Total time/ {print $4}')
        m=-
        [ "$r" = 1 ] && { m=no; [ "$(sort results-$v.txt | md5sum | cut -d' ' -f1)" = "$ref" ] && m=yes; }
        echo "$v,$b,$pr,$r,$pts,$t,$m" >> "$OUT/ablation/p2.csv"
      done
    done
  done
done
fi

# P3: BLOCK_COUNTERS of the blocked variant.
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

# ---------- N sweep: how the false-positive rate changes with filter size ----------
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

date > "$OUT/DONE"
echo "Finished. Copy the '$OUT' folder back."
