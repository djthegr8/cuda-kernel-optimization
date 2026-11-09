# CUDA Kernel Optimization (CS623 GPU, Assignment 2)

Three CUDA problems, each with a baseline and one or more optimized versions:

| # | Problem | Versions |
|---|---------|----------|
| 1 | 3D convolution (5×5×5 filter, N = 64 / 128 / 256) | basic global-memory kernel; tiled shared-memory kernel with constant-memory filter |
| 2 | 10-D grid search (13^10 ≈ 1.4×10^11 points, 11,608 hits) | v0 serial C; v1 one thread per point; v2 one thread per 7-variable prefix with pruning; v3 Thrust `copy_if` |
| 3 | Counting Bloom filter (insert / search / delete) | vanilla; blocked (extra hash picks a 32-byte block of counters) |

## Layout

```
230395_assign2.pdf   Report
Makefile
problem1/            Convolution
problem2/            Grid search (+ disp.txt, grid.txt inputs)
problem3/            Bloom filters (+ sample traces and generates_traces.py)
```

## Build

Needs CUDA (tested with 12.6) and gcc. The target architecture defaults to `sm_86` (A40 / RTX 30-series):

```sh
make                 # build everything
make problem3-blocked
make ARCH=sm_80      # other GPUs
```

Problem 2 is built with `--fmad=false` (and v3 with `-Xcompiler -ffp-contract=off`) so GPU results match the CPU version bit for bit.

## Run

Binaries land next to their sources as `*.out`.

```sh
# Problem 1: no arguments. The size N is the compile-time constants
# Input_Width / Input_Height in problem1/230395-prob1.cu (currently 1 << 8 = 256);
# edit them to 64 or 128 and rebuild to run the other sizes.
cd problem1 && ./problem1.out

# Problem 2: reads ./disp.txt and ./grid.txt, writes results-vN.txt in the working directory
cd ../problem2 && ./problem2-v2.out        # also problem2-v0.out / -v1.out / -v3.out

# Problem 3: N = number of counters, then absolute paths to the binary uint32 traces
cd ../problem3
./problem3-vanilla.out 20000000 $PWD/traces/insert.bin $PWD/traces/search.bin $PWD/traces/delete.bin
./problem3-blocked.out 20000000 $PWD/traces/insert.bin $PWD/traces/search.bin $PWD/traces/delete.bin
```

Sample traces are in `problem3/traces/`. To generate new ones (needs numpy): `python3 generates_traces.py --n-insert 1000000 --n-search 1000000 --n-delete 300000 --out traces`.

## Results summary

Development was done on an RTX 4060 laptop (Core i7-14650HX); the final results in the report were run on the GPU3 server (NVIDIA A40, Xeon Silver 4208, CUDA 12.6).

- **Problem 1:** the tiled kernel is 6–10× faster than the basic kernel (1.2–1.35× end-to-end, since PCIe transfers dominate), and about 45–57× faster end-to-end than the CPU.
- **Problem 2:** v0 takes ~657 s on the server CPU, v1 ~30 s, v2 ~0.14 s with pruning (the default; ~5.5 s when built with `-DPRUNE=0`), and v3 (Thrust) ~40 s. All versions find the same 11,608 points.
- **Problem 3:** no false negatives in any run. The blocked filter has ~25% faster searches but a higher false-positive rate (0.229% vs 0.119% after the inserts at N = 20M); the report includes the block-size ablation and Nsight Compute analysis.

See `230395_assign2.pdf` for the full write-up.
