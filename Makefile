# Build: make            (all problems)    make problem3-blocked   (one target)
# Target GPU: NVIDIA A40 (sm_86). For another GPU: make ARCH=sm_80  (etc.)
NVCC ?= nvcc
CC   ?= gcc
ARCH ?= sm_86

CFLAGS         = -O2 -std=c17 -D_POSIX_C_SOURCE=199309L
CUDAFLAGS      = -O2 -std=c++17 -arch=$(ARCH)
# Problem 2: --fmad=false so GPU arithmetic matches the CPU version bit for bit
CUDAFLAGS_P2   = -O2 -std=c++17 -arch=$(ARCH) --fmad=false
# v3 (Thrust): also keep the host-side index decoding free of FMA contraction
CUDAFLAGS_P2V3 = -O2 -std=c++14 -arch=$(ARCH) --fmad=false -Xcompiler -ffp-contract=off

P1 = problem1
P2 = problem2
P3 = problem3

all: problem1 problem2-v0 problem2-v1 problem2-v2 problem2-v3 problem3-vanilla problem3-blocked

problem1: $(P1)/230395-prob1.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(P1)/problem1.out

problem2-v0: $(P2)/230395-prob2-v0.c
	$(CC) $(CFLAGS) $< -o $(P2)/problem2-v0.out

problem2-v1: $(P2)/230395-prob2-v1.cu
	$(NVCC) $(CUDAFLAGS_P2) $< -o $(P2)/problem2-v1.out

problem2-v2: $(P2)/230395-prob2-v2.cu
	$(NVCC) $(CUDAFLAGS_P2) $< -o $(P2)/problem2-v2.out

problem2-v3: $(P2)/230395-prob2-v3.cu
	$(NVCC) $(CUDAFLAGS_P2V3) $< -o $(P2)/problem2-v3.out

problem3-vanilla: $(P3)/230395-prob3-vanilla.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(P3)/problem3-vanilla.out

problem3-blocked: $(P3)/230395-prob3-blocked.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(P3)/problem3-blocked.out

clean:
	rm -f $(P1)/*.out $(P2)/*.out $(P3)/*.out $(P2)/results-*.txt

.PHONY: all clean problem1 problem2-v0 problem2-v1 problem2-v2 problem2-v3 problem3-vanilla problem3-blocked
