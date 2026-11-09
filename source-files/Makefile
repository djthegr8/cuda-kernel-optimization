CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-O2 -std=c17 -D_POSIX_C_SOURCE=199309L
CXXFLAGS=-O2 -std=c++17
CPPFLAGS=
CUDAFLAGS=-O2 -std=c++17 -arch=sm_86 
CUDAFLAGS_P2=-O2 -std=c++17 -arch=sm_86 --fmad=false
CUDAFLAGS_P23=-O2 -std=c++14 -arch=sm_86 --fmad=false -Xcompiler -ffp-contract=off
# First target is the default target
all: problem1 problem2-v0 problem2-v1 problem2-v2 problem2-v3 problem3-vanilla problem3-blocked

# $@ is a variable that contains the target name.
problem1: problem1.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem2-v0: problem2-v0.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $(patsubst %.c,%.out,$<)

problem2-v1: problem2-v1.cu
	$(NVCC) $(CUDAFLAGS_P2) $< -o $(patsubst %.cu,%.out,$<)

problem2-v2: problem2-v2.cu
	$(NVCC) $(CUDAFLAGS_P2) $< -o $(patsubst %.cu,%.out,$<)

problem2-v3: problem2-v3.cu
	$(NVCC) $(CUDAFLAGS_P23) $< -o $(patsubst %.cu,%.out,$<)

problem3-vanilla: problem3-vanilla.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-blocked: problem3-blocked.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

# The target is not a filename
clean:
	rm *.out compile_commands.json
