.PHONY: all clean distclean test-clean test

all: brandes

COMPILER_OPTIONS := -Wextra -Wall -O3
G++_COMPILER_OPTIONS := -Wall -Wextra -Wpedantic -O3
CUDA_COMPILER_OPTIONS := $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))

brandes:
	echo "asd"

brandes-par-vert-queue-comp: src/brandes.cu src/brandes.hpp src/sizes.hpp src/brandes-par-vert-queue-comp.cu
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes.cu src/brandes-par-vert-queue-comp.cu -arch=sm_61

brandes-par-edge-comp: src/brandes.cu src/brandes.hpp src/sizes.hpp src/brandes-par-edge-comp.cu
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes.cu src/brandes-par-edge-comp.cu -arch=sm_61

brandes-par-vert-comp: src/brandes.cu src/brandes.hpp src/sizes.hpp src/brandes-par-vert-comp.cu
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes.cu src/brandes-par-vert-comp.cu -arch=sm_61

brandes-par-vert-queue: src/brandes-old.cu src/brandes-old.hpp src/sizes.hpp src/brandes-par-vert-queue.cu
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-old.cu src/brandes-par-vert-queue.cu -arch=sm_61

brandes-par-edge: src/brandes-old.cu src/brandes-old.hpp src/sizes.hpp src/brandes-par-edge.cu
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-old.cu src/brandes-par-edge.cu -arch=sm_61

brandes-par-vert: src/brandes-old.cu src/brandes-old.hpp src/sizes.hpp src/brandes-par-vert.cu
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-old.cu src/brandes-par-vert.cu -arch=sm_61

brandes-seq-array: src/brandes.cpp src/brandes-old.hpp
	g++ $(G++_COMPILER_OPTIONS) -o brandes src/brandes.cpp src/brandes-seq-array.cpp

brandes-seq-vector: src/brandes.cpp src/brandes-old.hpp
	g++ $(G++_COMPILER_OPTIONS) -o brandes src/brandes.cpp src/brandes-seq-vector.cpp

brandes-seq: src/brandes.cpp src/brandes-old.hpp
	g++ $(G++_COMPILER_OPTIONS) -o brandes src/brandes.cpp src/brandes-seq.cpp

clean:
	rm -f src/*.o

distclean: clean
	rm -f brandes

test-clean:
	rm -f res-*.txt errors.txt *.csv *.log

test:
	cd .. ; python3 verifyprograms.py
