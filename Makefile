.PHONY: all clean distclean test-clean test

all: brandes

COMPILER_OPTIONS := -Wextra -Wall -O3
G++_COMPILER_OPTIONS := -Wall -Wextra -Wpedantic -O3
CUDA_COMPILER_OPTIONS := $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))

brandes:
	echo "brandes-par-vert-comp-virt-stride"

brandes-par-vert-comp-virt-stride: src/brandes-virt-stride.cu src/brandes-virt-stride.hpp src/sizes.hpp src/brandes-par-vert-comp-virt-stride.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-virt-stride.cu src/brandes-par-vert-comp-virt-stride.cu -arch=sm_61

brandes-par-vert-comp-virt: src/brandes-virt.cu src/brandes-virt.hpp src/sizes.hpp src/brandes-par-vert-comp-virt.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-virt.cu src/brandes-par-vert-comp-virt.cu -arch=sm_61

brandes-par-vert-queue-comp: src/brandes.cu src/brandes.hpp src/sizes.hpp src/brandes-par-vert-queue-comp.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes.cu src/brandes-par-vert-queue-comp.cu -arch=sm_61

brandes-par-edge-comp: src/brandes.cu src/brandes.hpp src/sizes.hpp src/brandes-par-edge-comp.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes.cu src/brandes-par-edge-comp.cu -arch=sm_61

brandes-par-vert-comp: src/brandes.cu src/brandes.hpp src/sizes.hpp src/brandes-par-vert-comp.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes.cu src/brandes-par-vert-comp.cu -arch=sm_61

brandes-par-vert-queue: src/brandes-old.cu src/brandes-old.hpp src/sizes.hpp src/brandes-par-vert-queue.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-old.cu src/brandes-par-vert-queue.cu -arch=sm_61

brandes-par-edge: src/brandes-old.cu src/brandes-old.hpp src/sizes.hpp src/brandes-par-edge.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-old.cu src/brandes-par-edge.cu -arch=sm_61

brandes-par-vert: src/brandes-old.cu src/brandes-old.hpp src/sizes.hpp src/brandes-par-vert.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-old.cu src/brandes-par-vert.cu -arch=sm_61

brandes-seq-array: src/brandes.cpp src/brandes-old.hpp src/compact_graph_representation.hpp
	g++ $(G++_COMPILER_OPTIONS) -o brandes src/brandes.cpp src/brandes-seq-array.cpp

brandes-seq-vector: src/brandes.cpp src/brandes-old.hpp src/compact_graph_representation.hpp
	g++ $(G++_COMPILER_OPTIONS) -o brandes src/brandes.cpp src/brandes-seq-vector.cpp

brandes-seq: src/brandes.cpp src/brandes-old.hpp src/compact_graph_representation.hpp
	g++ $(G++_COMPILER_OPTIONS) -o brandes src/brandes.cpp src/brandes-seq.cpp

clean:
	rm -f src/*.o

distclean: clean
	rm -f brandes

test-clean:
	rm -f res-*.txt errors.txt *.csv *.log

test:
	cd .. ; python3 verifyprograms.py
