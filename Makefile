.PHONY: all clean distclean test-clean test

all: brandes

COMPILER_OPTIONS := -Wextra -Wall -O3
G++_COMPILER_OPTIONS := -Wall -Wextra -Wpedantic -O3
CUDA_COMPILER_OPTIONS := -O3 -gencode=arch=compute_70,code=sm_70 $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))

brandes: brandes-par-vert-comp-virt-stride

brandes-par-vert-comp-virt-stride: src/brandes-virt-stride.cu src/brandes-virt-stride.hpp src/sizes.hpp src/brandes-par-vert-comp-virt-stride.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-virt-stride.cu src/brandes-par-vert-comp-virt-stride.cu

brandes-par-vert-comp-virt-stride-const: src/brandes-virt-stride.cu src/brandes-virt-stride.hpp src/sizes.hpp src/brandes-par-vert-comp-virt-stride-const.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-virt-stride.cu src/brandes-par-vert-comp-virt-stride-const.cu

brandes-par-vert-virt-stride: src/brandes-virt-stride-nocomp.cu src/brandes-virt-stride-nocomp.hpp src/sizes.hpp src/brandes-par-vert-virt-stride.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-virt-stride-nocomp.cu src/brandes-par-vert-virt-stride.cu

brandes-par-vert-virt: src/brandes-virt-nocomp.cu src/brandes-virt-nocomp.hpp src/sizes.hpp src/brandes-par-vert-virt.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-virt-nocomp.cu src/brandes-par-vert-virt.cu

brandes-par-vert-comp-virt: src/brandes-virt.cu src/brandes-virt.hpp src/sizes.hpp src/brandes-par-vert-comp-virt.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-virt.cu src/brandes-par-vert-comp-virt.cu

brandes-par-vert-queue-comp: src/brandes.cu src/brandes.hpp src/sizes.hpp src/brandes-par-vert-queue-comp.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes.cu src/brandes-par-vert-queue-comp.cu

brandes-par-edge-comp: src/brandes.cu src/brandes.hpp src/sizes.hpp src/brandes-par-edge-comp.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes.cu src/brandes-par-edge-comp.cu

brandes-par-vert-comp: src/brandes.cu src/brandes.hpp src/sizes.hpp src/brandes-par-vert-comp.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes.cu src/brandes-par-vert-comp.cu

brandes-par-vert-queue: src/brandes-old.cu src/brandes-old.hpp src/sizes.hpp src/brandes-par-vert-queue.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-old.cu src/brandes-par-vert-queue.cu

brandes-par-edge: src/brandes-old.cu src/brandes-old.hpp src/sizes.hpp src/brandes-par-edge.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-old.cu src/brandes-par-edge.cu

brandes-par-vert: src/brandes-old.cu src/brandes-old.hpp src/sizes.hpp src/brandes-par-vert.cu src/compact_graph_representation.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes-old.cu src/brandes-par-vert.cu

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

readme:
	pandoc README.md -o README.pdf
