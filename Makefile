.PHONY: all clean distclean test-clean test

all: brandes

COMPILER_OPTIONS := -Wextra -Wall -O3
G++_COMPILER_OPTIONS := -Wall -Wextra -Wpedantic -O3
CUDA_COMPILER_OPTIONS := $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))

brandes: src/brandes.cu src/brandes.hpp
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/brandes.cu

brandes-seq: src/brandes.cpp src/brandes.hpp
	g++ $(G++_COMPILER_OPTIONS) -o brandes src/brandes.cpp src/brandes-seq.cpp

brandes-seq-vector: src/brandes.cpp src/brandes.hpp
	g++ $(G++_COMPILER_OPTIONS) -o brandes src/brandes.cpp src/brandes-seq-vector.cpp

brandes-seq-array: src/brandes.cpp src/brandes.hpp
	g++ $(G++_COMPILER_OPTIONS) -o brandes src/brandes.cpp src/brandes-seq-array.cpp

clean:
	rm -f src/*.o

distclean: clean
	rm -f brandes

test-clean:
	rm -f res-*.txt errors.txt *.csv *.log

test:
	cd .. ; python3 verifyprograms.py
