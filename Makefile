.PHONY: all clean

all: brandes

COMPILER_OPTIONS := -Wextra -Wall -O3
CUDA_COMPILER_OPTIONS := $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))

brandes: $(wildcard src/*.cu) $(wildcard src/*.h)
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/*.cu

clean:
	rm -f src/*.o
	rm -f brandes

distclean: clean
	rm -f brandes
