.PHONY: all clean distclean testsclean

all: brandes

COMPILER_OPTIONS := -Wextra -Wall -O3
CUDA_COMPILER_OPTIONS := $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))

brandes: $(wildcard src/*.cu) $(wildcard src/*.h)
	nvcc $(CUDA_COMPILER_OPTIONS) -o brandes src/*.cu

clean:
	rm -f src/*.o

distclean: clean
	rm -f brandes

testsclean:
	rm -f res-*.txt errors.txt *.csv *.log

test:
	cd .. ; python3 verifyprograms.py
