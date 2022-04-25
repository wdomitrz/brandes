.PHONY: all clean

all: brandes

brandes: $(wildcard src/*.cu) $(wildcard src/*.h)
	nvcc -o brandes src/*.cu

clean:
	rm -f src/*.o
	rm -f brandes

distclean: clean
	rm -f brandes
