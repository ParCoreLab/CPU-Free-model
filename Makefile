include common.mk

include Stencil/Makefile
include CG/Makefile

all: stencil cg

SOURCES := $(shell find . -type f -name '*.cu' -or -name '*.c' -or -name '*.cuh' -or -name '*.h' -or -name '*.cpp')

.PHONY format:
format: $(SOURCES)
	clang-format --style=file:.clang-format -i $^
