NVCC ?= nvcc

GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

#ifdef PROFILE
#	NVCC_FLAGS = -lineinfo --generate-line-info
#endif
#
#ifdef USE_NVTX
#        NVCC_FLAGS += -DUSE_NVTX -lnvToolsExt
#endif

#NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++17
NVCC_FLAGS += -Xcompiler -g -Xcompiler -fopenmp -std=c++17 -gencode="arch=compute_80,code=sm_80" -ccbin=g++-11 -D"__PRINT__" -D"GENWR"

#MAKEFLAGS += -j$(shell grep -c 'processor' /proc/cpuinfo)

SRC=$(shell sh -c "find ./src/ -name '*.cu'")
OUT=$(SRC:=.o)
#OBJECTS := $(patsubst ./src/%.cu, obj/%.o, $(SRC))
OBJECTS := $(addprefix obj/,$(SRC:.cu=.o))

all: jacobi

jacobi: $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -I./include -o $@ $^

$(OBJECTS): obj/%.o: %.cu
	mkdir -p $(@D)
	$(NVCC) -dc $(NVCC_FLAGS) -I./include -o $@ $<

run: jacobi
	./jacobi

clean:
	$(RM) ./jacobi
	$(RM) -r ./obj
