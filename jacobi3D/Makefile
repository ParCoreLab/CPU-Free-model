NVCC ?= nvcc

GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

ifdef PROFILE
	NVCC_FLAGS = -lineinfo --generate-line-info
endif

ifdef USE_NVTX
        NVCC_FLAGS += -DUSE_NVTX -lnvToolsExt
endif

#NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++17
NVCC_FLAGS += -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(CXX)

MAKEFLAGS += -j$(shell grep -c 'processor' /proc/cpuinfo)

SRC=$(shell sh -c "find ./src/ -name '*.cu'")
#SRC= src/main.cu src/common.cu src/single-stream/multi-threaded-one-block-comm.cu
OUT=$(SRC:=.o)
#OBJECTS := $(patsubst ./src/%.cu, obj/%.o, $(SRC))
OBJECTS := $(addprefix obj/,$(SRC:.cu=.o))
#HED=$(shell sh -c find . -name '*.*h')

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
