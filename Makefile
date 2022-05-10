NVCC=nvcc
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)
ifdef DISABLE_CUB
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
else
        NVCC_FLAGS = -DHAVE_CUB
endif

#NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++17
NVCC_FLAGS += -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=`command -v ${CC}`

MAKEFLAGS += -j$(shell grep -c 'processor' /proc/cpuinfo)

SRC=$(shell sh -c "find ./src/ -name '*.cu'")
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
