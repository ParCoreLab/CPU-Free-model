NVCC ?= nvcc
MPIRUN ?= mpirun

ifndef NVSHMEM_HOME
$(error NVSHMEM_HOME is not set)
endif
ifndef MPI_HOME
$(error MPI_HOME is not set)
endif

GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

ifdef PROFILE
	NVCC_FLAGS = -lineinfo --generate-line-info
endif

#NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++17
NVCC_FLAGS += -I$(NVSHMEM_HOME)/include -L $(NVSHMEM_HOME)/lib -lnvshmem -lcuda -rdc=true -Xcompiler -fopenmp $(GENCODE_FLAGS) -ccbin=mpic++ -std=c++17

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
