NP ?= 1
NVCC ?= nvcc
MPIRUN ?= mpirun
MPICCX = mpic++

ifndef NVSHMEM_HOME
$(error NVSHMEM_HOME is not set)
endif
ifndef MPI_HOME
$(error MPI_HOME is not set)
endif
ifndef UCX_HOME
$(error UCX_HOME is not set)
endif

SRCDIR := src
OBJDIR := obj
DEPDIR := $(OBJDIR)/.deps

GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

NVCC_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include
NVCC_LDFLAGS = -ccbin=$(MPICCX) -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvToolsExt -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5

ifdef DISABLE_CUB
        NVCC_FLAGS += -Xptxas --optimize-float-atomics
else
        NVCC_FLAGS += -DHAVE_CUB
endif

ifdef PROFILE
		NVCC_FLAGS += -lineinfo --generate-line-info
endif

ifdef USE_NVTX
    	NVCC_FLAGS += -DUSE_NVTX -lnvToolsExt
endif

MAKEFLAGS += -j$(shell grep -c 'processor' /proc/cpuinfo)

rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

SRCS = $(call rwildcard,$(SRCDIR),*.cu)
OBJS := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(SRCS))
DEPS := $(patsubst $(SRCDIR)/%.cu, $(DEPDIR)/%.d, $(SRCS))

jacobi: $(OBJS)
	$(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(NVCC_LDFLAGS)

$(OBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cu $(DEPDIR)/%.d | $(DEPDIR)
	@mkdir -p "$(dir $(DEPDIR)/$*)"
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(DEPFLAGS) -o $@ $<

run: jacobi
	$(MPIRUN) -np $(NP) ./jacobi

clean:
	$(RM) ./jacobi
	$(RM) -rd ./obj

$(DEPS):

include $(wildcard $(DEPS))







#NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++17
#NVCC_FLAGS += -Xptxas="-v" -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(CXX)
#
#MAKEFLAGS += -j$(shell grep -c 'processor' /proc/cpuinfo)
#
#SRC=$(shell sh -c "find ./src/ -name '*.cu'")
#SRC= src/main.cu src/common.cu src/single-stream/multi-threaded-one-block-comm.cu
#OUT=$(SRC:=.o)
#OBJECTS := $(patsubst ./src/%.cu, obj/%.o, $(SRC))
#OBJECTS := $(addprefix obj/,$(SRC:.cu=.o))
#HED=$(shell sh -c find . -name '*.*h')

#jacobi: $(OBJECTS)
#	$(NVCC) $(NVCC_FLAGS) -I./include -o $@ $^

#$(OBJECTS): obj/%.o: %.cu
#	mkdir -p $(@D)
#	$(NVCC) -dc $(NVCC_FLAGS) -I./include -o $@ $<

#run: jacobi
#	./jacobi

#clean:
#	$(RM) ./jacobi
#	$(RM) -r ./obj
