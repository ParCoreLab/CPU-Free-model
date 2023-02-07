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

NV_SRCDIR := src
NV_OBJDIR := obj
NV_DEPDIR := $(NV_OBJDIR)/.deps

GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

.DEFAULT_GOAL := cg

NV_DEPFLAGS = -MT $@ -MMD -MP -MF $(NV_DEPDIR)/$*.d

NVCC_NV_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include
NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5

ifdef PROFILE
	NVCC_NV_FLAGS += -lineinfo --generate-line-info
endif

ifdef USE_NVTX
        NVCC_NV_FLAGS += -DUSE_NVTX
		NVCC_NV_LDFLAGS += -lnvToolsExt
endif

MAKEFLAGS += -j

rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

NV_SRCS = $(call rwildcard,$(NV_SRCDIR),*.cu)
NV_OBJS := $(patsubst $(NV_SRCDIR)/%.cu, $(NV_OBJDIR)/%.o, $(NV_SRCS))
NV_DEPS := $(patsubst $(NV_SRCDIR)/%.cu, $(NV_DEPDIR)/%.d, $(NV_SRCS))

obj/mmio.c.o: src/mmio.c
	$(NVCC) -o $@ -c $<

obj/mmio_wrapper.o: src/mmio_wrapper.cpp
	$(NVCC) -o $@ -c $<

$(NV_OBJS) : $(NV_OBJDIR)/%.o : $(NV_SRCDIR)/%.cu $(NV_DEPDIR)/%.d | $(NV_DEPDIR)
	@mkdir -p "$(dir $(NV_DEPDIR)/$*)"
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_NV_FLAGS) $(NV_DEPFLAGS) -o $@ $<

cg: $(NV_OBJS) obj/mmio.c.o obj/mmio_wrapper.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(NVCC_NV_LDFLAGS)

run: cg
	./cg

clean:
	$(RM) ./cg
	$(RM) -rd ./obj

$(NV_DEPDIR): 
	@mkdir -p $(NV_DEPDIR)

$(NV_DEPS):

include $(wildcard $(NV_DEPS))
