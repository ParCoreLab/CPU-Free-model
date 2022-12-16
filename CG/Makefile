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

GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

.DEFAULT_GOAL := cg

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

NVCC_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include
NVCC_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5

ifdef PROFILE
	NVCC_FLAGS = -lineinfo --generate-line-info
endif

ifdef USE_NVTX
        NVCC_FLAGS += -DUSE_NVTX -lnvToolsExt
endif

MAKEFLAGS += -j

rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

SRCS = $(call rwildcard,$(SRCDIR),*.cu)
OBJS := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(SRCS))
DEPS := $(patsubst $(SRCDIR)/%.cu, $(DEPDIR)/%.d, $(SRCS))

obj/mmio.c.o: src/mmio.c
	$(NVCC) -o $@ -c $<

obj/mmio_wrapper.o: src/mmio_wrapper.cpp
	$(NVCC) -o $@ -c $<

$(OBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cu $(DEPDIR)/%.d | $(DEPDIR)
	@mkdir -p "$(dir $(DEPDIR)/$*)"
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(DEPFLAGS) -o $@ $<

cg: $(OBJS) obj/mmio.c.o obj/mmio_wrapper.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(NVCC_LDFLAGS)

run: cg
	./cg

clean:
	$(RM) ./cg
	$(RM) -r ./obj

$(DEPDIR): 
	@mkdir -p $(DEPDIR)

$(DEPS):

include $(wildcard $(DEPS))
