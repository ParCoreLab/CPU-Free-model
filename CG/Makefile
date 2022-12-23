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

NV_SRCDIR := src_nvshmem
NV_OBJDIR := obj_nvshmem
NV_DEPDIR := $(NV_OBJDIR)/.deps

GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

.DEFAULT_GOAL := all

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
NV_DEPFLAGS = -MT $@ -MMD -MP -MF $(NV_DEPDIR)/$*.d

NVCC_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(CXX) -I./include
NVCC_LDFLAGS = -ccbin=$(CXX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart

NVCC_NV_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include_nvshmem
NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5


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

NV_SRCS = $(call rwildcard,$(NV_SRCDIR),*.cu)
NV_OBJS := $(patsubst $(NV_SRCDIR)/%.cu, $(NV_OBJDIR)/%.o, $(NV_SRCS))
NV_DEPS := $(patsubst $(NV_SRCDIR)/%.cu, $(NV_DEPDIR)/%.d, $(NV_SRCS))

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

obj_nvshmem/mmio.c.o: src/mmio.c
	$(NVCC) -o $@ -c $<

obj_nvshmem/mmio_wrapper.o: src/mmio_wrapper.cpp
	$(NVCC) -o $@ -c $<

$(NV_OBJS) : $(NV_OBJDIR)/%.o : $(NV_SRCDIR)/%.cu $(NV_DEPDIR)/%.d | $(NV_DEPDIR)
	@mkdir -p "$(dir $(NV_DEPDIR)/$*)"
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_NV_FLAGS) $(NV_DEPFLAGS) -o $@ $<

cg_nvshmem: $(NV_OBJS) obj/mmio.c.o obj/mmio_wrapper.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(NVCC_NV_LDFLAGS)

all: cg cg_nvshmem

run: cg
	./cg

run_nvshmem: cg_nvshmem
	./cg_nvshmem

clean:
	$(RM) ./jacobi
	$(RM) -rd ./obj
	$(RM) ./jacobi_nvshmem
	$(RM) -rd ./obj_nvshmem

$(DEPDIR): 
	@mkdir -p $(DEPDIR)

$(NV_DEPDIR): 
	@mkdir -p $(NV_DEPDIR)

$(DEPS):

$(NV_DEPS):

include $(wildcard $(DEPS))
include $(wildcard $(NV_DEPS))
