NP ?= 1
NVCC ?= nvcc
MPIRUN ?= mpirun
MPICCX = mpic++
CXX ?= g++ 

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

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
NV_DEPFLAGS = -MT $@ -MMD -MP -MF $(NV_DEPDIR)/$*.d

NVCC_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(CXX)
NVCC_LDFLAGS = -ccbin=$(CXX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart

NVCC_NV_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include
NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5

MAKEFLAGS += -j

rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

SRCS = $(call rwildcard,$(SRCDIR),*.cu)
OBJS := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(SRCS))
DEPS := $(patsubst $(SRCDIR)/%.cu, $(DEPDIR)/%.d, $(SRCS))

NV_SRCS = $(call rwildcard,$(NV_SRCDIR),*.cu)
NV_OBJS := $(patsubst $(NV_SRCDIR)/%.cu, $(NV_OBJDIR)/%.o, $(NV_SRCS))
NV_DEPS := $(patsubst $(NV_SRCDIR)/%.cu, $(NV_DEPDIR)/%.d, $(NV_SRCS))

jacobi: $(OBJS)
	$(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(NVCC_LDFLAGS)

$(OBJS) : $(OBJDIR)/%.o : $(SRCDIR)/%.cu $(DEPDIR)/%.d | $(DEPDIR)
	@mkdir -p "$(dir $(DEPDIR)/$*)"
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(DEPFLAGS) -o $@ $<

jacobi_nvshmem: $(NV_OBJS)
	$(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(NVCC_NV_LDFLAGS)

$(NV_OBJS) : $(NV_OBJDIR)/%.o : $(NV_SRCDIR)/%.cu $(NV_DEPDIR)/%.d | $(NV_DEPDIR)
	@mkdir -p "$(dir $(NV_DEPDIR)/$*)"
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_NV_FLAGS) $(NV_DEPFLAGS) -o $@ $<

run_nvshmem: jacobi_nvshmem
	$(MPIRUN) -np $(NP) ./jacobi_nvshmem

run: jacobi
	./jacobi
clean:
	$(RM) ./jacobi
	$(RM) -rd ./obj
	$(RM) ./jacobi_nvshmem
	$(RM) -rd ./obj_nvshmem

$(DEPDIR): 
	@mkdir -p $(DEPDIR)

$(DEPS):

include $(wildcard $(DEPS))




# NVCC ?= nvcc

# GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
# GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
# GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

# #ifdef PROFILE
# #	NVCC_FLAGS = -lineinfo --generate-line-info
# #endif
# #
# #ifdef USE_NVTX
# #        NVCC_FLAGS += -DUSE_NVTX -lnvToolsExt
# #endif

# #NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++17
# NVCC_FLAGS += -Xcompiler -g -Xcompiler -fopenmp -std=c++17 -gencode="arch=compute_80,code=sm_80" -ccbin=g++-11 -D"__PRINT__" -D"GENWR"

# #MAKEFLAGS += -j$(shell grep -c 'processor' /proc/cpuinfo)

# SRC=$(shell sh -c "find ./src/ -name '*.cu'")
# OUT=$(SRC:=.o)
# #OBJECTS := $(patsubst ./src/%.cu, obj/%.o, $(SRC))
# OBJECTS := $(addprefix obj/,$(SRC:.cu=.o))

# all: jacobi

# jacobi: $(OBJECTS)
# 	$(NVCC) $(NVCC_FLAGS) -I./include -o $@ $^

# $(OBJECTS): obj/%.o: %.cu
# 	mkdir -p $(@D)
# 	$(NVCC) -dc $(NVCC_FLAGS) -I./include -o $@ $<

# run: jacobi
# 	./jacobi

# clean:
# 	$(RM) ./jacobi
# 	$(RM) -r ./obj



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
