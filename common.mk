ifeq ($(_COMMON_),)
_COMMON_ := defined

NVCC ?= nvcc
MPIRUN ?= mpirun
MPICCX ?= mpic++
CXX ?= g++

BUILD_ROOT := bin
OBJ_ROOT := $(BUILD_ROOT)/obj

ifndef NVSHMEM_HOME
$(warning NVSHMEM_HOME is not set)
endif
ifndef MPI_HOME
$(warning MPI_HOME is not set)
endif
ifndef UCX_HOME
$(warning UCX_HOME is not set)
endif

MAKEFLAGS += -j

# Can't compile CUDA with -Wpedantic
WARN_FLAGS = "-Wall -Wno-comment -Werror -Wextra"

rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

GENCODE_SM70    := -gencode 'arch=compute_70,code=sm_70'
GENCODE_SM80    := -gencode 'arch=compute_80,code=sm_80' -gencode 'arch=compute_80,code=compute_80'
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

DEP_FLAGS = -MT $@ -MMD -MP -MF

NVCC_FLAGS_GENERIC = -O2 -dc -Xcompiler $(WARN_FLAGS) -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17

# Regular
NVCC_FLAGS = $(NVCC_FLAGS_GENERIC) -ccbin=$(CXX) -I./include
NVCC_LDFLAGS = -ccbin=$(CXX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart

# NVSHMEM
NVCC_NV_FLAGS = $(NVCC_FLAGS_GENERIC) -ccbin=$(MPICCX) -isystem $(NVSHMEM_HOME)/include -isystem $(MPI_HOME)/include -I./include_nvshmem
NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5

# Example
#$(OBJS_2D) : $(OBJ_DIR_2D)/%.o : $(SRC_DIR_2D)/%.cu $(DEP_DIR_2D)/%.d | $(DEP_DIR_2D)
#	$(call COMPILE, $(DEP_DIR_2D))

define LINK =
	@mkdir -p $(BUILD_ROOT)
	$(NVCC) $(GENCODE_FLAGS) -o $(BUILD_ROOT)/$@ $^ $(NVCC_LDFLAGS)
endef

define COMPILE =
	@mkdir -p "$(dir $(1)/$*)"
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(DEP_FLAGS) $(1)/$*.d -o $@ $<
endef

define LINK_NVSHMEM =
	@mkdir -p $(BUILD_ROOT)
	$(NVCC) $(GENCODE_FLAGS) -o $(BUILD_ROOT)/$@ $^ $(NVCC_NV_LDFLAGS)
endef

define COMPILE_NVSHMEM =
	@mkdir -p "$(dir $(1)/$*)"
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_NV_FLAGS) $(DEP_FLAGS) $(1)/$*.d -o $@ $<
endef

clean:
	$(RM) -rd $(BUILD_ROOT)

endif
