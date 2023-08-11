SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)/../common.mk

BUILD_ROOT ?= bin
OBJ_ROOT ?= $(BUILD_ROOT)/obj

VERSION_2D = $(SELF_DIR)/jacobi2D
VERSION_3D = $(SELF_DIR)/jacobi3D

2D_PERKS_WRAPPER = $(VERSION_2D)/PERKS/*.cu

# =========================================================================
# 2D
SRC_DIR_2D = $(VERSION_2D)/src/
OBJ_DIR_2D := $(OBJ_ROOT)/$(VERSION_2D)/
DEP_DIR_2D := $(OBJ_DIR_2D)/.deps/

SRCS_2D = $(call rwildcard,$(SRC_DIR_2D),*.cu)
OBJS_2D := $(patsubst $(SRC_DIR_2D)/%.cu, $(OBJ_DIR_2D)/%.o, $(SRCS_2D))
DEPS_2D := $(patsubst $(SRC_DIR_2D)/%.cu, $(DEP_DIR_2D)/%.d, $(SRCS_2D))
# =========================================================================

# =========================================================================
# 3D
SRC_DIR_3D = $(VERSION_3D)/src/
OBJ_DIR_3D := $(OBJ_ROOT)/$(VERSION_3D)/
DEP_DIR_3D := $(OBJ_DIR_3D)/.deps/

SRCS_3D = $(call rwildcard,$(SRC_DIR_3D),*.cu)
OBJS_3D := $(patsubst $(SRC_DIR_3D)/%.cu, $(OBJ_DIR_3D)/%.o, $(SRCS_3D))
DEPS_3D := $(patsubst $(SRC_DIR_3D)/%.cu, $(DEP_DIR_3D)/%.d, $(SRCS_3D))
# =========================================================================

# =========================================================================
# 2D NVSHMEM
SRC_DIR_2D_NVSHMEM = $(VERSION_2D)/src_nvshmem/
OBJ_DIR_2D_NVSHMEM := $(OBJ_ROOT)/$(VERSION_2D)_nvshmem/
DEP_DIR_2D_NVSHMEM := $(OBJ_DIR_2D_NVSHMEM)/.deps/

SRCS_2D_NVSHMEM = $(call rwildcard,$(SRC_DIR_2D_NVSHMEM),*.cu)
OBJS_2D_NVSHMEM := $(patsubst $(SRC_DIR_2D_NVSHMEM)/%.cu, $(OBJ_DIR_2D_NVSHMEM)/%.o, $(SRCS_2D_NVSHMEM))
DEPS_2D_NVSHMEM := $(patsubst $(SRC_DIR_2D_NVSHMEM)/%.cu, $(DEP_DIR_2D_NVSHMEM)/%.d, $(SRCS_2D_NVSHMEM))
# =========================================================================

# =========================================================================
# 3D NVSHMEM
SRC_DIR_3D_NVSHMEM = $(VERSION_3D)/src_nvshmem/
OBJ_DIR_3D_NVSHMEM := $(OBJ_ROOT)/$(VERSION_3D)_nvshmem/
DEP_DIR_3D_NVSHMEM := $(OBJ_DIR_3D_NVSHMEM)/.deps/

SRCS_3D_NVSHMEM = $(call rwildcard,$(SRC_DIR_3D_NVSHMEM),*.cu)
OBJS_3D_NVSHMEM := $(patsubst $(SRC_DIR_3D_NVSHMEM)/%.cu, $(OBJ_DIR_3D_NVSHMEM)/%.o, $(SRCS_3D_NVSHMEM))
DEPS_3D_NVSHMEM := $(patsubst $(SRC_DIR_3D_NVSHMEM)/%.cu, $(DEP_DIR_3D_NVSHMEM)/%.d, $(SRCS_3D_NVSHMEM))
# =========================================================================
stencil: jacobi jacobi_nvshmem
jacobi: jacobi2D jacobi3D
jacobi_nvshmem: jacobi2D_nvshmem jacobi3D_nvshmem

# =========================================================================
jacobi2D: $(OBJS_2D) $(2D_PERKS_WRAPPER)
	$(LINK)

jacobi3D: $(OBJS_3D)
	$(LINK)

$(OBJS_2D) : $(OBJ_DIR_2D)/%.o : $(SRC_DIR_2D)/%.cu $(DEP_DIR_2D)/%.d | $(DEP_DIR_2D)
	$(call COMPILE, $(DEP_DIR_2D))

$(OBJS_3D) : $(OBJ_DIR_3D)/%.o : $(SRC_DIR_3D)/%.cu $(DEP_DIR_3D)/%.d | $(DEP_DIR_3D)
	$(call COMPILE, $(DEP_DIR_3D))
# =========================================================================

# =========================================================================
jacobi2D_nvshmem: $(OBJS_2D_NVSHMEM) $(2D_PERKS_WRAPPER)
	$(LINK_NVSHMEM)

jacobi3D_nvshmem: $(OBJS_3D_NVSHMEM)
	$(LINK_NVSHMEM)

$(OBJS_2D_NVSHMEM) : $(OBJ_DIR_2D_NVSHMEM)/%.o : $(SRC_DIR_2D_NVSHMEM)/%.cu $(DEP_DIR_2D_NVSHMEM)/%.d | $(DEP_DIR_2D_NVSHMEM)
	$(call COMPILE_NVSHMEM, $(DEP_DIR_2D_NVSHMEM))

$(OBJS_3D_NVSHMEM) : $(OBJ_DIR_3D_NVSHMEM)/%.o : $(SRC_DIR_3D_NVSHMEM)/%.cu $(DEP_DIR_3D_NVSHMEM)/%.d | $(DEP_DIR_3D_NVSHMEM)
	$(call COMPILE_NVSHMEM, $(DEP_DIR_3D_NVSHMEM))
# =========================================================================

$(DEP_DIR_2D):
	@mkdir -p $(DEP_DIR_2D)

$(DEP_DIR_3D):
	@mkdir -p $(DEP_DIR_3D)

$(DEP_DIR_2D_NVSHMEM):
	@mkdir -p $(DEP_DIR_2D_NVSHMEM)

$(DEP_DIR_3D_NVSHMEM):
	@mkdir -p $(DEP_DIR_3D_NVSHMEM)

$(DEPS_2D):

$(DEPS_3D):

$(DEPS_2D_NVSHMEM):

$(DEPS_3D_NVSHMEM):

include $(wildcard $(DEPS_2D))
include $(wildcard $(DEPS_3D))
include $(wildcard $(DEPS_2D_NVSHMEM))
include $(wildcard $(DEPS_3D_NVSHMEM))
