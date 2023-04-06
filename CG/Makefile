SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)/../common.mk

BUILD_ROOT ?= bin
OBJ_ROOT ?= $(BUILD_ROOT)/obj

NV_SRCDIR := $(SELF_DIR)/src
NV_OBJDIR := $(BUILD_ROOT)
NV_DEPDIR := $(NV_OBJDIR)/.deps

.DEFAULT_GOAL := cg

ifdef PROFILE
	NVCC_NV_FLAGS += -lineinfo --generate-line-info
endif

ifdef USE_NVTX
	NVCC_NV_FLAGS += -DUSE_NVTX
	NVCC_NV_LDFLAGS += -lnvToolsExt
endif

NV_SRCS = $(call rwildcard,$(NV_SRCDIR),*.cu)
NV_OBJS := $(patsubst $(NV_SRCDIR)/%.cu, $(NV_OBJDIR)/%.o, $(NV_SRCS))
NV_DEPS := $(patsubst $(NV_SRCDIR)/%.cu, $(NV_DEPDIR)/%.d, $(NV_SRCS))

cg: $(NV_OBJS) $(NV_OBJDIR)/mmio.c.o $(NV_OBJDIR)/mmio_wrapper.o
	$(LINK_NVSHMEM)

$(NV_OBJS) : $(NV_OBJDIR)/%.o : $(NV_SRCDIR)/%.cu $(NV_DEPDIR)/%.d | $(NV_DEPDIR)
	$(call COMPILE_NVSHMEM, $(NV_DEPDIR))

$(NV_OBJDIR)/mmio.c.o: $(NV_SRCDIR)/mmio.c
	$(NVCC) -o $@ -c $<

$(NV_OBJDIR)/mmio_wrapper.o: $(NV_SRCDIR)/mmio_wrapper.cpp
	$(NVCC) -o $@ -c $<

run: cg
	./cg

$(NV_DEPDIR):
	@mkdir -p $(NV_DEPDIR)

$(NV_DEPS):

include $(wildcard $(NV_DEPS))
