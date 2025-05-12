SHELL := /bin/bash
CUDA_PATH       ?= /opt/cuda
CUTLASS_PATH    ?= $(CURDIR)/cutlass

SRC_DIR         := src
INC_DIR         := inc
OBJ_DIR         := obj
BIN_DIR         := bin

NVCC            := $(CUDA_PATH)/bin/nvcc
CC              := gcc

INCLUDES        := -I$(INC_DIR) \
                   -I$(CUTLASS_PATH)/include \
                   -I$(CUDA_PATH)/include

NVCCFLAGS       := -c -O3 -std=c++17 \
                   -Xcompiler -Wall -Xcompiler -Wextra \
                   $(INCLUDES)
CFLAGS          := -c -O3 -std=c11 -Wall -Wextra $(INCLUDES)
LDFLAGS         := -L$(CUDA_PATH)/lib64 -lcudart -lcublas \
                   -lcudart_static -lcutensor -lopenblas -lm

TARGET          := matmul_bench

SRCS_C          := $(SRC_DIR)/matmul_bench.c
SRCS_CU         := $(SRC_DIR)/cutlass_sgemm.cu \
                   $(SRC_DIR)/cuda_sgemm.cu

# auto-generate .o names from the source lists:
OBJ_C      := $(patsubst $(SRC_DIR)/%.c,    $(OBJ_DIR)/%.o, $(SRCS_C))
OBJ_CU     := $(patsubst $(SRC_DIR)/%.cu,   $(OBJ_DIR)/%.o, $(SRCS_CU))
OBJS       := $(OBJ_C) $(OBJ_CU)

BIN_TARGET := $(BIN_DIR)/$(TARGET)

.SHELLFLAGS := -e -o pipefail -c

all: dirs $(BIN_TARGET)

# link everything
$(BIN_TARGET): $(OBJS)
	$(NVCC) $^ -o $@ $(LDFLAGS)

# generic rule for .c → .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling C: $<"
	$(CC) $(CFLAGS) $< -o $@

# generic rule for .cu → .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) $< -o $@

dirs:
	@echo "Creating directories..."
	mkdir -p $(OBJ_DIR) $(BIN_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean dirs
