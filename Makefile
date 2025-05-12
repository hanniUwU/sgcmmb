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
OBJS            := $(OBJ_DIR)/matmul_bench.o \
                   $(OBJ_DIR)/cutlass_sgemm.o \
		   $(OBJ_DIR)/cuda_sgemm.o
BIN_TARGET      := $(BIN_DIR)/$(TARGET)

.SHELLFLAGS := -e -o pipefail -c


all: dirs $(BIN_TARGET)

# linking
$(BIN_TARGET): $(OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

# compile .c
$(OBJ_DIR)/matmul_bench.o: $(SRCS_C)
	@echo "Compiling C: $<"
	$(CC) $(CFLAGS) $< -o $@

# compile .cu
$(OBJ_DIR)/cutlass_gemm.o: $(SRCS_CU) $(INC_DIR)/cutlass_gemm.h
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) $< -o $@

dirs:
	@echo "Creating directories..."
	mkdir -p $(OBJ_DIR) $(BIN_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean dirs
