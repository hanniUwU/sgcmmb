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

NVCCFLAGS       := -c -O3 -std=c++17 $(INCLUDES)
CFLAGS          := -c -O3 -std=c11 $(INCLUDES)
LDFLAGS         := -L$(CUDA_PATH)/lib64 -lcudart -lcublas \
                   -lcudart_static -lcutensor -lopenblas

TARGET          := matmul_bench
SRCS_C          := $(SRC_DIR)/matmul_bench.c
SRCS_CU         := $(SRC_DIR)/cutlass_gemm.cu
OBJS            := $(OBJ_DIR)/matmul_bench.o \
                   $(OBJ_DIR)/cutlass_gemm.o
BIN_TARGET      := $(BIN_DIR)/$(TARGET)

dirs: setup
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

setup:
	@echo "Building project in $(BIN_DIR)..."

all: dirs $(BIN_TARGET)

# link
$(BIN_TARGET): $(OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

# compile c
$(OBJ_DIR)/matmul_bench.o: $(SRCS_C)
	$(CC) $(CFLAGS) $< -o $@

# compile cu
$(OBJ_DIR)/cutlass_gemm.o: $(SRCS_CU) $(INC_DIR)/cutlass_gemm.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean dirs setup
