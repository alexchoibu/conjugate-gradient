# Compilers
CC := gcc
NVCC := nvcc

# Directories
SRC_DIR := src
EXE_DIR := exe

# Flags
CFLAGS := -O1 -Wall
NVCCFLAGS := -arch compute_60 -code sm_60 -O1 -Xcompiler -Wall -lm

# Find all source files in src/
C_SRCS := $(wildcard $(SRC_DIR)/*.c)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)

# Generate list of executables in exe/
C_EXES := $(patsubst $(SRC_DIR)/%.c,$(EXE_DIR)/%,$(C_SRCS))
CU_EXES := $(patsubst $(SRC_DIR)/%.cu,$(EXE_DIR)/%,$(CU_SRCS))

# Default target: build all
all: $(EXE_DIR) $(C_EXES) $(CU_EXES)

# Ensure the exe directory exists
$(EXE_DIR):
	mkdir -p $(EXE_DIR)

# Rule for .c files
$(EXE_DIR)/%: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@

# Rule for .cu files
$(EXE_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Clean up
clean:
	rm -rf $(EXE_DIR)