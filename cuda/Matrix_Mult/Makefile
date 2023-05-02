# Compiler
NVCC = nvcc

# Compiler flags
#NVCCFLAGS = -arch=sm_30

# Source files
SRCS = matmul.cu matrix_cpu.cu

# Object files
OBJS = matmul.o matrix_cpu.o

# Header files
INCS = matrix_cpu.h

# Executable file name
EXE = MatMul.exe

all: $(EXE)

$(EXE): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(EXE)

matmul.o: matmul.cu $(INCS)
	$(NVCC) $(NVCCFLAGS) -c matmul.cu

matrix_cpu.o: matrix_cpu.cu $(INCS)
	$(NVCC) $(NVCCFLAGS) -c matrix_cpu.cu

clean:
	rm -f $(EXE) $(OBJS)
