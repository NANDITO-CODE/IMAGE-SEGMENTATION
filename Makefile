# Compiladores
NVCC = nvcc
CXX = g++

# Ruta CUDA en Colab típica
CUDA_PATH = /usr/local/cuda

# Flags para CUDA (ajusta sm_75 si tu GPU es diferente)
NVCC_FLAGS = -arch=sm_75

# Flags para C++ (incluye el directorio actual y CUDA includes)
CXXFLAGS = -O2 -std=c++11 -I. -I$(CUDA_PATH)/include

# Flags para linkear librerías CUDA
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

# Archivos fuente
CU_SRCS = cuda_image_ops.cu
CPP_SRCS = segment.cpp

# Archivos objeto
CU_OBJS = $(CU_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)

# Ejecutable final
TARGET = segment

# Regla para compilar archivos .cu
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Regla para compilar archivos .cpp
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla para linkear todo
$(TARGET): $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) -o $@

# Limpieza de objetos y ejecutable
clean:
	rm -f $(CU_OBJS) $(CPP_OBJS) $(TARGET)
