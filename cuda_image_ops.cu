#include "cuda_image_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "image.h"
#include "misc.h"


// Macro para chequeo rápido de errores CUDA
#define CUDA_CHECK(call) \
  { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
      fprintf(stderr, "CUDA error in %s (%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  }

// Kernel threshold: cada hilo procesa 1 píxel
__global__ void kernelThreshold(const unsigned char* src, unsigned char* dst, int width, int height, int threshold) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * width + x;
    dst[idx] = (src[idx] >= threshold) ? 255 : 0;
  }
}

void thresholdCUDA(const unsigned char* src, unsigned char* dst, int width, int height, int threshold) {
  size_t size = width * height * sizeof(unsigned char);
  unsigned char *d_src, *d_dst;

  CUDA_CHECK(cudaMalloc(&d_src, size));
  CUDA_CHECK(cudaMalloc(&d_dst, size));

  CUDA_CHECK(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x -1)/blockSize.x, (height + blockSize.y -1)/blockSize.y);
  kernelThreshold<<<gridSize, blockSize>>>(d_src, d_dst, width, height, threshold);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(dst, d_dst, size, cudaMemcpyDeviceToHost));

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Kernel RGB a GRAY: cada hilo procesa 1 píxel, src en formato RGB intercalado uchar3
__global__ void kernelConvertRGBtoGRAY(const unsigned char* src, unsigned char* dst, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * width + x;
    int rgb_idx = idx * 3;
    unsigned char r = src[rgb_idx];
    unsigned char g = src[rgb_idx + 1];
    unsigned char b = src[rgb_idx + 2];
    // Pesos estándar
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    dst[idx] = (unsigned char)(gray);
  }
}

void convertRGBtoGRAYCUDA(const unsigned char* src, unsigned char* dst, int width, int height) {
  size_t size_src = width * height * 3 * sizeof(unsigned char);
  size_t size_dst = width * height * sizeof(unsigned char);
  unsigned char *d_src, *d_dst;

  CUDA_CHECK(cudaMalloc(&d_src, size_src));
  CUDA_CHECK(cudaMalloc(&d_dst, size_dst));

  CUDA_CHECK(cudaMemcpy(d_src, src, size_src, cudaMemcpyHostToDevice));

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x -1)/blockSize.x, (height + blockSize.y -1)/blockSize.y);
  kernelConvertRGBtoGRAY<<<gridSize, blockSize>>>(d_src, d_dst, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(dst, d_dst, size_dst, cudaMemcpyDeviceToHost));

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Kernel convolución simple (1D mask horizontal, convolve_even)
__global__ void kernelConvolveEven(const float* src, float* dst, int width, int height, const float* mask, int mask_len) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    float sum = mask[0] * src[y * width + x];
    for (int i = 1; i < mask_len; i++) {
      int left = max(x - i, 0);
      int right = min(x + i, width - 1);
      sum += mask[i] * (src[y * width + left] + src[y * width + right]);
    }
    dst[y * width + x] = sum;
  }
}

void convolveEvenCUDA(const float* src, float* dst, int width, int height, const float* mask, int mask_len) {
  size_t size_img = width * height * sizeof(float);
  size_t size_mask = mask_len * sizeof(float);
  float *d_src, *d_dst, *d_mask;

  CUDA_CHECK(cudaMalloc(&d_src, size_img));
  CUDA_CHECK(cudaMalloc(&d_dst, size_img));
  CUDA_CHECK(cudaMalloc(&d_mask, size_mask));

  CUDA_CHECK(cudaMemcpy(d_src, src, size_img, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mask, mask, size_mask, cudaMemcpyHostToDevice));

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x -1)/blockSize.x, (height + blockSize.y -1)/blockSize.y);
  kernelConvolveEven<<<gridSize, blockSize>>>(d_src, d_dst, width, height, d_mask, mask_len);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(dst, d_dst, size_img, cudaMemcpyDeviceToHost));

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_mask);
}

// Kernel laplaciano 2D
__global__ void kernelLaplacian(const float* src, float* dst, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    float d2x = src[y * width + (x-1)] + src[y * width + (x+1)] - 2.f * src[y * width + x];
    float d2y = src[(y-1) * width + x] + src[(y+1) * width + x] - 2.f * src[y * width + x];
    dst[y * width + x] = d2x + d2y;
  }
}

void laplacianCUDA(const float* src, float* dst, int width, int height) {
  size_t size_img = width * height * sizeof(float);
  float *d_src, *d_dst;

  CUDA_CHECK(cudaMalloc(&d_src, size_img));
  CUDA_CHECK(cudaMalloc(&d_dst, size_img));

  CUDA_CHECK(cudaMemcpy(d_src, src, size_img, cudaMemcpyHostToDevice));

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x -1)/blockSize.x, (height + blockSize.y -1)/blockSize.y);
  kernelLaplacian<<<gridSize, blockSize>>>(d_src, d_dst, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(dst, d_dst, size_img, cudaMemcpyDeviceToHost));

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Kernel y función para colorear componentes en paralelo
__global__ void kernelColorComponents(rgb* output, int* components, rgb* colors, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * width + x;
    int comp = components[idx];
    output[idx] = colors[comp];
  }
}

void colorComponentsCUDA(rgb* output, int* components, rgb* colors, int width, int height) {
  dim3 blockSize(16,16);
  dim3 gridSize((width + blockSize.x -1)/blockSize.x, (height + blockSize.y -1)/blockSize.y);
  kernelColorComponents<<<gridSize, blockSize>>>(output, components, colors, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
