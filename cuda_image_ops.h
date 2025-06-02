#ifndef CUDA_IMAGE_OPS_H
#define CUDA_IMAGE_OPS_H

#include <cuda_runtime.h>
#include "misc.h"  // para el tipo rgb

#ifdef __cplusplus
extern "C" {
#endif

// Aplica threshold sobre imagen grayscale (uchar)
void thresholdCUDA(const unsigned char* src, unsigned char* dst, int width, int height, int threshold);

// Convierte imagen RGB a GRAY (uchar)
void convertRGBtoGRAYCUDA(const unsigned char* src, unsigned char* dst, int width, int height);

// Convolución básica (convolve_even)
void convolveEvenCUDA(const float* src, float* dst, int width, int height, const float* mask, int mask_len);

// Laplaciano
void laplacianCUDA(const float* src, float* dst, int width, int height);

// Reducción min/max
void minMaxCUDA(const float* src, int width, int height, float* min_val, float* max_val);

// Colorea componentes paralelamente (función host que lanza kernel)
void colorComponentsCUDA(rgb* output, int* components, rgb* colors, int width, int height);

#ifdef __cplusplus
}
#endif

#endif
