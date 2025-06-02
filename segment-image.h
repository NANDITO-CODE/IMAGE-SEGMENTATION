/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE

#include <cstdlib>
#include "image.h"
#include "misc.h"
#include "filter.h"
#include "segment-graph.h"
#include "cuda_image_ops.h"  // para funciones host CUDA

// random color
rgb random_rgb(){ 
  rgb c;
  double r;
  
  c.r = (uchar)random();
  c.g = (uchar)random();
  c.b = (uchar)random();

  return c;
}

// dissimilarity measure between pixels
static inline float diff(image<float> *r, image<float> *g, image<float> *b,
                         int x1, int y1, int x2, int y2) {
  return sqrt(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
              square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
              square(imRef(b, x1, y1)-imRef(b, x2, y2)));
}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for threshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
image<rgb> *segment_image(image<rgb> *im, float sigma, float c, int min_size,
                          int *num_ccs) {
  int width = im->width();
  int height = im->height();

  image<float> *r = new image<float>(width, height);
  image<float> *g = new image<float>(width, height);
  image<float> *b = new image<float>(width, height);

  // Copiar canales R,G,B
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      imRef(r, x, y) = imRef(im, x, y).r;
      imRef(g, x, y) = imRef(im, x, y).g;
      imRef(b, x, y) = imRef(im, x, y).b;
    }
  }

  // Suavizado con CUDA
  std::vector<float> mask = make_fgauss(sigma);
  normalize(mask);

  image<float> *smooth_r = new image<float>(width, height, false);
  image<float> *smooth_g = new image<float>(width, height, false);
  image<float> *smooth_b = new image<float>(width, height, false);

  convolveEvenCUDA(imPtr(r, 0, 0), imPtr(smooth_r, 0, 0), width, height, mask.data(), (int)mask.size());
  convolveEvenCUDA(imPtr(g, 0, 0), imPtr(smooth_g, 0, 0), width, height, mask.data(), (int)mask.size());
  convolveEvenCUDA(imPtr(b, 0, 0), imPtr(smooth_b, 0, 0), width, height, mask.data(), (int)mask.size());

  delete r;
  delete g;
  delete b;

  // Construcción del grafo (secuencial)
  edge *edges = new edge[width*height*4];
  int num = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x < width-1) {
        edges[num].a = y * width + x;
        edges[num].b = y * width + (x+1);
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
        num++;
      }
      if (y < height-1) {
        edges[num].a = y * width + x;
        edges[num].b = (y+1) * width + x;
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
        num++;
      }
      if ((x < width-1) && (y < height-1)) {
        edges[num].a = y * width + x;
        edges[num].b = (y+1) * width + (x+1);
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
        num++;
      }
      if ((x < width-1) && (y > 0)) {
        edges[num].a = y * width + x;
        edges[num].b = (y-1) * width + (x+1);
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
        num++;
      }
    }
  }

  delete smooth_r;
  delete smooth_g;
  delete smooth_b;

  // Segmentación
  universe *u = segment_graph(width*height, num, edges, c);

  // Post proceso componentes pequeñas
  for (int i = 0; i < num; i++) {
    int a = u->find(edges[i].a);
    int b = u->find(edges[i].b);
    if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
      u->join(a, b);
  }
  delete [] edges;
  *num_ccs = u->num_sets();

  image<rgb> *output = new image<rgb>(width, height);

  // Preparar colores para GPU
  rgb *colors = new rgb[width*height];
  for (int i = 0; i < width*height; i++)
    colors[i] = random_rgb();

  // Reservar memoria GPU
  rgb* d_output;
  int* d_components;
  rgb* d_colors;
  int n = width * height;

  cudaMalloc(&d_output, n * sizeof(rgb));
  cudaMalloc(&d_components, n * sizeof(int));
  cudaMalloc(&d_colors, n * sizeof(rgb));

  // Preparar arreglo components desde Union-Find
  int* h_components = new int[n];
  for (int i = 0; i < n; i++)
    h_components[i] = u->find(i);

  cudaMemcpy(d_components, h_components, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_colors, colors, n * sizeof(rgb), cudaMemcpyHostToDevice);

  // Llamar a la función host CUDA para colorear
  colorComponentsCUDA(d_output, d_components, d_colors, width, height);

  // Copiar resultado a host
  cudaMemcpy(imPtr(output, 0, 0), d_output, n * sizeof(rgb), cudaMemcpyDeviceToHost);

  // Liberar memoria GPU y host temporal
  cudaFree(d_output);
  cudaFree(d_components);
  cudaFree(d_colors);
  delete [] h_components;
  delete [] colors;
  delete u;

  return output;
}

#endif
