/*
Copyright (C) 2006 Pedro Felzenszwalb
Modified for MPI parallelization

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

#ifndef SEGMENT_IMAGE_MPI
#define SEGMENT_IMAGE_MPI

#include <cstdlib>
#include <mpi.h>
#include <image.h>
#include <misc.h>
#include <filter.h>
#include "segment-graph.h"

// Estructura para intercambiar información de bordes entre procesos
typedef struct {
    int global_id;     // ID global del pixel
    float r, g, b;     // Valores de color suavizados
    int local_id;      // ID local en el proceso
} border_pixel;

// random color
rgb random_rgb(){ 
  rgb c;
  
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

// Calcula diferencia usando valores directos (para bordes entre procesos)
static inline float diff_direct(float r1, float g1, float b1, 
                                float r2, float g2, float b2) {
  return sqrt(square(r1-r2) + square(g1-g2) + square(b1-b2));
}

/*
 * Segment an image block in parallel using MPI
 *
 * Returns a color image representing the segmentation of the local block.
 *
 * local_im: local image block to segment.
 * sigma: to smooth the image.
 * c: constant for threshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the local segmentation.
 * rank: MPI rank of current process.
 * size: total number of MPI processes.
 * global_width: width of the complete image.
 * global_height: height of the complete image.
 * local_start_row: starting row of this block in the global image.
 */
image<rgb> *segment_image_mpi(image<rgb> *local_im, float sigma, float c, int min_size,
                             int *num_ccs, int rank, int size, 
                             int global_width, int global_height, 
                             int local_start_row) {
  int width = local_im->width();
  int height = local_im->height();

  image<float> *r = new image<float>(width, height);
  image<float> *g = new image<float>(width, height);
  image<float> *b = new image<float>(width, height);

  // smooth each color channel  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      imRef(r, x, y) = imRef(local_im, x, y).r;
      imRef(g, x, y) = imRef(local_im, x, y).g;
      imRef(b, x, y) = imRef(local_im, x, y).b;
    }
  }
  image<float> *smooth_r = smooth(r, sigma);
  image<float> *smooth_g = smooth(g, sigma);
  image<float> *smooth_b = smooth(b, sigma);
  delete r;
  delete g;
  delete b;

  // Calcular número máximo de aristas locales + aristas de frontera
  int max_edges = width * height * 4 + width * 2; // extra espacio para bordes entre procesos
  edge *edges = new edge[max_edges];
  int num = 0;

  // Construir grafo local
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int current_global_id = (local_start_row + y) * global_width + x;
      
      // Arista horizontal derecha
      if (x < width-1) {
        edges[num].a = y * width + x;
        edges[num].b = y * width + (x+1);
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
        num++;
      }

      // Arista vertical abajo
      if (y < height-1) {
        edges[num].a = y * width + x;
        edges[num].b = (y+1) * width + x;
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
        num++;
      }

      // Arista diagonal abajo-derecha
      if ((x < width-1) && (y < height-1)) {
        edges[num].a = y * width + x;
        edges[num].b = (y+1) * width + (x+1);
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
        num++;
      }

      // Arista diagonal arriba-derecha
      if ((x < width-1) && (y > 0)) {
        edges[num].a = y * width + x;
        edges[num].b = (y-1) * width + (x+1);
        edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
        num++;
      }
    }
  }

  // Intercambiar información de bordes con procesos vecinos
  // Solo necesitamos la primera y última fila para conectividad vertical
  if (size > 1) {
    // Preparar datos de la primera fila para enviar al proceso anterior
    if (rank > 0) {
      border_pixel *first_row = new border_pixel[width];
      for (int x = 0; x < width; x++) {
        first_row[x].global_id = local_start_row * global_width + x;
        first_row[x].r = imRef(smooth_r, x, 0);
        first_row[x].g = imRef(smooth_g, x, 0);
        first_row[x].b = imRef(smooth_b, x, 0);
        first_row[x].local_id = x;
      }
      
      // Recibir última fila del proceso anterior
      border_pixel *prev_last_row = new border_pixel[width];
      MPI_Sendrecv(first_row, width * sizeof(border_pixel), MPI_BYTE, rank-1, 0,
                   prev_last_row, width * sizeof(border_pixel), MPI_BYTE, rank-1, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      // Crear aristas con el proceso anterior
      for (int x = 0; x < width; x++) {
        // Arista vertical hacia arriba
        edges[num].a = x; // primera fila local
        edges[num].b = width * height + x; // índice especial para pixeles externos
        edges[num].w = diff_direct(imRef(smooth_r, x, 0), imRef(smooth_g, x, 0), imRef(smooth_b, x, 0),
                                  prev_last_row[x].r, prev_last_row[x].g, prev_last_row[x].b);
        num++;
      }
      
      delete[] first_row;
      delete[] prev_last_row;
    }

    // Preparar datos de la última fila para enviar al proceso siguiente
    if (rank < size - 1) {
      border_pixel *last_row = new border_pixel[width];
      for (int x = 0; x < width; x++) {
        last_row[x].global_id = (local_start_row + height - 1) * global_width + x;
        last_row[x].r = imRef(smooth_r, x, height-1);
        last_row[x].g = imRef(smooth_g, x, height-1);
        last_row[x].b = imRef(smooth_b, x, height-1);
        last_row[x].local_id = (height-1) * width + x;
      }
      
      // Recibir primera fila del proceso siguiente
      border_pixel *next_first_row = new border_pixel[width];
      MPI_Sendrecv(last_row, width * sizeof(border_pixel), MPI_BYTE, rank+1, 1,
                   next_first_row, width * sizeof(border_pixel), MPI_BYTE, rank+1, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      // Crear aristas con el proceso siguiente
      for (int x = 0; x < width; x++) {
        // Arista vertical hacia abajo
        edges[num].a = (height-1) * width + x; // última fila local
        edges[num].b = width * height + width + x; // índice especial para pixeles externos
        edges[num].w = diff_direct(imRef(smooth_r, x, height-1), imRef(smooth_g, x, height-1), imRef(smooth_b, x, height-1),
                                  next_first_row[x].r, next_first_row[x].g, next_first_row[x].b);
        num++;
      }
      
      delete[] last_row;
      delete[] next_first_row;
    }
  }

  delete smooth_r;
  delete smooth_g;
  delete smooth_b;

  // Segmentar el grafo local (incluyendo aristas de frontera)
  universe *u = segment_graph(width * height + 2 * width, num, edges, c);
  
  // Post-procesar componentes pequeños
  for (int i = 0; i < num; i++) {
    int a = u->find(edges[i].a);
    int b = u->find(edges[i].b);
    if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
      u->join(a, b);
  }
  delete [] edges;
  *num_ccs = u->num_sets();

  image<rgb> *output = new image<rgb>(width, height);

  // Asignar colores aleatorios a cada componente
  rgb *colors = new rgb[width * height + 2 * width];
  for (int i = 0; i < width * height + 2 * width; i++)
    colors[i] = random_rgb();
  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int comp = u->find(y * width + x);
      imRef(output, x, y) = colors[comp];
    }
  }  

  delete [] colors;  
  delete u;

  return output;
}

#endif
