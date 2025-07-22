#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <image.h>
#include <misc.h>
#include <pnmfile.h>
#include "segment-image-mpi.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 6) {
    if (rank == 0)
      fprintf(stderr, "usage: %s sigma k min input.ppm output.ppm\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  float sigma = atof(argv[1]);
  float k = atof(argv[2]);
  int min_size = atoi(argv[3]);

  image<rgb> *input = nullptr;
  int width = 0, height = 0;

  // Solo el proceso 0 carga la imagen
  if (rank == 0) {
    printf("[RANK 0] Loading image: %s\n", argv[4]);
    input = loadPPM(argv[4]);
    width = input->width();
    height = input->height();
    printf("[RANK 0] Image size: %dx%d pixels\n", width, height);
  }

  // Broadcast dimensions to all processes
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Calculate work distribution for each process
  int local_start = (rank * height) / size;
  int local_end = ((rank + 1) * height) / size;
  int local_rows = local_end - local_start;
  
  if (rank == 0) {
    printf("[RANK 0] Distributing work among %d processes\n", size);
    for (int i = 0; i < size; i++) {
      int start = (i * height) / size;
      int end = ((i + 1) * height) / size;
      int rows = end - start;
      printf("  Process %d: rows %d-%d (%d rows)\n", i, start, end-1, rows);
    }
  }

  // Allocate space for local image block
  rgb *local_block = new rgb[width * local_rows];

  // Distribute image data to all processes
  if (rank == 0) {
    rgb *flat = input->data;
    int *sendcounts = new int[size];
    int *displs = new int[size];
    
    for (int i = 0; i < size; i++) {
      int start = (i * height) / size;
      int end = ((i + 1) * height) / size;
      int rows = end - start;
      sendcounts[i] = rows * width * sizeof(rgb);
      displs[i] = start * width * sizeof(rgb);
    }
    
    printf("[RANK 0] Scattering image data...\n");
    MPI_Scatterv(flat, sendcounts, displs, MPI_BYTE,
                 local_block, width * local_rows * sizeof(rgb), MPI_BYTE,
                 0, MPI_COMM_WORLD);
    delete[] sendcounts;
    delete[] displs;
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_BYTE,
                 local_block, width * local_rows * sizeof(rgb), MPI_BYTE,
                 0, MPI_COMM_WORLD);
  }

  // Reconstruct local image block
  image<rgb> *local_img = new image<rgb>(width, local_rows);
  memcpy(local_img->data, local_block, width * local_rows * sizeof(rgb));
  delete[] local_block;

  printf("[RANK %d] Processing local block (%dx%d)\n", rank, width, local_rows);
  
  // Parallel processing using MPI-aware segmentation
  int num_ccs = 0;
  image<rgb> *result = segment_image_mpi(local_img, sigma, k, min_size, &num_ccs, 
                                        rank, size, width, height, local_start);
  
  printf("[RANK %d] Local segmentation complete, found %d components\n", rank, num_ccs);
  delete local_img;

  // Gather segmented image from all processes
  rgb* full_data = nullptr;
  int* recvcounts = new int[size];
  int* displs = new int[size];
  
  for (int i = 0; i < size; i++) {
    int start = (i * height) / size;
    int end = ((i + 1) * height) / size;
    int rows = end - start;
    recvcounts[i] = rows * width * sizeof(rgb);
    displs[i] = start * width * sizeof(rgb);
  }
  
  if (rank == 0) {
    full_data = new rgb[width * height];
    printf("[RANK 0] Gathering results from all processes...\n");
  }

  MPI_Gatherv(result->data, width * local_rows * sizeof(rgb), MPI_BYTE,
              full_data, recvcounts, displs, MPI_BYTE,
              0, MPI_COMM_WORLD);

  // Collect total number of components from all processes
  int total_components = 0;
  MPI_Reduce(&num_ccs, &total_components, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    // Reconstruct and save the final segmented image
    image<rgb>* full_img = new image<rgb>(width, height);
    memcpy(full_img->data, full_data, width * height * sizeof(rgb));
    
    printf("[RANK 0] Saving segmented image to: %s\n", argv[5]);
    savePPM(full_img, argv[5]);
    printf("[RANK 0] Total components found across all processes: %d\n", total_components);
    printf("[RANK 0] Segmentation complete!\n");
    
    delete full_img;
    delete[] full_data;
    delete input;
  }

  delete[] recvcounts;
  delete[] displs;
  delete result;

  MPI_Finalize();
  return 0;
}
