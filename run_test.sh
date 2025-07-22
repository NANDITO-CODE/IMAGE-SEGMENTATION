#!/bin/bash

# Script para ejecutar y comparar las versiones serial y paralela
# del algoritmo de segmentación

echo "=== Compilando versiones serial y MPI ==="
make clean
make all

if [ $? -ne 0 ]; then
    echo "Error en la compilación"
    exit 1
fi

# Parámetros por defecto
SIGMA=${1:-0.5}
K=${2:-500}
MIN_SIZE=${3:-20}
INPUT_IMAGE=${4:-"image_data/peppers.pnm"}
NUM_PROCS=${5:-4}

echo "=== Parámetros de ejecución ==="
echo "Sigma: $SIGMA"
echo "K: $K"
echo "Min Size: $MIN_SIZE"
echo "Input Image: $INPUT_IMAGE"
echo "Number of processes: $NUM_PROCS"
echo ""

# Verificar que la imagen existe
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Error: La imagen $INPUT_IMAGE no existe"
    echo "Imágenes disponibles:"
    ls image_data/*.pnm | head -10
    exit 1
fi

echo "=== Ejecutando versión serial ==="
time ./segment $SIGMA $K $MIN_SIZE $INPUT_IMAGE salida-serial.ppm
echo ""

echo "=== Ejecutando versión MPI con $NUM_PROCS procesos ==="
time mpirun -np $NUM_PROCS ./segment-mpi $SIGMA $K $MIN_SIZE $INPUT_IMAGE salida-mpi.ppm
echo ""

echo "=== Resultados ==="
echo "Imágenes generadas:"
ls -la salida-*.ppm

echo ""
echo "Para ver las imágenes, usa un visor como:"
echo "  display salida-serial.ppm"
echo "  display salida-mpi.ppm"
echo ""
echo "O cópialas a Windows y ábrelas con cualquier visor de imágenes."
