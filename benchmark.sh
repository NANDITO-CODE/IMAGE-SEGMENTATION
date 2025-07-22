#!/bin/bash

# Script de benchmark para comparar rendimiento entre versión serial y MPI

echo "=== BENCHMARK DE SEGMENTACIÓN DE IMÁGENES ==="
echo "Compilando..."
make clean &> /dev/null
make all &> /dev/null

if [ $? -ne 0 ]; then
    echo "Error en la compilación"
    exit 1
fi

# Parámetros de prueba
SIGMA=0.5
K=500
MIN_SIZE=20
INPUT_IMAGE="image_data/peppers.pnm"

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Usando imagen por defecto..."
    INPUT_IMAGE=$(find image_data -name "*.pnm" | head -1)
fi

echo "Imagen de prueba: $INPUT_IMAGE"
echo "Parámetros: sigma=$SIGMA, k=$K, min_size=$MIN_SIZE"
echo ""

# Función para medir tiempo
measure_time() {
    local cmd="$1"
    local output_file="$2"
    
    echo "Ejecutando: $cmd"
    local start_time=$(date +%s.%N)
    eval $cmd > /dev/null 2>&1
    local end_time=$(date +%s.%N)
    
    if [ $? -eq 0 ]; then
        local elapsed=$(echo "$end_time - $start_time" | bc -l)
        printf "Tiempo: %.3f segundos\n" $elapsed
        return 0
    else
        echo "Error en la ejecución"
        return 1
    fi
}

echo "=== VERSIÓN SERIAL ==="
measure_time "./segment $SIGMA $K $MIN_SIZE $INPUT_IMAGE salida-serial.ppm"
echo ""

echo "=== VERSIÓN MPI ==="
for np in 1 2 4 8; do
    echo "--- Con $np procesos ---"
    measure_time "mpirun -np $np ./segment-mpi $SIGMA $K $MIN_SIZE $INPUT_IMAGE salida-mpi-$np.ppm"
    echo ""
done

echo "=== RESUMEN DE ARCHIVOS GENERADOS ==="
ls -la salida-*.ppm 2>/dev/null || echo "No se generaron archivos de salida"

echo ""
echo "=== VERIFICACIÓN DE DIFERENCIAS ==="
if command -v compare &> /dev/null; then
    echo "Comparando resultados con ImageMagick..."
    for f in salida-mpi-*.ppm; do
        if [ -f "$f" ] && [ -f "salida-serial.ppm" ]; then
            echo "Diferencia entre serial y $f:"
            compare -metric RMSE salida-serial.ppm "$f" null: 2>&1 || true
        fi
    done
else
    echo "ImageMagick no disponible para comparar imágenes"
    echo "Instalalo con: sudo apt-get install imagemagick"
fi
