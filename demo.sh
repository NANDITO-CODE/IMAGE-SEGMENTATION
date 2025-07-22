#!/bin/bash

# Demostración completa del sistema de segmentación MPI
# Ejecuta desde PowerShell con: wsl -d Ubuntu -e bash -c "cd /home/nandito/programacionParalela/segment3 && ./demo.sh"

echo "==================================================================="
echo "    DEMOSTRACIÓN DE SEGMENTACIÓN DE IMÁGENES CON MPI"
echo "==================================================================="
echo ""

# Verificar que todo esté compilado
if [ ! -f "segment" ] || [ ! -f "segment-mpi" ]; then
    echo "Compilando el proyecto..."
    make clean > /dev/null 2>&1
    make all > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Error en la compilación"
        exit 1
    fi
    echo "✅ Compilación exitosa"
    echo ""
fi

# Configuración de prueba
SIGMA=0.5
K=500
MIN_SIZE=20
INPUT_IMAGE="image_data/machupicchu.pnm"

echo "Configuración de la prueba:"
echo "  📸 Imagen: $INPUT_IMAGE"
echo "  🎛️  Sigma: $SIGMA (suavizado)"
echo "  🎛️  K: $K (umbral)" 
echo "  🎛️  Min Size: $MIN_SIZE (componente mínimo)"
echo ""

# Verificar imagen de entrada
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "❌ Imagen no encontrada, usando la primera disponible..."
    INPUT_IMAGE=$(find image_data -name "*.pnm" | head -1)
    echo "  📸 Nueva imagen: $INPUT_IMAGE"
fi

echo "=== VERSIÓN SERIAL ==="
echo "⏱️  Ejecutando versión serial..."
time_start=$(date +%s.%N)
./segment $SIGMA $K $MIN_SIZE $INPUT_IMAGE machupicchu-fianl.ppm 2>/dev/null
time_end=$(date +%s.%N)
serial_time=$(echo "$time_end - $time_start" | bc -l)
printf "✅ Completado en %.3f segundos\n" $serial_time
echo ""

echo "=== VERSIONES MPI ==="
declare -a processes=(1 2 4)
declare -a mpi_times=()

for np in "${processes[@]}"; do
    echo "⏱️  Ejecutando con $np proceso(s)..."
    time_start=$(date +%s.%N)
    mpirun -np $np ./segment-mpi $SIGMA $K $MIN_SIZE $INPUT_IMAGE salida-mpi-${np}p-demo.ppm 2>/dev/null | grep -v "\[RANK"
    time_end=$(date +%s.%N)
    mpi_time=$(echo "$time_end - $time_start" | bc -l)
    mpi_times+=($mpi_time)
    printf "✅ Completado en %.3f segundos\n" $mpi_time
    
    # Calcular speedup
    if (( $(echo "$serial_time > 0" | bc -l) )); then
        speedup=$(echo "scale=2; $serial_time / $mpi_time" | bc -l)
        printf "📈 Speedup: %.2fx\n" $speedup
    fi
    echo ""
done

echo "=== RESUMEN DE RESULTADOS ==="
echo "📊 Tiempos de ejecución:"
printf "   Serial:     %.3f segundos\n" $serial_time
for i in "${!processes[@]}"; do
    printf "   MPI (%dp):   %.3f segundos\n" ${processes[$i]} ${mpi_times[$i]}
done

echo ""
echo "📁 Archivos generados:"
ls -la salida-*-demo.ppm 2>/dev/null | while read line; do
    echo "   $line"
done

