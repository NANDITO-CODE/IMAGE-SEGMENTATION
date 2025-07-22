# Segmentación de Imágenes con MPI

Este proyecto implementa el algoritmo de segmentación de grafos de Felzenszwalb-Huttenlocher con soporte para procesamiento paralelo usando MPI.

## Archivos principales

- `segment.cpp` - Versión serial original
- `segment-mpi.cpp` - Versión paralela con MPI
- `segment-image.h` - Funciones de segmentación serial
- `segment-image-mpi.h` - Funciones de segmentación paralela con MPI
- `segment-graph.h` - Algoritmo de segmentación de grafos
- `disjoint-set.h` - Estructura de datos union-find

## Compilación

```bash
# Compilar todas las versiones
make all

# O compilar individualmente
make segment      # versión serial
make segment-mpi  # versión MPI
```

## Ejecución

### Versión Serial
```bash
./segment <sigma> <k> <min_size> <input.ppm> <output.ppm>
```

### Versión MPI
```bash
mpirun -np <num_procesos> ./segment-mpi <sigma> <k> <min_size> <input.ppm> <output.ppm>
```

### Ejemplos
```bash
# Serial
./segment 0.5 500 20 image_data/peppers.pnm salida-serial.ppm

# MPI con 4 procesos
mpirun -np 4 ./segment-mpi 0.5 500 20 image_data/peppers.pnm salida-mpi.ppm
```

## Scripts de prueba

### Ejecución rápida
```bash
# En WSL (desde el directorio del proyecto)
chmod +x run_test.sh benchmark.sh

# Ejecutar prueba con parámetros por defecto
./run_test.sh

# Ejecutar con parámetros personalizados
./run_test.sh <sigma> <k> <min_size> <imagen> <num_procesos>
```

### Benchmark de rendimiento
```bash
./benchmark.sh
```

## Parámetros

- **sigma**: Parámetro de suavizado gaussiano (típicamente 0.5-1.0)
- **k**: Parámetro de umbral para la función de segmentación (típicamente 500-1000)
- **min_size**: Tamaño mínimo de componente (típicamente 20-100)

## Mejoras implementadas en la versión MPI

1. **Distribución balanceada**: Los píxeles se distribuyen uniformemente entre procesos
2. **Comunicación de bordes**: Los procesos intercambian información de píxeles frontera
3. **Conectividad global**: Se mantiene la conectividad entre regiones que cruzan fronteras de procesos
4. **Recolección eficiente**: Los resultados se reúnen de manera optimizada

## Estructura de la paralelización

La imagen se divide horizontalmente entre procesos:
- Proceso 0: filas 0 a N/P-1
- Proceso 1: filas N/P a 2*N/P-1
- ...
- Proceso P-1: filas (P-1)*N/P a N-1

Cada proceso:
1. Recibe su bloque de filas
2. Aplica suavizado gaussiano
3. Construye grafo local + aristas de frontera
4. Ejecuta segmentación local
5. Envía resultado al proceso 0

## Imágenes de prueba

El directorio `image_data/` contiene varias imágenes de prueba en formato PNM/PPM.

## Comandos útiles en WSL

```bash
# Compilar
make all

# Ejecutar versión serial
./segment 0.5 500 20 image_data/peppers.pnm salida.ppm

# Ejecutar versión MPI
mpirun -np 4 ./segment-mpi 0.5 500 20 image_data/peppers.pnm salida-mpi.ppm

# Ver resultados (si tienes display instalado)
display salida.ppm
display salida-mpi.ppm

# Comparar archivos
ls -la salida*.ppm
```

## Requisitos

- Compilador C++ (g++)
- MPI (OpenMPI o MPICH)
- make
- bc (para scripts de benchmark)

En Ubuntu/WSL:
```bash
sudo apt-get update
sudo apt-get install build-essential openmpi-bin openmpi-dev bc
```
