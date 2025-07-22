#!/bin/bash

# Script para generar visualización del análisis de rendimiento

echo "=== ANÁLISIS DETALLADO DE LA SOLUCIÓN MPI ==="
echo ""

echo "📊 ARQUITECTURA DEL SISTEMA:"
echo ""
echo "   ┌─────────────────────────────────────────────────┐"
echo "   │              IMAGEN ORIGINAL (W×H)              │"
echo "   └─────────────────────────────────────────────────┘"
echo "                            │"
echo "                            ▼"
echo "   ┌─────────────────────────────────────────────────┐"
echo "   │                PROCESO 0                        │"
echo "   │           (Carga y distribución)                │"
echo "   └─────────────────────────────────────────────────┘"
echo "                            │"
echo "                      MPI_Scatterv"
echo "                            │"
echo "   ┌──────────┬──────────┬──────────┬──────────────┐"
echo "   │ Proceso 0│ Proceso 1│ Proceso 2│   Proceso P-1│"
echo "   │  (filas  │  (filas  │  (filas  │    (filas    │"
echo "   │  0-H/P)  │ H/P-2H/P)│2H/P-3H/P)│ (P-1)H/P-H) │"
echo "   └──────────┴──────────┴──────────┴──────────────┘"
echo "                            │"
echo "                ┌───────────┼───────────┐"
echo "                ▼           ▼           ▼"
echo "   ┌─────────────────┬─────────────────┬─────────────────┐"
echo "   │  Suavizado      │ Construcción    │ Comunicación    │"
echo "   │  Gaussiano      │ de Grafo        │ Inter-Proceso   │"
echo "   │  Local          │ Local           │ (Bordes)        │"
echo "   └─────────────────┴─────────────────┴─────────────────┘"
echo "                            │"
echo "                            ▼"
echo "   ┌─────────────────────────────────────────────────┐"
echo "   │              SEGMENTACIÓN LOCAL                 │"
echo "   │           (Con conectividad global)             │"
echo "   └─────────────────────────────────────────────────┘"
echo "                            │"
echo "                      MPI_Gatherv"
echo "                            ▼"
echo "   ┌─────────────────────────────────────────────────┐"
echo "   │                RESULTADO FINAL                  │"
echo "   │             (Imagen segmentada)                 │"
echo "   └─────────────────────────────────────────────────┘"
echo ""

echo "🔄 FLUJO DE COMUNICACIÓN MPI:"
echo ""
echo "   Proceso 0  ←→  Proceso 1  ←→  Proceso 2  ←→  ... ←→  Proceso P-1"
echo "     │                │              │                      │"
echo "   Primera            Primera        Primera               Primera"
echo "   y última           y última       y última             y última"
echo "   fila               fila           fila                  fila"
echo ""

echo "🧮 ANÁLISIS DE COMPLEJIDAD:"
echo ""
echo "   Serial:    O(E log E)     donde E = W×H×4"
echo "   Paralelo:  O(E/P log E/P + W)  donde P = procesos"
echo "   Memoria:   O(W×H/P)       por proceso"
echo "   Comunicación: O(W)        por intercambio de bordes"
echo ""

echo "📈 FACTORES DE RENDIMIENTO:"
echo ""
echo "   ✅ Ventajas:"
echo "   - Distribución balanceada de carga"
echo "   - Comunicación mínima (solo bordes)"
echo "   - Escalabilidad lineal teórica"
echo "   - Preservación de conectividad global"
echo ""
echo "   ⚠️  Limitaciones:"
echo "   - Overhead de comunicación MPI"
echo "   - Speedup sublineal para imágenes pequeñas"
echo "   - Sincronización entre procesos"
echo ""

echo "🎯 PARÁMETROS CRÍTICOS:"
echo ""
echo "   - Tamaño de imagen: Mayor → Mejor speedup"
echo "   - Número de procesos: Óptimo ≈ Número de cores"
echo "   - Sigma (suavizado): 0.5-1.0"
echo "   - K (umbral): 500-1000"
echo "   - Min_size: 20-100"
echo ""

echo "🔧 OPTIMIZACIONES IMPLEMENTADAS:"
echo ""
echo "   1. MPI_Sendrecv para comunicación bidireccional"
echo "   2. Índices especiales para píxeles externos"
echo "   3. Gestión eficiente de memoria"
echo "   4. Balance automático de carga"
echo "   5. Minimización de comunicación colectiva"
echo ""

# Análisis de archivos si existen
if [ -f "salida-serial-demo.ppm" ] && [ -f "salida-mpi-2p-demo.ppm" ]; then
    echo "📁 ANÁLISIS DE RESULTADOS:"
    echo ""
    echo "   Archivos generados:"
    ls -la salida-*-demo.ppm | while read -r line; do
        echo "   $line"
    done
    echo ""
    
    # Verificar si ImageMagick está disponible
    if command -v compare &> /dev/null; then
        echo "   🔍 Comparación de resultados (RMSE):"
        for f in salida-mpi-*-demo.ppm; do
            if [ -f "$f" ]; then
                result=$(compare -metric RMSE salida-serial-demo.ppm "$f" null: 2>&1 || true)
                echo "   Serial vs $(basename "$f"): $result"
            fi
        done
    else
        echo "   ℹ️  ImageMagick no disponible para comparación"
    fi
fi

echo ""
echo "=== ANÁLISIS COMPLETADO ==="
