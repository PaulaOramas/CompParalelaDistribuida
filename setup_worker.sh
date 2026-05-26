#!/bin/bash
# ═══════════════════════════════════════════════════════════
# DNA Distributed Worker Setup v2.0
# ═══════════════════════════════════════════════════════════
# Este script configura un nodo worker para conectarse
# al coordinador distribuido de comparación de ADN.
#
# Uso:
#   chmod +x setup_worker.sh
#   ./setup_worker.sh <COORDINATOR_IP> <COORDINATOR_PORT> [WORKER_NAME]
#
# Ejemplo:
#   ./setup_worker.sh 192.168.1.100 5555 mi-worker
# ═══════════════════════════════════════════════════════════

set -e

COORDINATOR_IP="${1:-}"
COORDINATOR_PORT="${2:-5555}"
WORKER_NAME="${3:-}"

if [ -z "$COORDINATOR_IP" ]; then
    echo ""
    echo "  🧬 DNA Distributed Worker Setup v2.0"
    echo "  ════════════════════════════════════"
    echo ""
    echo "  Uso: $0 <COORDINATOR_IP> [PORT] [WORKER_NAME]"
    echo ""
    echo "  Ejemplo:"
    echo "    $0 192.168.1.100 5555 mi-worker"
    echo ""
    echo "  Características del worker:"
    echo "    - Procesamiento GPU (OpenCL) para comparación"
    echo "    - Failover: se convierte en coordinador si el maestro cae"
    echo "    - Configuración remota de GPU desde la interfaz web"
    echo ""
    exit 1
fi

echo ""
echo "  🧬 DNA Distributed Worker Setup v2.0"
echo "  ════════════════════════════════════"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "  ❌ Python 3 no encontrado. Instálalo primero."
    exit 1
fi

PYTHON=$(command -v python3)
echo "  ✅ Python: $($PYTHON --version 2>&1)"

# Create virtual environment if needed
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "  📦 Creando entorno virtual..."
    $PYTHON -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install dependencies
echo "  📦 Instalando dependencias..."
pip install --quiet pyzmq pyopencl numpy 2>/dev/null || true

# Check GPU
echo ""
python3 -c "
try:
    import pyopencl as cl
    platforms = cl.get_platforms()
    for p in platforms:
        for d in p.get_devices():
            if d.type & cl.device_type.GPU:
                print(f'  🎮 GPU detectada: {d.name.strip()}')
                print(f'     Compute Units: {d.max_compute_units}')
                print(f'     Max Work Group: {d.max_work_group_size}')
                print(f'     Memoria: {d.global_mem_size / (1024**2):.0f} MB')
                exit(0)
    print('  ⚠️  No se detectó GPU — usará CPU como fallback')
except ImportError:
    print('  ⚠️  PyOpenCL no instalado — usará CPU como fallback')
except Exception as e:
    print(f'  ⚠️  Error GPU: {e} — usará CPU como fallback')
"

# Start worker
echo ""
echo "  🚀 Conectando al coordinador: $COORDINATOR_IP:$COORDINATOR_PORT"
echo ""

if [ -n "$WORKER_NAME" ]; then
    python3 "$SCRIPT_DIR/dna_distributed_node.py" \
        --coordinator "$COORDINATOR_IP:$COORDINATOR_PORT" \
        --name "$WORKER_NAME"
else
    python3 "$SCRIPT_DIR/dna_distributed_node.py" \
        --coordinator "$COORDINATOR_IP:$COORDINATOR_PORT"
fi
