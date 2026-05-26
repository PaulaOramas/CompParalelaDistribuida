#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# DNA Cloud Worker Setup
# ═══════════════════════════════════════════════════════════════
# Clona el repo y conecta al coordinador en Google Cloud.
#
# Uso (ejecutar en la máquina del worker):
#   bash setup_cloud_worker.sh <TU_NOMBRE> [CLAVE_SECRETA]
#
# Ejemplo:
#   bash setup_cloud_worker.sh juan dna2024
#
# O en un solo comando sin clonar antes:
#   bash <(curl -s https://raw.githubusercontent.com/TU_USUARIO/TU_REPO/main/setup_cloud_worker.sh) juan dna2024
# ═══════════════════════════════════════════════════════════════

set -e

# ─── CONFIGURACIÓN — editar estos valores ───────────────────────
COORDINATOR_IP="34.68.177.178"       # IP pública de la VM Google Cloud
COORDINATOR_PORT="5555"
REPO_URL="https://github.com/PaulaOramas/CompParalelaDistribuida"  # ← cambiar por tu repo
REPO_DIR="dna_checker"
# ────────────────────────────────────────────────────────────────

WORKER_NAME="${1:-worker-$(hostname)}"
SECRET="${2:-}"

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   🧬 DNA Cloud Worker Setup              ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  Coordinador : $COORDINATOR_IP:$COORDINATOR_PORT"
echo "  Nombre      : $WORKER_NAME"
echo "  Seguridad   : $([ -n "$SECRET" ] && echo '🔐 Clave activa' || echo '⚠️  Sin clave')"
echo ""

# ── Verificar Python ───────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "  ❌ Python 3 no encontrado."
    echo "     Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "     macOS:         brew install python3"
    exit 1
fi
echo "  ✅ Python: $(python3 --version)"

# ── Clonar repo si no existe ───────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    echo "  📥 Clonando repositorio..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "  🔄 Actualizando repositorio..."
    cd "$REPO_DIR" && git pull && cd ..
fi

cd "$REPO_DIR"

# ── Crear entorno virtual ──────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "  📦 Creando entorno virtual..."
    python3 -m venv venv
fi

source venv/bin/activate

# ── Instalar dependencias ──────────────────────────────────────
echo "  📦 Instalando dependencias..."

if [ -f "requirements_distributed.txt" ]; then
    pip install --quiet -r requirements_distributed.txt
else
    pip install --quiet pyzmq pyopencl numpy flask
fi

# ── Detectar GPU ──────────────────────────────────────────────
echo ""
python3 -c "
try:
    import pyopencl as cl
    platforms = cl.get_platforms()
    found = False
    for p in platforms:
        for d in p.get_devices():
            if d.type & cl.device_type.GPU:
                print(f'  🎮 GPU detectada: {d.name.strip()}')
                print(f'     Compute Units : {d.max_compute_units}')
                print(f'     Max WorkGroup : {d.max_work_group_size}')
                print(f'     Memoria       : {d.global_mem_size / (1024**2):.0f} MB')
                found = True
                break
        if found:
            break
    if not found:
        print('  ⚠️  No se detectó GPU — el worker usará CPU')
except ImportError:
    print('  ⚠️  PyOpenCL no instalado — el worker usará CPU')
except Exception as e:
    print(f'  ⚠️  Error GPU: {e} — el worker usará CPU')
" 2>/dev/null || echo "  ⚠️  No se pudo verificar GPU"

# ── Conectar al coordinador ────────────────────────────────────
echo ""
echo "  🚀 Conectando al coordinador $COORDINATOR_IP:$COORDINATOR_PORT..."
echo "  📌 Ctrl+C para detener"
echo ""

if [ -n "$SECRET" ]; then
    python3 dna_distributed_node.py \
        --coordinator "$COORDINATOR_IP:$COORDINATOR_PORT" \
        --name "$WORKER_NAME" \
        --secret "$SECRET"
else
    python3 dna_distributed_node.py \
        --coordinator "$COORDINATOR_IP:$COORDINATOR_PORT" \
        --name "$WORKER_NAME"
fi