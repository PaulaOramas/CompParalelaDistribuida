"""
DNA Parallel Checker v1.0 — Python
Detecta nucleótidos inválidos (no A, T, C, G) en archivos .txt / .fna
usando multiprocessing (CPU) o Numba CUDA (GPU).
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import threading
from pathlib import Path

from tqdm import tqdm
from colored import fg, attr

# ─── GPU availability check ─────────────────────────────────────────────────
GPU_AVAILABLE = False
GPU_NAME = "No disponible"
try:
    import numpy as np
    from numba import cuda, int32
    if cuda.is_available():
        GPU_AVAILABLE = True
        gpu_device = cuda.get_current_device()
        GPU_NAME = gpu_device.name.decode() if isinstance(gpu_device.name, bytes) else str(gpu_device.name)
except (ImportError, Exception) as e:
    GPU_NAME = f"Error: {e}"
    # numpy is still needed even without GPU for some operations
    try:
        import numpy as np
    except ImportError:
        pass

# ─── Constantes ─────────────────────────────────────────────────────────────

VALID       = frozenset("ATCGNatcgn")
RESET       = attr("reset")
CYAN        = fg("cyan")
GREEN       = fg("green")
YELLOW      = fg("yellow")
RED         = fg("red")
BRIGHT_BLUE = fg("dodger_blue_1")
BOLD        = attr("bold")

# ─── Worker (se ejecuta en cada proceso hijo) ────────────────────────────────

def process_chunk(args):
    """
    Recibe una lista de (row_number, line) y devuelve:
      - lista de líneas transformadas (. y ?)
      - conteo de errores
      - pid del proceso (equivalente al núcleo)
      - cantidad de filas procesadas
    """
    chunk, = args  # desempaquetar tupla de un elemento
    transformed_lines = []
    errors = 0
    pid = os.getpid()

    for _, line in chunk:
        transformed = "".join("." if c in VALID else "?" for c in line)
        errors += transformed.count("?")
        transformed_lines.append(transformed)

    return transformed_lines, errors, pid, len(chunk)

# ─── Monitor de núcleos ──────────────────────────────────────────────────────
def gpu_process_chunk(chunk):
    """Process a chunk of lines on the GPU using Numba CUDA.
    Returns a list of transformed lines and the total error count.
    """
    # Prepare data: separate rows and lines
    rows, lines = zip(*chunk)
    max_len = max(len(line) for line in lines)
    # Create a 2D uint8 array padded with spaces (ASCII 32)
    host_array = np.full((len(lines), max_len), 32, dtype=np.uint8)
    for i, line in enumerate(lines):
        encoded = line.encode('utf-8')
        host_array[i, :len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)

    # Transfer to device
    d_chars = cuda.to_device(host_array)
    d_errors = cuda.to_device(np.zeros(1, dtype=np.int32))

    # Prepare valid character codes
    valid_codes = np.array([ord(c) for c in VALID], dtype=np.uint8)
    d_valid = cuda.to_device(valid_codes)

    threadsperblock = (16, 16)
    blockspergrid_x = (len(lines) + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (max_len + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    @cuda.jit
    def kernel(chars, valid_set, errors):
        i, j = cuda.grid(2)
        if i < chars.shape[0] and j < chars.shape[1]:
            c = chars[i, j]
            # Skip padding spaces
            if c == 32:
                return
            # Check if character is valid
            is_valid = False
            for k in range(valid_set.size):
                if c == valid_set[k]:
                    is_valid = True
                    break
            if not is_valid:
                chars[i, j] = ord('?')
                cuda.atomic.add(errors, 0, 1)

    kernel[blockspergrid, threadsperblock](d_chars, d_valid, d_errors)
    # Copy back results
    result_array = d_chars.copy_to_host()
    error_count = int(d_errors.copy_to_host()[0])

    # Reconstruct lines
    transformed = []
    for i in range(result_array.shape[0]):
        line_bytes = result_array[i]
        line_str = bytes(line_bytes).decode('utf-8', errors='ignore').rstrip()
        transformed.append(line_str)
    return transformed, error_count

def print_core_monitor(core_load: dict, cores: int, elapsed: float, rows_done: int):
    total = sum(core_load.values()) or 1
    print(f"\n{CYAN}{BOLD}══════════ Monitor de núcleos ══════════{RESET}")
    print(f"  Tiempo: {elapsed:.1f}s   Filas totales procesadas: {rows_done}")
    print(f"{CYAN}────────────────────────────────────────{RESET}")
    for i, (pid, count) in enumerate(sorted(core_load.items())):
        pct  = count * 100 // total
        bar  = "█" * (pct // 5)
        print(f"  Núcleo {i:>2} (pid {pid})  [{GREEN}{bar:<20}{RESET}] {pct:>3}%  ({count} filas)")
    print(f"{CYAN}════════════════════════════════════════{RESET}")

# ─── Hilo de monitor en tiempo real ─────────────────────────────────────────

def monitor_thread(core_load_ref: dict, lock: threading.Lock,
                   cores: int, start: float, stop_event: threading.Event,
                   rows_done_ref: list, pb):
    while not stop_event.is_set():
        time.sleep(0.5)
        with lock:
            load_copy = dict(core_load_ref)
            rows      = rows_done_ref[0]
        elapsed = time.time() - start
        pb.clear()
        print_core_monitor(load_copy, cores, elapsed, rows)

# ─── Contar líneas sin cargar el archivo ─────────────────────────────────────

def count_lines(path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count

# ─── Main ────────────────────────────────────────────────────────────────────

def ask_mode():
    """Muestra menú interactivo para elegir CPU o GPU."""
    available_cores = mp.cpu_count()

    print(f"\n{CYAN}{BOLD}┌──────────────────────────────────────┐{RESET}")
    print(f"{CYAN}{BOLD}│    Selecciona el modo de ejecución   │{RESET}")
    print(f"{CYAN}{BOLD}├──────────────────────────────────────┤{RESET}")
    print(f"{CYAN}{BOLD}│{RESET}                                      {CYAN}{BOLD}│{RESET}")
    print(f"{CYAN}{BOLD}│{RESET}  {GREEN}{BOLD}1){RESET} CPU  (multiprocessing)            {CYAN}{BOLD}│{RESET}")
    print(f"{CYAN}{BOLD}│{RESET}     └─ {available_cores} núcleos disponibles        {CYAN}{BOLD}│{RESET}")
    print(f"{CYAN}{BOLD}│{RESET}                                      {CYAN}{BOLD}│{RESET}")

    if GPU_AVAILABLE:
        print(f"{CYAN}{BOLD}│{RESET}  {YELLOW}{BOLD}2){RESET} GPU  (Numba CUDA)                {CYAN}{BOLD}│{RESET}")
        print(f"{CYAN}{BOLD}│{RESET}     └─ {GREEN}{GPU_NAME}{RESET}  {CYAN}{BOLD}│{RESET}")
    else:
        print(f"{CYAN}{BOLD}│{RESET}  {RED}2) GPU  (no disponible){RESET}             {CYAN}{BOLD}│{RESET}")
        print(f"{CYAN}{BOLD}│{RESET}     └─ {RED}{GPU_NAME}{RESET}  {CYAN}{BOLD}│{RESET}")

    print(f"{CYAN}{BOLD}│{RESET}                                      {CYAN}{BOLD}│{RESET}")
    print(f"{CYAN}{BOLD}└──────────────────────────────────────┘{RESET}")

    while True:
        try:
            choice = input(f"\n  {BRIGHT_BLUE}Elige (1/2):{RESET} ").strip()
            if choice == "1":
                return False  # use_gpu = False
            elif choice == "2":
                if not GPU_AVAILABLE:
                    print(f"  {RED}{BOLD}✘ GPU no disponible: {GPU_NAME}{RESET}")
                    print(f"  {YELLOW}Instala Numba y CUDA Toolkit, o selecciona CPU.{RESET}")
                    continue
                return True  # use_gpu = True
            else:
                print(f"  {RED}Opción inválida. Escribe 1 o 2.{RESET}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {YELLOW}Cancelado.{RESET}")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        prog        = "dna_checker",
        description = "Detector paralelo de nucleótidos inválidos en secuencias de ADN"
    )
    parser.add_argument("-i", "--input",      required=True,  type=Path, help="Archivo .txt o .fna")
    parser.add_argument("-j", "--cores",      default=0,      type=int,  help="Núcleos a usar (0 = todos, solo CPU)")
    parser.add_argument("--chunk-size",       default=50_000, type=int,  help="Líneas por chunk (default: 50000)")
    parser.add_argument("--no-skip-headers",  action="store_true",       help="No omitir líneas FASTA con >")
    args = parser.parse_args()

    # ── Banner ─────────────────────────────────────────────────────────────
    print(f"\n{BRIGHT_BLUE}{BOLD}╔══════════════════════════════════════╗{RESET}")
    print(f"{BRIGHT_BLUE}{BOLD}║      DNA Parallel Checker  v1.0      ║{RESET}")
    print(f"{BRIGHT_BLUE}{BOLD}╚══════════════════════════════════════╝{RESET}\n")

    # ── Preguntar CPU o GPU ────────────────────────────────────────────────
    use_gpu = ask_mode()

    # ── Número de núcleos (relevante solo para CPU) ────────────────────────
    available = mp.cpu_count()
    cores     = available if args.cores == 0 or args.cores > available else args.cores

    print(f"\n  Archivo    : {args.input}")
    if use_gpu:
        print(f"  Modo       : {YELLOW}{BOLD}GPU{RESET} — {GPU_NAME}")
    else:
        print(f"  Modo       : {GREEN}{BOLD}CPU{RESET} — {cores}/{available} núcleos")
    print(f"  Chunk size : {args.chunk_size:,} líneas\n")

    if not args.input.exists():
        print(f"{RED}{BOLD}Error: archivo no encontrado: {args.input}{RESET}")
        sys.exit(1)

    # ── Contar líneas para la barra ────────────────────────────────────────
    print("  Contando líneas...", end=" ", flush=True)
    total_lines = count_lines(args.input)
    print(f"{total_lines:,} líneas encontradas\n")

    # ── Archivo de salida ──────────────────────────────────────────────────
    out_path = args.input.parent / f"{args.input.stem}_errors.txt"

    # ── Estructuras compartidas para el monitor ────────────────────────────
    core_load   = {}          # pid → filas procesadas
    load_lock   = threading.Lock()
    rows_done   = [0]         # lista mutable para pasar por referencia al hilo
    stop_event  = threading.Event()

    # ── Barra de progreso ──────────────────────────────────────────────────
    pb = tqdm(
        total  = total_lines,
        desc   = "  Procesando",
        unit   = " filas",
        ncols  = 70,
        colour = "cyan",
        bar_format = "  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    # ── Hilo monitor ───────────────────────────────────────────────────────
    mon = threading.Thread(
        target = monitor_thread,
        args   = (core_load, load_lock, cores,
                  time.time(), stop_event, rows_done, pb),
        daemon = True
    )
    mon.start()

    # ── Procesamiento streaming por chunks ─────────────────────────────────
    total_errors = 0
    start_time   = time.time()

    skip_headers = not args.no_skip_headers

    with open(args.input, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as fout, \
         mp.Pool(processes=cores) as pool:

        chunk      = []
        global_row = 0

        def flush_chunk(chunk):
            """Envía el chunk al pool o GPU y escribe los resultados."""
            nonlocal total_errors

            if use_gpu:
                # GPU processing (single batch)
                transformed_lines, errors = gpu_process_chunk(chunk)
                for line in transformed_lines:
                    fout.write(line + "\n")
                total_errors += errors
                # Update monitor with approximate counts
                with load_lock:
                    # Assuming one GPU device, use pid 0
                    core_load[0] = core_load.get(0, 0) + len(chunk)
                    rows_done[0]  += len(chunk)
                fout.flush()
                pb.update(len(chunk))
            else:
                # Dividir el chunk entre los núcleos disponibles
                sub_size   = max(1, len(chunk) // cores)
                sub_chunks = [
                    chunk[i : i + sub_size]
                    for i in range(0, len(chunk), sub_size)
                ]

                # Procesar en paralelo con multiprocessing
                job_results = pool.map(process_chunk, [(sc,) for sc in sub_chunks])

                # Escribir resultados en orden y actualizar monitor
                for transformed_lines, errors, pid, count in job_results:
                    for line in transformed_lines:
                        fout.write(line + "\n")
                    total_errors += errors
                    with load_lock:
                        core_load[pid] = core_load.get(pid, 0) + count
                        rows_done[0]  += count

                fout.flush()
                pb.update(len(chunk))

        for raw_line in fin:
            global_row += 1
            line = raw_line.rstrip("\n\r")

            # Saltar cabeceras FASTA y líneas vacías
            if skip_headers and line.lstrip().startswith(">"):
                pb.update(1)
                continue
            if not line.strip():
                pb.update(1)
                continue

            chunk.append((global_row, line))

            if len(chunk) >= args.chunk_size:
                flush_chunk(chunk)
                chunk.clear()

        # Último chunk
        if chunk:
            flush_chunk(chunk)

    # ── Cerrar monitor y barra ─────────────────────────────────────────────
    stop_event.set()
    pb.close()
    mon.join(timeout=1)

    # Monitor final
    print_core_monitor(core_load, cores, time.time() - start_time, rows_done[0])

    # ── Resumen ────────────────────────────────────────────────────────────
    if total_errors == 0:
        print(f"\n  {GREEN}{BOLD}✔ No se encontraron errores.{RESET}\n")
    else:
        print(
            f"\n  {YELLOW}{BOLD}⚠ {total_errors:,} errores exportados a: "
            f"{CYAN}{out_path}{RESET}\n"
        )


if __name__ == "__main__":
    main()