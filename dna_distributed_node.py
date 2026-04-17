"""
DNA Distributed Worker Node v2.0
================================
Nodo worker que se conecta al coordinador para procesar
chunks de comparación de ADN de forma distribuida.
Usa GPU (OpenCL) para procesamiento y CPU para lectura.
Incluye failover: puede convertirse en coordinador si el maestro cae.

Uso:
    python dna_distributed_node.py --coordinator <IP>:<PORT>
    python dna_distributed_node.py --coordinator 192.168.1.100:5555
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import zmq

# ─── GPU detection via OpenCL ──────────────────────────────────────────────

GPU_AVAILABLE = False
GPU_NAME = "No disponible"
GPU_DRIVER = "N/A"
GPU_COMPUTE_UNITS = 0
GPU_MAX_WORK_GROUP = 0
GPU_GLOBAL_MEM = 0
GPU_LOCAL_MEM = 0
GPU_DEVICE_TYPE = "N/A"
GPU_PLATFORM_NAME = "N/A"
_cl_context = None
_cl_queue = None
_cl_program = None
_gpu_device = None

try:
    import pyopencl as cl
    import numpy as np

    platforms = cl.get_platforms()

    for plat in platforms:
        for device in plat.get_devices():
            if device.type & cl.device_type.GPU:
                _gpu_device = device
                GPU_PLATFORM_NAME = plat.name
                break
        if _gpu_device:
            break

    if _gpu_device is None:
        for plat in platforms:
            for device in plat.get_devices():
                if device.type & cl.device_type.ACCELERATOR:
                    _gpu_device = device
                    GPU_PLATFORM_NAME = plat.name
                    break
            if _gpu_device:
                break

    if _gpu_device:
        GPU_AVAILABLE = True
        GPU_NAME = _gpu_device.name.strip()
        GPU_DRIVER = _gpu_device.driver_version
        GPU_COMPUTE_UNITS = _gpu_device.max_compute_units
        GPU_MAX_WORK_GROUP = _gpu_device.max_work_group_size
        GPU_GLOBAL_MEM = _gpu_device.global_mem_size
        GPU_LOCAL_MEM = _gpu_device.local_mem_size
        GPU_DEVICE_TYPE = cl.device_type.to_string(_gpu_device.type)

        _cl_context = cl.Context([_gpu_device])
        _cl_queue = cl.CommandQueue(
            _cl_context, _gpu_device,
            properties=cl.command_queue_properties.PROFILING_ENABLE,
        )

        _kernel_source = """
        __kernel void dna_compare_2d(
            __global const uchar* seq_a,
            __global const uchar* seq_b,
            const int line_width,
            __global const int* lengths_a,
            __global const int* lengths_b,
            __global int* match_count,
            __global int* compare_count
        ) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            int min_len = lengths_a[row] < lengths_b[row]
                        ? lengths_a[row] : lengths_b[row];
            if (col >= min_len) return;
            int idx = row * line_width + col;
            atomic_add(compare_count, 1);
            uchar a = seq_a[idx];
            uchar b = seq_b[idx];
            if (a >= 97 && a <= 122) a -= 32;
            if (b >= 97 && b <= 122) b -= 32;
            if (a == b) { atomic_add(match_count, 1); }
        }

        __kernel void dna_check_2d(
            __global uchar* chars,
            __global const uchar* valid_set,
            const int num_valid,
            const int line_width,
            __global const int* lengths,
            __global int* error_count
        ) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            int num_rows = get_global_size(0);
            if (row >= num_rows) return;
            if (col >= lengths[row]) return;
            int idx = row * line_width + col;
            uchar c = chars[idx];
            int is_valid = 0;
            for (int k = 0; k < num_valid; k++) {
                if (c == valid_set[k]) { is_valid = 1; break; }
            }
            if (!is_valid) {
                chars[idx] = 63;
                atomic_add(error_count, 1);
            }
        }
        """
        _cl_program = cl.Program(_cl_context, _kernel_source).build()

except ImportError:
    GPU_NAME = "PyOpenCL no instalado"
    try:
        import numpy as np
    except ImportError:
        pass
except Exception as e:
    GPU_NAME = f"Error: {e}"
    try:
        import numpy as np
    except ImportError:
        pass

# ─── Constants ──────────────────────────────────────────────────────────────

VALID = frozenset("ATCGNatcgn")
VALID_ARRAY = np.array([ord(c) for c in VALID], dtype=np.uint8) if 'np' in dir() else None
HEARTBEAT_INTERVAL = 2.0   # seconds between heartbeats
COORDINATOR_TIMEOUT = 10.0  # seconds before assuming coordinator is dead
ELECTION_TIMEOUT = 5.0      # seconds to wait for election response
FAILOVER_STATE_FILE = Path(__file__).parent / ".failover_state.json"

# ─── GPU Processing ────────────────────────────────────────────────────────

def gpu_compare_chunk(lines_a: list, lines_b: list,
                      work_group_size: int = 64,
                      compute_units_to_use: int = 0,
                      cpu_cores: int = 1) -> dict:
    """
    Compare two lists of DNA lines on GPU using OpenCL.
    If compute_units_to_use < GPU_COMPUTE_UNITS, limits global_work_size
    proportionally to simulate using fewer CUs.
    Returns dict with matches, compared, line_details, gpu_metrics.
    """
    if not GPU_AVAILABLE or _cl_context is None:
        return cpu_compare_chunk(lines_a, lines_b, cpu_cores=cpu_cores)

    num_lines = len(lines_a)
    max_len = max(
        max((len(l) for l in lines_a), default=1),
        max((len(l) for l in lines_b), default=1),
    )

    arr_a = np.zeros(num_lines * max_len, dtype=np.uint8)
    arr_b = np.zeros(num_lines * max_len, dtype=np.uint8)
    lens_a = np.zeros(num_lines, dtype=np.int32)
    lens_b = np.zeros(num_lines, dtype=np.int32)

    for i in range(num_lines):
        ea = lines_a[i].encode("utf-8")
        eb = lines_b[i].encode("utf-8")
        lens_a[i] = len(ea)
        lens_b[i] = len(eb)
        off = i * max_len
        arr_a[off:off + len(ea)] = np.frombuffer(ea, dtype=np.uint8)
        arr_b[off:off + len(eb)] = np.frombuffer(eb, dtype=np.uint8)

    match_host = np.zeros(1, dtype=np.int32)
    compare_host = np.zeros(1, dtype=np.int32)

    # H2D
    t_h2d_start = time.perf_counter()
    mf = cl.mem_flags
    d_a = cl.Buffer(_cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
    d_b = cl.Buffer(_cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_b)
    d_la = cl.Buffer(_cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lens_a)
    d_lb = cl.Buffer(_cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lens_b)
    d_match = cl.Buffer(_cl_context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=match_host)
    d_compare = cl.Buffer(_cl_context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=compare_host)
    _cl_queue.finish()
    t_h2d = time.perf_counter() - t_h2d_start

    # Kernel 2D — limit global_work_size based on CU config
    wg_cols = min(work_group_size, GPU_MAX_WORK_GROUP, max_len)
    wg_rows = max(1, min(GPU_MAX_WORK_GROUP // wg_cols, num_lines))
    global_rows = ((num_lines + wg_rows - 1) // wg_rows) * wg_rows
    global_cols = ((max_len + wg_cols - 1) // wg_cols) * wg_cols

    # CU limiting: scale down global rows to use fewer compute units
    if 0 < compute_units_to_use < GPU_COMPUTE_UNITS:
        cu_ratio = compute_units_to_use / GPU_COMPUTE_UNITS
        limited_rows = max(wg_rows, int(global_rows * cu_ratio))
        limited_rows = ((limited_rows + wg_rows - 1) // wg_rows) * wg_rows
        effective_global_rows = limited_rows
    else:
        effective_global_rows = global_rows

    total_matches = 0
    total_compared = 0
    t_kernel_total = 0

    # Process in batches if CU-limited (more rows than effective global)
    for batch_start in range(0, num_lines, effective_global_rows):
        batch_end = min(batch_start + effective_global_rows, num_lines)
        batch_count = batch_end - batch_start
        batch_global_rows = ((batch_count + wg_rows - 1) // wg_rows) * wg_rows

        # Create sub-buffers for this batch
        batch_arr_a = arr_a[batch_start * max_len : (batch_start + batch_count) * max_len]
        batch_arr_b = arr_b[batch_start * max_len : (batch_start + batch_count) * max_len]
        batch_lens_a = lens_a[batch_start:batch_end]
        batch_lens_b = lens_b[batch_start:batch_end]

        # Zero out match/compare for this batch
        batch_match = np.zeros(1, dtype=np.int32)
        batch_compare = np.zeros(1, dtype=np.int32)

        d_ba = cl.Buffer(_cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=batch_arr_a)
        d_bb = cl.Buffer(_cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=batch_arr_b)
        d_bla = cl.Buffer(_cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=batch_lens_a)
        d_blb = cl.Buffer(_cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=batch_lens_b)
        d_bm = cl.Buffer(_cl_context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=batch_match)
        d_bc = cl.Buffer(_cl_context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=batch_compare)

        t_kernel_start = time.perf_counter()
        _cl_program.dna_compare_2d(
            _cl_queue,
            (batch_global_rows, global_cols),
            (wg_rows, wg_cols),
            d_ba, d_bb,
            np.int32(max_len),
            d_bla, d_blb,
            d_bm, d_bc,
        )
        _cl_queue.finish()
        t_kernel_total += time.perf_counter() - t_kernel_start

        cl.enqueue_copy(_cl_queue, batch_match, d_bm)
        cl.enqueue_copy(_cl_queue, batch_compare, d_bc)
        _cl_queue.finish()

        total_matches += int(batch_match[0])
        total_compared += int(batch_compare[0])

    # D2H already done in loop
    t_d2h_start = time.perf_counter()
    t_d2h = time.perf_counter() - t_d2h_start

    # Per-line details (CPU, first 50 lines only)
    line_details = []
    for i in range(min(num_lines, 50)):
        la, lb = lines_a[i], lines_b[i]
        min_len_l = min(len(la), len(lb))
        if min_len_l > 0:
            lm = sum(1 for x, y in zip(la.upper(), lb.upper()) if x == y)
            line_details.append({
                "line_idx": i,
                "len_a": len(la),
                "len_b": len(lb),
                "matches": lm,
                "compared": min_len_l,
                "similarity": round(lm / min_len_l * 100, 2),
            })

    cu_used = compute_units_to_use if 0 < compute_units_to_use < GPU_COMPUTE_UNITS else GPU_COMPUTE_UNITS
    gpu_metrics = {
        "mode": "GPU",
        "gpu_name": GPU_NAME,
        "transfer_h2d_ms": round(t_h2d * 1000, 3),
        "kernel_time_ms": round(t_kernel_total * 1000, 3),
        "transfer_d2h_ms": round(t_d2h * 1000, 3),
        "total_gpu_time_ms": round((t_h2d + t_kernel_total + t_d2h) * 1000, 3),
        "global_work_size": effective_global_rows * global_cols,
        "grid_shape": f"{effective_global_rows}×{global_cols}",
        "work_group_shape": f"{wg_rows}×{wg_cols}",
        "work_group_size": wg_rows * wg_cols,
        "compute_units_used": cu_used,
    }

    return {
        "matches": total_matches,
        "compared": total_compared,
        "line_details": line_details,
        "gpu_metrics": gpu_metrics,
    }


def gpu_validate_chunk(lines: list, row_numbers: list,
                       work_group_size: int = 64,
                       compute_units_to_use: int = 0,
                       cpu_cores: int = 1) -> dict:
    """
    Validate DNA lines on GPU — find invalid characters (not ACGTN).
    Returns dict with total_errors, error_details [{row, col, char}], gpu_metrics.
    """
    if not GPU_AVAILABLE or _cl_context is None or VALID_ARRAY is None:
        return cpu_validate_chunk(lines, row_numbers, cpu_cores=cpu_cores)

    num_lines = len(lines)
    max_len = max(len(l) for l in lines) if lines else 1

    host_array = np.zeros(num_lines * max_len, dtype=np.uint8)
    line_lengths = np.zeros(num_lines, dtype=np.int32)

    for i, line in enumerate(lines):
        encoded = line.encode('utf-8')
        line_lengths[i] = len(encoded)
        offset = i * max_len
        host_array[offset:offset + len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)

    error_count_host = np.zeros(1, dtype=np.int32)

    t_h2d_start = time.perf_counter()
    mf = cl.mem_flags
    d_chars = cl.Buffer(_cl_context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_array)
    d_valid = cl.Buffer(_cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=VALID_ARRAY)
    d_lengths = cl.Buffer(_cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=line_lengths)
    d_errors = cl.Buffer(_cl_context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=error_count_host)
    _cl_queue.finish()
    t_h2d = time.perf_counter() - t_h2d_start

    wg_cols = min(work_group_size, GPU_MAX_WORK_GROUP, max_len)
    wg_rows = max(1, min(GPU_MAX_WORK_GROUP // wg_cols, num_lines))
    global_rows = ((num_lines + wg_rows - 1) // wg_rows) * wg_rows
    global_cols = ((max_len + wg_cols - 1) // wg_cols) * wg_cols

    t_kernel_start = time.perf_counter()
    _cl_program.dna_check_2d(
        _cl_queue,
        (global_rows, global_cols),
        (wg_rows, wg_cols),
        d_chars, d_valid,
        np.int32(len(VALID_ARRAY)),
        np.int32(max_len),
        d_lengths, d_errors,
    )
    _cl_queue.finish()
    t_kernel = time.perf_counter() - t_kernel_start

    t_d2h_start = time.perf_counter()
    cl.enqueue_copy(_cl_queue, host_array, d_chars)
    cl.enqueue_copy(_cl_queue, error_count_host, d_errors)
    _cl_queue.finish()
    t_d2h = time.perf_counter() - t_d2h_start

    error_count = int(error_count_host[0])

    # Extract error positions
    error_details = []
    if error_count > 0:
        for i in range(num_lines):
            if len(error_details) >= 500:
                break
            offset = i * max_len
            for j in range(line_lengths[i]):
                if host_array[offset + j] == 63:  # '?'
                    error_details.append({
                        "row": row_numbers[i],
                        "col": j + 1,
                        "char": lines[i][j] if j < len(lines[i]) else "?",
                    })
                    if len(error_details) >= 500:
                        break

    gpu_metrics = {
        "mode": "GPU",
        "gpu_name": GPU_NAME,
        "transfer_h2d_ms": round(t_h2d * 1000, 3),
        "kernel_time_ms": round(t_kernel * 1000, 3),
        "transfer_d2h_ms": round(t_d2h * 1000, 3),
        "total_gpu_time_ms": round((t_h2d + t_kernel + t_d2h) * 1000, 3),
        "global_work_size": global_rows * global_cols,
        "grid_shape": f"{global_rows}×{global_cols}",
        "work_group_shape": f"{wg_rows}×{wg_cols}",
    }

    return {
        "total_errors": error_count,
        "error_details": error_details,
        "lines_processed": num_lines,
        "gpu_metrics": gpu_metrics,
    }


def _validate_worker(args):
    """Worker function for parallel CPU validation (must be top-level for pickle)."""
    lines, row_numbers, valid_set = args
    errors = []
    for i, line in enumerate(lines):
        for j, c in enumerate(line):
            if c not in valid_set:
                errors.append({
                    "row": row_numbers[i],
                    "col": j + 1,
                    "char": c,
                })
    return errors


def cpu_validate_chunk(lines: list, row_numbers: list,
                       cpu_cores: int = 1) -> dict:
    """
    Fallback: Validate DNA lines on CPU — find invalid characters.
    Uses multiprocessing with cpu_cores workers for parallelism.
    """
    if cpu_cores <= 1 or len(lines) < 100:
        # Single-core path
        total_errors = 0
        error_details = []
        for i, line in enumerate(lines):
            for j, c in enumerate(line):
                if c not in VALID:
                    total_errors += 1
                    if len(error_details) < 500:
                        error_details.append({
                            "row": row_numbers[i],
                            "col": j + 1,
                            "char": c,
                        })
        return {
            "total_errors": total_errors,
            "error_details": error_details,
            "lines_processed": len(lines),
            "gpu_metrics": {"mode": "CPU", "cores": 1},
        }

    # Multi-core path
    import multiprocessing as mp_pool
    n = len(lines)
    chunk_size = max(1, n // cpu_cores)
    tasks = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        tasks.append((lines[i:end], row_numbers[i:end], VALID))

    all_errors = []
    try:
        with mp_pool.Pool(processes=min(cpu_cores, len(tasks))) as pool:
            results = pool.map(_validate_worker, tasks)
        for errs in results:
            all_errors.extend(errs)
    except Exception:
        # Fallback to single-core if multiprocessing fails
        for task in tasks:
            all_errors.extend(_validate_worker(task))

    return {
        "total_errors": len(all_errors),
        "error_details": all_errors[:500],
        "lines_processed": len(lines),
        "gpu_metrics": {"mode": "CPU", "cores": cpu_cores},
    }


def _compare_worker(args):
    """Worker function for parallel CPU comparison (must be top-level for pickle)."""
    lines_a, lines_b = args
    matches = 0
    compared = 0
    for la, lb in zip(lines_a, lines_b):
        min_len = min(len(la), len(lb))
        for j in range(min_len):
            compared += 1
            if la[j].upper() == lb[j].upper():
                matches += 1
    return matches, compared


def cpu_compare_chunk(lines_a: list, lines_b: list,
                      cpu_cores: int = 1) -> dict:
    """
    Fallback: Compare two lists of DNA lines on CPU.
    Uses multiprocessing with cpu_cores workers for parallelism.
    """
    if cpu_cores <= 1 or len(lines_a) < 100:
        # Single-core path
        matches = 0
        compared = 0
        line_details = []
        for i, (la, lb) in enumerate(zip(lines_a, lines_b)):
            min_len = min(len(la), len(lb))
            line_matches = 0
            for j in range(min_len):
                compared += 1
                if la[j].upper() == lb[j].upper():
                    matches += 1
                    line_matches += 1
            if min_len > 0 and len(line_details) < 50:
                line_details.append({
                    "line_idx": i,
                    "len_a": len(la),
                    "len_b": len(lb),
                    "matches": line_matches,
                    "compared": min_len,
                    "similarity": round(line_matches / min_len * 100, 2),
                })
        return {
            "matches": matches,
            "compared": compared,
            "line_details": line_details,
            "gpu_metrics": {"mode": "CPU", "cores": 1},
        }

    # Multi-core path
    import multiprocessing as mp_pool
    n = min(len(lines_a), len(lines_b))
    chunk_size = max(1, n // cpu_cores)
    tasks = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        tasks.append((lines_a[i:end], lines_b[i:end]))

    total_matches = 0
    total_compared = 0
    try:
        with mp_pool.Pool(processes=min(cpu_cores, len(tasks))) as pool:
            results = pool.map(_compare_worker, tasks)
        for m, c in results:
            total_matches += m
            total_compared += c
    except Exception:
        for task in tasks:
            m, c = _compare_worker(task)
            total_matches += m
            total_compared += c

    return {
        "matches": total_matches,
        "compared": total_compared,
        "line_details": [],
        "gpu_metrics": {"mode": "CPU", "cores": cpu_cores},
    }


# ─── Worker Node Class ────────────────────────────────────────────────────

class WorkerNode:
    def __init__(self, coordinator_addr: str, node_name: str = None):
        self.node_id = str(uuid.uuid4())[:8]
        self.node_name = node_name or f"worker-{self.node_id}"
        self.coordinator_addr = coordinator_addr
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.cpu_count = os.cpu_count() or 1

        self.context = zmq.Context()
        self.running = False
        self.connected = False
        self.processing = False
        self.enabled = True           # Can be toggled by coordinator
        self.current_chunk_id = None
        self._dealer_lock = threading.Lock()  # ZMQ sockets are NOT thread-safe

        # GPU config (can be updated by coordinator)
        self.gpu_work_group_size = min(64, GPU_MAX_WORK_GROUP) if GPU_MAX_WORK_GROUP > 0 else 64
        self.gpu_compute_units_to_use = GPU_COMPUTE_UNITS

        # CPU config (can be updated by coordinator)
        self.cpu_cores_to_use = self.cpu_count  # use all by default

        # For leader election
        self.known_peers = {}
        self.is_coordinator = False
        self.coordinator_last_seen = 0
        self.last_coordinator_state = {}  # Saved state for failover
        self.election_in_progress = False

        # Stats
        self.chunks_processed = 0
        self.total_lines_processed = 0
        self.total_matches = 0
        self.total_compared = 0

        print(f"\n{'='*60}")
        print(f"  🧬 DNA Distributed Worker Node v2.0")
        print(f"  ID:          {self.node_id}")
        print(f"  Nombre:      {self.node_name}")
        print(f"  Host:        {self.hostname}")
        print(f"  PID:         {self.pid}")
        print(f"  CPUs:        {self.cpu_count}")
        print(f"  GPU:         {'✅ ' + GPU_NAME if GPU_AVAILABLE else '❌ ' + GPU_NAME}")
        if GPU_AVAILABLE:
            print(f"  GPU CUs:     {GPU_COMPUTE_UNITS}")
            print(f"  GPU Mem:     {GPU_GLOBAL_MEM / (1024**2):.0f} MB")
            print(f"  Max WG:      {GPU_MAX_WORK_GROUP}")
        print(f"  Coordinador: {coordinator_addr}")
        print(f"{'='*60}\n")

    def _get_gpu_info(self) -> dict:
        """Get GPU info dict for registration."""
        return {
            "available": GPU_AVAILABLE,
            "name": GPU_NAME,
            "driver": GPU_DRIVER,
            "platform": GPU_PLATFORM_NAME,
            "device_type": GPU_DEVICE_TYPE,
            "compute_units": GPU_COMPUTE_UNITS,
            "max_work_group_size": GPU_MAX_WORK_GROUP,
            "global_memory": GPU_GLOBAL_MEM,
            "local_memory": GPU_LOCAL_MEM,
        }

    def start(self):
        """Start the worker node."""
        self.running = True

        # DEALER socket for work requests/responses
        self.dealer = self.context.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.IDENTITY, self.node_id)
        self.dealer.setsockopt(zmq.RECONNECT_IVL, 1000)
        self.dealer.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)
        self.dealer.setsockopt(zmq.LINGER, 0)

        # SUB socket for broadcasts from coordinator
        self.sub = self.context.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.setsockopt(zmq.RECONNECT_IVL, 1000)
        self.sub.setsockopt(zmq.LINGER, 0)

        try:
            coord_host, coord_port = self.coordinator_addr.rsplit(":", 1)
            coord_port = int(coord_port)

            self.dealer.connect(f"tcp://{coord_host}:{coord_port}")
            self.sub.connect(f"tcp://{coord_host}:{coord_port + 1}")

            print(f"  ✅ Conectado al coordinador en {self.coordinator_addr}")
            self.connected = True
            self.coordinator_last_seen = time.time()

        except Exception as e:
            print(f"  ❌ Error conectando: {e}")
            return

        # Start threads
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()

        self.broadcast_thread = threading.Thread(
            target=self._broadcast_listener, daemon=True
        )
        self.broadcast_thread.start()

        # Main work loop
        self._work_loop()

    def _send_message(self, msg_type: str, data: dict = None):
        """Send a message to the coordinator (thread-safe)."""
        message = {
            "type": msg_type,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "timestamp": time.time(),
            "data": data or {},
        }
        with self._dealer_lock:
            try:
                self.dealer.send_json(message, zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

    def _heartbeat_loop(self):
        """Send heartbeat to coordinator periodically."""
        while self.running:
            self._send_message("HEARTBEAT", {
                "hostname": self.hostname,
                "pid": self.pid,
                "cpu_count": self.cpu_count,
                "processing": self.processing,
                "current_chunk": self.current_chunk_id,
                "chunks_processed": self.chunks_processed,
                "total_lines": self.total_lines_processed,
                "total_matches": self.total_matches,
                "total_compared": self.total_compared,
                "gpu_info": self._get_gpu_info(),
                "enabled": self.enabled,
            })
            time.sleep(HEARTBEAT_INTERVAL)

    def _broadcast_listener(self):
        """Listen for broadcast messages from coordinator."""
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        while self.running:
            try:
                events = dict(poller.poll(1000))
                if self.sub in events:
                    msg = self.sub.recv_json()
                    self._handle_broadcast(msg)
            except zmq.ZMQError:
                time.sleep(0.1)
            except Exception as e:
                print(f"  ⚠ Broadcast error: {e}")
                time.sleep(0.1)

    def _handle_broadcast(self, msg: dict):
        """Handle broadcast messages."""
        msg_type = msg.get("type", "")

        if msg_type == "COORDINATOR_HEARTBEAT":
            self.coordinator_last_seen = time.time()
            self.election_in_progress = False
            # Save state for failover
            state = msg.get("state", {})
            if state:
                self.last_coordinator_state = state

        elif msg_type == "PEER_LIST":
            self.known_peers = msg.get("data", {}).get("peers", {})

        elif msg_type == "ELECTION":
            election_id = msg.get("data", {}).get("initiator", "")
            if self.node_id < election_id:
                self._send_message("ELECTION_RESPONSE", {
                    "to": election_id,
                    "from": self.node_id,
                })

        elif msg_type == "NEW_COORDINATOR":
            new_coord = msg.get("data", {})
            new_addr = new_coord.get("addr", "")
            new_id = new_coord.get("node_id", "")
            if new_id != self.node_id and new_addr:
                print(f"\n  🔄 Nuevo coordinador: {new_id} en {new_addr}")
                self._reconnect_to_coordinator(new_addr)
                self.election_in_progress = False

        elif msg_type == "SHUTDOWN":
            print("\n  ⚠ Coordinador se apagó. Iniciando elección de nuevo coordinador...")
            self.coordinator_last_seen = 0  # Force timeout
            if not self.election_in_progress:
                threading.Thread(
                    target=self._start_election, daemon=True
                ).start()

    def _reconnect_to_coordinator(self, new_addr: str):
        """Reconnect to a new coordinator."""
        try:
            old_host, old_port = self.coordinator_addr.rsplit(":", 1)
            self.dealer.disconnect(f"tcp://{old_host}:{old_port}")
            self.sub.disconnect(f"tcp://{old_host}:{int(old_port) + 1}")
        except zmq.ZMQError:
            pass

        self.coordinator_addr = new_addr
        coord_host, coord_port = new_addr.rsplit(":", 1)
        coord_port = int(coord_port)

        self.dealer.connect(f"tcp://{coord_host}:{coord_port}")
        self.sub.connect(f"tcp://{coord_host}:{coord_port + 1}")
        self.coordinator_last_seen = time.time()
        print(f"  ✅ Reconectado a {new_addr}")

        # Re-register
        self._send_message("REGISTER", {
            "hostname": self.hostname,
            "pid": self.pid,
            "cpu_count": self.cpu_count,
            "gpu_info": self._get_gpu_info(),
        })

    def _work_loop(self):
        """Main loop: receive and process chunks."""
        poller = zmq.Poller()
        poller.register(self.dealer, zmq.POLLIN)

        # Register with coordinator
        self._send_message("REGISTER", {
            "hostname": self.hostname,
            "pid": self.pid,
            "cpu_count": self.cpu_count,
            "gpu_info": self._get_gpu_info(),
        })
        print("  📡 Registrado con el coordinador. Esperando trabajo...")

        while self.running:
            try:
                events = dict(poller.poll(1000))

                if self.dealer in events:
                    with self._dealer_lock:
                        msg = self.dealer.recv_json()
                    self._handle_work(msg)

                # Check coordinator health
                if time.time() - self.coordinator_last_seen > COORDINATOR_TIMEOUT:
                    if not self.is_coordinator and not self.election_in_progress:
                        print("\n  ⚠ Coordinador no responde. Iniciando elección...")
                        self._start_election()

            except zmq.ZMQError:
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n  ⛔ Detenido por el usuario")
                break
            except Exception as e:
                import traceback
                print(f"  ❌ Error en work loop: {e}")
                traceback.print_exc()
                time.sleep(0.5)

        self.stop()

    def _handle_work(self, msg: dict):
        """Handle a work message from the coordinator."""
        msg_type = msg.get("type", "")

        if msg_type == "CHUNK_COMPARE":
            if self.enabled:
                self._process_compare_chunk(msg)
            else:
                data = msg.get("data", {})
                self._send_message("CHUNK_REJECTED", {
                    "chunk_id": data.get("chunk_id"),
                    "job_id": data.get("job_id"),
                    "reason": "Node disabled",
                })

        elif msg_type == "CHUNK_VALIDATE":
            if self.enabled:
                self._process_validate_chunk(msg)
            else:
                data = msg.get("data", {})
                self._send_message("CHUNK_REJECTED", {
                    "chunk_id": data.get("chunk_id"),
                    "job_id": data.get("job_id"),
                    "reason": "Node disabled",
                })

        elif msg_type == "PING":
            self._send_message("PONG", {"responding_to": msg.get("data", {}).get("ping_id")})

        elif msg_type == "COORDINATOR_HEARTBEAT":
            self.coordinator_last_seen = time.time()

        elif msg_type == "STATUS_CHANGE":
            data = msg.get("data", {})
            self.enabled = data.get("enabled", True)
            state = "ACTIVADO" if self.enabled else "DESACTIVADO"
            print(f"\n  {'✅' if self.enabled else '⏸️'} Nodo {state} por el coordinador")

        elif msg_type == "GPU_CONFIG":
            data = msg.get("data", {})
            wg = data.get("work_group_size", 0)
            cu = data.get("compute_units", 0)
            if wg > 0:
                self.gpu_work_group_size = min(wg, GPU_MAX_WORK_GROUP)
            if cu > 0:
                self.gpu_compute_units_to_use = min(cu, GPU_COMPUTE_UNITS)
            print(f"\n  ⚙️ GPU config actualizada: WG={self.gpu_work_group_size}, CUs={self.gpu_compute_units_to_use}")

        elif msg_type == "CPU_CONFIG":
            data = msg.get("data", {})
            cores = data.get("cpu_cores", 0)
            if 0 < cores <= self.cpu_count:
                self.cpu_cores_to_use = cores
            print(f"\n  ⚙️ CPU config actualizada: Cores={self.cpu_cores_to_use}/{self.cpu_count}")

        elif msg_type == "KILL":
            reason = msg.get("data", {}).get("reason", "Unknown")
            print(f"\n  ⛔ Eliminado por el coordinador: {reason}")
            self.running = False

    def _process_compare_chunk(self, msg: dict):
        """Process a DNA comparison chunk using GPU (or CPU fallback)."""
        data = msg.get("data", {})
        chunk_id = data.get("chunk_id", "unknown")
        job_id = data.get("job_id", "")
        lines_a = data.get("lines_a", [])
        lines_b = data.get("lines_b", [])
        chunk_index = data.get("chunk_index", 0)
        total_chunks = data.get("total_chunks", 1)

        # GPU config from coordinator or local settings
        gpu_config = data.get("gpu_config", {})
        wg_size = gpu_config.get("work_group_size", self.gpu_work_group_size)
        cu_use = gpu_config.get("compute_units", self.gpu_compute_units_to_use)

        self.processing = True
        self.current_chunk_id = chunk_id

        mode = "GPU" if GPU_AVAILABLE else f"CPU ({self.cpu_cores_to_use} cores)"
        print(f"\n  📥 Chunk {chunk_index+1}/{total_chunks} recibido "
              f"({len(lines_a)} líneas) — Procesando con {mode}")

        start = time.time()
        result = gpu_compare_chunk(lines_a, lines_b, wg_size, cu_use,
                                   cpu_cores=self.cpu_cores_to_use)
        elapsed = time.time() - start

        self.chunks_processed += 1
        self.total_lines_processed += len(lines_a)
        self.total_matches += result["matches"]
        self.total_compared += result["compared"]

        sim_pct = result['matches'] / max(result['compared'], 1) * 100
        print(f"  ✅ Chunk {chunk_index+1} procesado con {mode} en {elapsed:.2f}s — "
              f"Coincidencia: {result['matches']}/{result['compared']} "
              f"({sim_pct:.1f}%)")

        self._send_message("RESULT", {
            "chunk_id": chunk_id,
            "job_id": job_id,
            "chunk_index": chunk_index,
            "matches": result["matches"],
            "compared": result["compared"],
            "line_details": result["line_details"],
            "elapsed": round(elapsed, 3),
            "lines_processed": len(lines_a),
            "gpu_metrics": result.get("gpu_metrics", {}),
        })

        self.processing = False
        self.current_chunk_id = None

    def _process_validate_chunk(self, msg: dict):
        """Process a DNA validation chunk — find invalid characters."""
        data = msg.get("data", {})
        chunk_id = data.get("chunk_id", "unknown")
        job_id = data.get("job_id", "")
        lines = data.get("lines", [])
        row_numbers = data.get("row_numbers", [])
        chunk_index = data.get("chunk_index", 0)
        total_chunks = data.get("total_chunks", 1)

        gpu_config = data.get("gpu_config", {})
        wg_size = gpu_config.get("work_group_size", self.gpu_work_group_size)
        cu_use = gpu_config.get("compute_units", self.gpu_compute_units_to_use)

        self.processing = True
        self.current_chunk_id = chunk_id

        mode = "GPU" if GPU_AVAILABLE else f"CPU ({self.cpu_cores_to_use} cores)"
        print(f"\n  🔍 Validación chunk {chunk_index+1}/{total_chunks} "
              f"({len(lines)} líneas) — {mode}")

        start = time.time()
        result = gpu_validate_chunk(lines, row_numbers, wg_size, cu_use,
                                    cpu_cores=self.cpu_cores_to_use)
        elapsed = time.time() - start

        self.chunks_processed += 1
        self.total_lines_processed += result["lines_processed"]

        print(f"  ✅ Validación chunk {chunk_index+1}: "
              f"{result['total_errors']} errores en {elapsed:.2f}s ({mode})")

        self._send_message("VALIDATE_RESULT", {
            "chunk_id": chunk_id,
            "job_id": job_id,
            "chunk_index": chunk_index,
            "total_errors": result["total_errors"],
            "error_details": result["error_details"],
            "lines_processed": result["lines_processed"],
            "elapsed": round(elapsed, 3),
            "gpu_metrics": result.get("gpu_metrics", {}),
        })

        self.processing = False
        self.current_chunk_id = None

    def _start_election(self):
        """Start a leader election using Bully algorithm."""
        self.election_in_progress = True
        print(f"  🗳 Iniciando elección de líder (mi ID: {self.node_id})")

        # Broadcast election to all peers
        try:
            # Use the existing PUB/SUB channel indirectly through dealer
            self._send_message("ELECTION", {
                "initiator": self.node_id,
            })
        except Exception:
            pass

        # Wait for responses from higher-priority nodes
        time.sleep(ELECTION_TIMEOUT)

        if not self.running:
            return

        # Check if coordinator came back
        if time.time() - self.coordinator_last_seen < COORDINATOR_TIMEOUT:
            print(f"  ✅ Coordinador respondió — elección cancelada")
            self.election_in_progress = False
            return

        # No response: become the new coordinator
        print(f"\n  👑 ¡Este nodo es el nuevo coordinador!")
        self._become_coordinator()

    def _become_coordinator(self):
        """Transform this worker into the new coordinator.
        Saves the last coordinator state to a file so the new coordinator
        can restore active jobs and not lose processing.
        """
        self.is_coordinator = True
        self.election_in_progress = False

        # Get local IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"

        # Save saved coordinator state for failover
        if self.last_coordinator_state:
            try:
                state_to_save = {
                    "timestamp": time.time(),
                    "previous_coordinator_id": "unknown",
                    "state": self.last_coordinator_state,
                    "promoted_by": self.node_id,
                }
                with open(FAILOVER_STATE_FILE, "w") as f:
                    json.dump(state_to_save, f, indent=2)
                print(f"  💾 Estado del coordinador anterior guardado en {FAILOVER_STATE_FILE}")
            except Exception as e:
                print(f"  ⚠ No se pudo guardar estado de failover: {e}")

        # Start the coordinator process
        coord_port = 5557  # Use different port to avoid conflict
        web_port = 5001

        print(f"  🔄 Iniciando coordinador en {local_ip}:{coord_port}")
        print(f"  🌐 Web: http://{local_ip}:{web_port}")

        # Launch coordinator as subprocess with --restore-state flag
        try:
            script_dir = Path(__file__).parent
            coord_script = script_dir / "dna_distributed_coordinator.py"

            cmd = [
                sys.executable, str(coord_script),
                "--port", str(coord_port),
                "--web-port", str(web_port),
            ]
            if self.last_coordinator_state:
                cmd.extend(["--restore-state", str(FAILOVER_STATE_FILE)])

            subprocess.Popen(cmd, cwd=str(script_dir))

            print(f"  ✅ Nuevo coordinador lanzado (con restauración de estado)")

            # Give coordinator time to start
            time.sleep(2)

            # Reconnect this worker to the new coordinator
            new_addr = f"{local_ip}:{coord_port}"
            self._reconnect_to_coordinator(new_addr)

        except Exception as e:
            print(f"  ❌ Error al iniciar coordinador: {e}")
            self.is_coordinator = False

    def stop(self):
        """Stop the worker node."""
        self.running = False
        self._send_message("UNREGISTER", {})
        print(f"\n  🛑 Worker {self.node_id} detenido")
        try:
            self.dealer.close()
            self.sub.close()
            self.context.term()
        except Exception:
            pass


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="dna_distributed_node",
        description="Nodo worker para comparación distribuida de ADN v2.0 (GPU + Failover)",
    )
    parser.add_argument(
        "--coordinator", "-c",
        required=True,
        help="Dirección del coordinador (IP:PUERTO), ej: 192.168.1.100:5555",
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Nombre personalizado para este nodo",
    )
    args = parser.parse_args()

    node = WorkerNode(args.coordinator, args.name)

    def signal_handler(sig, frame):
        print("\n  🛑 Señal de interrupción recibida")
        node.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    node.start()


if __name__ == "__main__":
    main()