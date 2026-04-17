"""
DNA Unified Checker v4.0 — Web Interface (CPU + GPU)
Flask backend that lets the user choose between CPU (multiprocessing)
and GPU (OpenCL) for DNA analysis.
Streams real-time progress via Server-Sent Events.
"""

import json
import multiprocessing as mp
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify, Response

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

try:
    import pyopencl as cl

    platforms = cl.get_platforms()
    _gpu_device = None

    for platform in platforms:
        for device in platform.get_devices():
            if device.type & cl.device_type.GPU:
                _gpu_device = device
                GPU_PLATFORM_NAME = platform.name
                break
        if _gpu_device:
            break

    if _gpu_device is None:
        for platform in platforms:
            for device in platform.get_devices():
                if device.type & cl.device_type.ACCELERATOR:
                    _gpu_device = device
                    GPU_PLATFORM_NAME = platform.name
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
            properties=cl.command_queue_properties.PROFILING_ENABLE
        )

        _kernel_source = """
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
        """
        _cl_program = cl.Program(_cl_context, _kernel_source).build()

except ImportError:
    GPU_NAME = "PyOpenCL no instalado"
except Exception as e:
    GPU_NAME = f"Error: {e}"


app = Flask(__name__)

# ─── In-memory job store ────────────────────────────────────────────────────

jobs: dict = {}

# ─── DNA constants ─────────────────────────────────────────────────────────

VALID = frozenset("ATCGNatcgn")
VALID_ARRAY = np.array([ord(c) for c in VALID], dtype=np.uint8)


# ─── CPU Worker ────────────────────────────────────────────────────────────

def cpu_process_chunk(args):
    """Worker: transforms each character to '.' (valid) or '?' (invalid)."""
    (chunk,) = args
    transformed_lines = []
    error_details = []
    errors = 0
    pid = os.getpid()

    for row_num, line in chunk:
        transformed = []
        for col, c in enumerate(line):
            if c in VALID:
                transformed.append(".")
            else:
                transformed.append("?")
                errors += 1
                error_details.append(
                    {"row": row_num, "col": col + 1, "char": c}
                )
        transformed_lines.append("".join(transformed))

    return transformed_lines, errors, error_details, pid, len(chunk)


# ─── GPU Worker ────────────────────────────────────────────────────────────

def gpu_process_chunk(chunk, work_group_size=64):
    """Process a chunk of DNA lines on the GPU using OpenCL."""
    cl_ctx = _cl_context
    cl_queue_local = _cl_queue
    cl_program_local = _cl_program

    rows, lines = zip(*chunk)
    max_len = max(len(line) for line in lines)
    num_lines = len(lines)

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
    d_chars = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_array)
    d_valid = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=VALID_ARRAY)
    d_lengths = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=line_lengths)
    d_errors = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=error_count_host)
    cl_queue_local.finish()
    t_h2d = time.perf_counter() - t_h2d_start

    wg_cols = min(work_group_size, GPU_MAX_WORK_GROUP, max_len)
    max_total_wg = GPU_MAX_WORK_GROUP
    wg_rows = max(1, min(max_total_wg // wg_cols, num_lines))
    global_rows = ((num_lines + wg_rows - 1) // wg_rows) * wg_rows
    global_cols = ((max_len + wg_cols - 1) // wg_cols) * wg_cols

    t_kernel_start = time.perf_counter()
    cl_program_local.dna_check_2d(
        cl_queue_local,
        (global_rows, global_cols),
        (wg_rows, wg_cols),
        d_chars, d_valid,
        np.int32(len(VALID_ARRAY)),
        np.int32(max_len),
        d_lengths, d_errors
    )
    cl_queue_local.finish()
    t_kernel = time.perf_counter() - t_kernel_start

    t_d2h_start = time.perf_counter()
    cl.enqueue_copy(cl_queue_local, host_array, d_chars)
    cl.enqueue_copy(cl_queue_local, error_count_host, d_errors)
    cl_queue_local.finish()
    t_d2h = time.perf_counter() - t_d2h_start

    error_count = int(error_count_host[0])

    transformed = []
    for i in range(num_lines):
        offset = i * max_len
        length = line_lengths[i]
        line_bytes = host_array[offset:offset + length]
        transformed.append(bytes(line_bytes).decode('utf-8', errors='ignore'))

    error_details = []
    if error_count > 0:
        for i in range(num_lines):
            if len(error_details) >= 200:
                break
            offset = i * max_len
            for j in range(line_lengths[i]):
                if host_array[offset + j] == 63:
                    error_details.append({
                        "row": rows[i],
                        "col": j + 1,
                        "char": lines[i][j] if j < len(lines[i]) else "?"
                    })
                    if len(error_details) >= 200:
                        break

    data_size = host_array.nbytes + VALID_ARRAY.nbytes + line_lengths.nbytes + 4
    total_work_items = global_rows * global_cols
    total_work_groups = (global_rows // wg_rows) * (global_cols // wg_cols)

    gpu_metrics = {
        "transfer_h2d": round(t_h2d * 1000, 3),
        "kernel_time": round(t_kernel * 1000, 3),
        "transfer_d2h": round(t_d2h * 1000, 3),
        "total_gpu_time": round((t_h2d + t_kernel + t_d2h) * 1000, 3),
        "global_work_size": total_work_items,
        "grid_shape": f"{global_rows}×{global_cols}",
        "work_group_shape": f"{wg_rows}×{wg_cols}",
        "work_group_size": wg_rows * wg_cols,
        "work_groups": total_work_groups,
        "data_transferred_bytes": data_size,
        "lines_processed": num_lines,
        "chars_processed": int(np.sum(line_lengths)),
    }

    return transformed, error_count, error_details, gpu_metrics


# ─── Helpers ───────────────────────────────────────────────────────────────

def count_lines(path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def read_dna_lines(filepath: Path):
    """Read DNA lines from file, skipping headers (>) and empty lines."""
    lines = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n\r')
            if line.lstrip().startswith('>'):
                continue
            if not line.strip():
                continue
            lines.append(line)
    return lines


def auto_chunk_size_cpu(total_lines: int, cores: int) -> int:
    ideal = max(total_lines // (cores * 4), 5_000)
    return min(ideal, 200_000)


def auto_chunk_size_gpu(total_lines: int) -> int:
    if GPU_GLOBAL_MEM > 0:
        max_lines = int((GPU_GLOBAL_MEM * 0.3) / 200)
        ideal = min(max_lines, total_lines, 500_000)
        return max(ideal, 5_000)
    return min(max(total_lines // 10, 5_000), 200_000)


# ─── GPU Compare Worker ───────────────────────────────────────────────────

def gpu_compare_chunk(lines_a, lines_b, work_group_size=64):
    """Compare two lists of DNA lines on GPU. Returns (matches, compared, gpu_metrics)."""
    cl_ctx = _cl_context
    cl_q = _cl_queue
    num_lines = len(lines_a)
    max_len = max(
        max((len(l) for l in lines_a), default=1),
        max((len(l) for l in lines_b), default=1)
    )
    arr_a = np.zeros(num_lines * max_len, dtype=np.uint8)
    arr_b = np.zeros(num_lines * max_len, dtype=np.uint8)
    lens_a = np.zeros(num_lines, dtype=np.int32)
    lens_b = np.zeros(num_lines, dtype=np.int32)
    for i in range(num_lines):
        ea = lines_a[i].encode('utf-8'); eb = lines_b[i].encode('utf-8')
        lens_a[i] = len(ea); lens_b[i] = len(eb)
        off = i * max_len
        arr_a[off:off+len(ea)] = np.frombuffer(ea, dtype=np.uint8)
        arr_b[off:off+len(eb)] = np.frombuffer(eb, dtype=np.uint8)
    match_host = np.zeros(1, dtype=np.int32)
    compare_host = np.zeros(1, dtype=np.int32)
    t_h2d_start = time.perf_counter()
    mf = cl.mem_flags
    d_a = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_a)
    d_b = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr_b)
    d_la = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lens_a)
    d_lb = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lens_b)
    d_match = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=match_host)
    d_compare = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=compare_host)
    cl_q.finish()
    t_h2d = time.perf_counter() - t_h2d_start
    wg_cols = min(work_group_size, GPU_MAX_WORK_GROUP, max_len)
    wg_rows = max(1, min(GPU_MAX_WORK_GROUP // wg_cols, num_lines))
    global_rows = ((num_lines + wg_rows - 1) // wg_rows) * wg_rows
    global_cols = ((max_len + wg_cols - 1) // wg_cols) * wg_cols
    t_kernel_start = time.perf_counter()
    _cl_program.dna_compare_2d(cl_q, (global_rows, global_cols), (wg_rows, wg_cols),
        d_a, d_b, np.int32(max_len), d_la, d_lb, d_match, d_compare)
    cl_q.finish()
    t_kernel = time.perf_counter() - t_kernel_start
    t_d2h_start = time.perf_counter()
    cl.enqueue_copy(cl_q, match_host, d_match); cl.enqueue_copy(cl_q, compare_host, d_compare)
    cl_q.finish()
    t_d2h = time.perf_counter() - t_d2h_start
    gpu_metrics = {
        "transfer_h2d_ms": round(t_h2d * 1000, 3),
        "kernel_time_ms": round(t_kernel * 1000, 3),
        "transfer_d2h_ms": round(t_d2h * 1000, 3),
        "total_gpu_time_ms": round((t_h2d + t_kernel + t_d2h) * 1000, 3),
        "global_work_size": global_rows * global_cols,
        "grid_shape": f"{global_rows}×{global_cols}",
        "work_group_shape": f"{wg_rows}×{wg_cols}",
    }
    return int(match_host[0]), int(compare_host[0]), gpu_metrics


# ─── CPU Compare Worker ───────────────────────────────────────────────────

def cpu_compare_lines(args):
    """Compare two lists of line pairs on CPU. Returns (matches, compared, pid)."""
    (pairs,) = args
    matches = 0
    compared = 0
    for la, lb in pairs:
        min_len = min(len(la), len(lb))
        for i in range(min_len):
            compared += 1
            if la[i].upper() == lb[i].upper():
                matches += 1
    return matches, compared, os.getpid()


# ─── Run Compare (CPU) ────────────────────────────────────────────────────

def run_cpu_compare(job_id: str, path_a: Path, path_b: Path, cores: int):
    job = jobs[job_id]
    job["status"] = "reading"
    job["events"].append({"type": "status", "data": "reading"})
    try:
        lines_a = read_dna_lines(path_a); lines_b = read_dna_lines(path_b)
    except Exception as e:
        job["status"] = "error"
        job["events"].append({"type": "error", "data": f"Error leyendo archivos: {e}"})
        return
    total_a, total_b = len(lines_a), len(lines_b)
    min_lines = min(total_a, total_b)
    job["events"].append({"type": "info", "data": {"lines_a": total_a, "lines_b": total_b, "lines_compared": min_lines}})
    if min_lines == 0:
        job["status"] = "error"
        job["events"].append({"type": "error", "data": "Uno de los archivos no tiene líneas de ADN."})
        return
    job["status"] = "comparing"
    job["events"].append({"type": "status", "data": "comparing"})
    available = mp.cpu_count()
    cores = available if cores == 0 or cores > available else cores
    chunk_size = max(min_lines // (cores * 4), 1000)
    total_matches = 0; total_compared = 0; rows_processed = 0
    start_time = time.time()
    line_similarities = []
    with mp.Pool(processes=cores) as pool:
        for i in range(0, min_lines, chunk_size):
            end = min(i + chunk_size, min_lines)
            pairs = list(zip(lines_a[i:end], lines_b[i:end]))
            sub_size = max(1, len(pairs) // cores)
            sub_chunks = [pairs[j:j+sub_size] for j in range(0, len(pairs), sub_size)]
            results = pool.map(cpu_compare_lines, [(sc,) for sc in sub_chunks])
            for m, c, pid in results:
                total_matches += m; total_compared += c
            rows_processed += len(pairs)
            if len(line_similarities) < 100:
                for j in range(len(pairs)):
                    if len(line_similarities) >= 100: break
                    la, lb = pairs[j]
                    ml = min(len(la), len(lb))
                    if ml > 0:
                        lm = sum(1 for x, y in zip(la.upper(), lb.upper()) if x == y)
                        line_similarities.append({"line": i+j+1, "len_a": len(la), "len_b": len(lb), "matches": lm, "compared": ml, "similarity": round(lm/ml*100, 2)})
            elapsed = time.time() - start_time
            sim_pct = round(total_matches / total_compared * 100, 4) if total_compared > 0 else 0
            job["events"].append({"type": "progress", "data": {
                "rows_processed": rows_processed, "total_lines": min_lines,
                "total_matches": total_matches, "total_compared": total_compared,
                "similarity": sim_pct, "elapsed": round(elapsed, 2),
            }})
    elapsed = time.time() - start_time
    sim_pct = round(total_matches / total_compared * 100, 4) if total_compared > 0 else 0
    job["status"] = "done"
    job["events"].append({"type": "done", "data": {
        "mode": "cpu", "similarity": sim_pct,
        "total_matches": total_matches, "total_compared": total_compared,
        "total_mismatches": total_compared - total_matches,
        "lines_a": total_a, "lines_b": total_b, "lines_compared": min_lines,
        "elapsed": round(elapsed, 2), "line_details": line_similarities[:100],
    }})


# ─── Run Compare (GPU) ────────────────────────────────────────────────────

def run_gpu_compare(job_id: str, path_a: Path, path_b: Path, work_group_size: int):
    job = jobs[job_id]
    job["status"] = "reading"
    job["events"].append({"type": "status", "data": "reading"})
    try:
        lines_a = read_dna_lines(path_a); lines_b = read_dna_lines(path_b)
    except Exception as e:
        job["status"] = "error"
        job["events"].append({"type": "error", "data": f"Error leyendo archivos: {e}"})
        return
    total_a, total_b = len(lines_a), len(lines_b)
    min_lines = min(total_a, total_b)
    job["events"].append({"type": "info", "data": {"lines_a": total_a, "lines_b": total_b, "lines_compared": min_lines}})
    if min_lines == 0:
        job["status"] = "error"
        job["events"].append({"type": "error", "data": "Uno de los archivos no tiene líneas de ADN."})
        return
    job["status"] = "comparing"
    job["events"].append({"type": "status", "data": "comparing"})
    chunk_size = auto_chunk_size_gpu(min_lines)
    total_matches = 0; total_compared = 0; rows_processed = 0
    total_h2d = 0; total_kernel = 0; total_d2h = 0; total_work_items = 0
    chunks_processed = 0; start_time = time.time()
    line_similarities = []
    for i in range(0, min_lines, chunk_size):
        end = min(i + chunk_size, min_lines)
        chunk_a = lines_a[i:end]; chunk_b = lines_b[i:end]
        matches, compared, gm = gpu_compare_chunk(chunk_a, chunk_b, work_group_size)
        total_matches += matches; total_compared += compared
        chunks_processed += 1; rows_processed += len(chunk_a)
        total_h2d += gm["transfer_h2d_ms"]; total_kernel += gm["kernel_time_ms"]; total_d2h += gm["transfer_d2h_ms"]
        total_work_items += gm["global_work_size"]
        if len(line_similarities) < 100:
            for j in range(len(chunk_a)):
                if len(line_similarities) >= 100: break
                la, lb = chunk_a[j], chunk_b[j]
                ml = min(len(la), len(lb))
                if ml > 0:
                    lm = sum(1 for x, y in zip(la.upper(), lb.upper()) if x == y)
                    line_similarities.append({"line": i+j+1, "len_a": len(la), "len_b": len(lb), "matches": lm, "compared": ml, "similarity": round(lm/ml*100, 2)})
        elapsed = time.time() - start_time
        sim_pct = round(total_matches / total_compared * 100, 4) if total_compared > 0 else 0
        job["events"].append({"type": "progress", "data": {
            "rows_processed": rows_processed, "total_lines": min_lines,
            "total_matches": total_matches, "total_compared": total_compared,
            "similarity": sim_pct, "elapsed": round(elapsed, 2),
            "gpu_metrics": {
                "transfer_h2d_ms": round(total_h2d, 2), "kernel_time_ms": round(total_kernel, 2),
                "transfer_d2h_ms": round(total_d2h, 2),
                "total_gpu_time_ms": round(total_h2d + total_kernel + total_d2h, 2),
                "chunks_processed": chunks_processed, "total_work_items": total_work_items,
                "current_grid_shape": gm["grid_shape"], "current_wg_shape": gm["work_group_shape"],
            },
        }})
    elapsed = time.time() - start_time
    sim_pct = round(total_matches / total_compared * 100, 4) if total_compared > 0 else 0
    job["status"] = "done"
    job["events"].append({"type": "done", "data": {
        "mode": "gpu", "similarity": sim_pct,
        "total_matches": total_matches, "total_compared": total_compared,
        "total_mismatches": total_compared - total_matches,
        "lines_a": total_a, "lines_b": total_b, "lines_compared": min_lines,
        "elapsed": round(elapsed, 2), "line_details": line_similarities[:100],
        "gpu_summary": {
            "gpu_name": GPU_NAME, "total_h2d_ms": round(total_h2d, 2),
            "total_kernel_ms": round(total_kernel, 2), "total_d2h_ms": round(total_d2h, 2),
            "total_gpu_ms": round(total_h2d + total_kernel + total_d2h, 2),
            "chunks_processed": chunks_processed, "total_work_items": total_work_items,
            "throughput": round(total_compared / elapsed) if elapsed > 0 else 0,
        },
    }})


# ─── CPU Analysis ──────────────────────────────────────────────────────────

def run_cpu_analysis(job_id: str, filepath: Path, cores: int):
    """Run the full CPU analysis in a background thread."""
    job = jobs[job_id]
    job["status"] = "counting"
    job["events"].append({"type": "status", "data": "counting"})

    total_lines = count_lines(filepath)
    job["total_lines"] = total_lines
    job["events"].append({"type": "total_lines", "data": total_lines})

    available = mp.cpu_count()
    cores = available if cores == 0 or cores > available else cores
    job["cores_used"] = cores

    chunk_size = auto_chunk_size_cpu(total_lines, cores)

    job["events"].append({
        "type": "info",
        "data": {
            "mode": "cpu",
            "cores": cores,
            "chunk_size": chunk_size,
            "total_lines": total_lines,
        },
    })

    job["status"] = "processing"
    job["events"].append({"type": "status", "data": "processing"})

    core_load: dict = {}
    total_errors = 0
    all_error_details = []
    rows_processed = 0
    start_time = time.time()

    out_path = filepath.parent / f"{filepath.stem}_errors.txt"

    with (
        open(filepath, "r", encoding="utf-8", errors="replace") as fin,
        open(out_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as fout,
        mp.Pool(processes=cores) as pool,
    ):
        chunk = []
        global_row = 0

        def flush_chunk(chunk):
            nonlocal total_errors, rows_processed

            sub_size = max(1, len(chunk) // cores)
            sub_chunks = [
                chunk[i : i + sub_size] for i in range(0, len(chunk), sub_size)
            ]

            job_results = pool.map(cpu_process_chunk, [(sc,) for sc in sub_chunks])

            for transformed_lines, errors, error_details, pid, count in job_results:
                for line in transformed_lines:
                    fout.write(line + "\n")
                total_errors += errors
                rows_processed += count

                if len(all_error_details) < 200:
                    all_error_details.extend(
                        error_details[: 200 - len(all_error_details)]
                    )

                pid_str = str(pid)
                core_load[pid_str] = core_load.get(pid_str, 0) + count

            fout.flush()

            elapsed = time.time() - start_time
            job["events"].append({
                "type": "progress",
                "data": {
                    "rows_processed": rows_processed,
                    "total_lines": total_lines,
                    "errors": total_errors,
                    "elapsed": round(elapsed, 2),
                    "core_load": dict(core_load),
                },
            })

        for raw_line in fin:
            global_row += 1
            line = raw_line.rstrip("\n\r")

            if line.lstrip().startswith(">"):
                rows_processed += 1
                continue
            if not line.strip():
                rows_processed += 1
                continue

            chunk.append((global_row, line))

            if len(chunk) >= chunk_size:
                flush_chunk(chunk)
                chunk.clear()

        if chunk:
            flush_chunk(chunk)

    elapsed = time.time() - start_time

    job["status"] = "done"
    job["total_errors"] = total_errors
    job["error_details"] = all_error_details[:200]
    job["elapsed"] = round(elapsed, 2)
    job["core_load"] = core_load
    job["rows_processed"] = rows_processed
    job["output_file"] = str(out_path)

    job["events"].append({
        "type": "done",
        "data": {
            "mode": "cpu",
            "total_errors": total_errors,
            "elapsed": round(elapsed, 2),
            "core_load": core_load,
            "rows_processed": rows_processed,
            "error_details": all_error_details[:200],
            "output_file": str(out_path),
        },
    })


# ─── GPU Analysis ──────────────────────────────────────────────────────────

def run_gpu_analysis(job_id: str, filepath: Path, work_group_size: int):
    """Run the full GPU analysis in a background thread."""
    job = jobs[job_id]
    job["status"] = "counting"
    job["events"].append({"type": "status", "data": "counting"})

    total_lines = count_lines(filepath)
    job["total_lines"] = total_lines
    job["events"].append({"type": "total_lines", "data": total_lines})

    chunk_size = auto_chunk_size_gpu(total_lines)

    job["events"].append({
        "type": "info",
        "data": {
            "mode": "gpu",
            "gpu_name": GPU_NAME,
            "work_group_size": work_group_size,
            "chunk_size": chunk_size,
            "total_lines": total_lines,
        },
    })

    job["status"] = "processing"
    job["events"].append({"type": "status", "data": "processing"})

    total_errors = 0
    all_error_details = []
    rows_processed = 0
    start_time = time.time()

    total_h2d = 0
    total_kernel = 0
    total_d2h = 0
    total_work_items = 0
    total_chars = 0
    chunks_processed = 0

    out_path = filepath.parent / f"{filepath.stem}_gpu_errors.txt"

    try:
        with (
            open(filepath, "r", encoding="utf-8", errors="replace") as fin,
            open(out_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as fout,
        ):
            chunk = []
            global_row = 0

            def flush_chunk(chunk):
                nonlocal total_errors, rows_processed, total_h2d, total_kernel
                nonlocal total_d2h, total_work_items, total_chars, chunks_processed

                transformed_lines, errors, error_details, gpu_metrics = gpu_process_chunk(
                    chunk, work_group_size
                )

                for line in transformed_lines:
                    fout.write(line + "\n")

                total_errors += errors
                rows_processed += len(chunk)
                chunks_processed += 1

                total_h2d += gpu_metrics["transfer_h2d"]
                total_kernel += gpu_metrics["kernel_time"]
                total_d2h += gpu_metrics["transfer_d2h"]
                total_work_items += gpu_metrics["global_work_size"]
                total_chars += gpu_metrics["chars_processed"]

                if len(all_error_details) < 200:
                    all_error_details.extend(
                        error_details[:200 - len(all_error_details)]
                    )

                fout.flush()

                elapsed = time.time() - start_time
                job["events"].append({
                    "type": "progress",
                    "data": {
                        "rows_processed": rows_processed,
                        "total_lines": total_lines,
                        "errors": total_errors,
                        "elapsed": round(elapsed, 2),
                        "gpu_metrics": {
                            "transfer_h2d_ms": round(total_h2d, 2),
                            "kernel_time_ms": round(total_kernel, 2),
                            "transfer_d2h_ms": round(total_d2h, 2),
                            "total_gpu_time_ms": round(total_h2d + total_kernel + total_d2h, 2),
                            "chunks_processed": chunks_processed,
                            "total_work_items": total_work_items,
                            "total_chars_processed": total_chars,
                            "current_grid_shape": gpu_metrics["grid_shape"],
                            "current_wg_shape": gpu_metrics["work_group_shape"],
                        },
                    },
                })

            for raw_line in fin:
                global_row += 1
                line = raw_line.rstrip("\n\r")

                if line.lstrip().startswith(">"):
                    rows_processed += 1
                    continue
                if not line.strip():
                    rows_processed += 1
                    continue

                chunk.append((global_row, line))

                if len(chunk) >= chunk_size:
                    flush_chunk(chunk)
                    chunk.clear()

            if chunk:
                flush_chunk(chunk)

        elapsed = time.time() - start_time

        job["status"] = "done"
        job["total_errors"] = total_errors
        job["error_details"] = all_error_details[:200]
        job["elapsed"] = round(elapsed, 2)
        job["rows_processed"] = rows_processed
        job["output_file"] = str(out_path)

        job["events"].append({
            "type": "done",
            "data": {
                "mode": "gpu",
                "total_errors": total_errors,
                "elapsed": round(elapsed, 2),
                "rows_processed": rows_processed,
                "error_details": all_error_details[:200],
                "output_file": str(out_path),
                "gpu_summary": {
                    "gpu_name": GPU_NAME,
                    "platform": GPU_PLATFORM_NAME,
                    "compute_units": GPU_COMPUTE_UNITS,
                    "total_transfer_h2d_ms": round(total_h2d, 2),
                    "total_kernel_time_ms": round(total_kernel, 2),
                    "total_transfer_d2h_ms": round(total_d2h, 2),
                    "total_gpu_time_ms": round(total_h2d + total_kernel + total_d2h, 2),
                    "chunks_processed": chunks_processed,
                    "total_work_items": total_work_items,
                    "total_chars_processed": total_chars,
                    "throughput_chars_per_sec": round(total_chars / elapsed) if elapsed > 0 else 0,
                },
            },
        })

    except Exception as e:
        job["status"] = "error"
        job["events"].append({"type": "error", "data": str(e)})


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("unified_index.html",
        cpu_count=mp.cpu_count(),
        gpu_available=GPU_AVAILABLE,
        gpu_name=GPU_NAME,
        gpu_driver=GPU_DRIVER,
        gpu_compute_units=GPU_COMPUTE_UNITS,
        gpu_max_work_group=GPU_MAX_WORK_GROUP,
        gpu_global_mem=GPU_GLOBAL_MEM,
        gpu_local_mem=GPU_LOCAL_MEM,
        gpu_device_type=GPU_DEVICE_TYPE,
        gpu_platform=GPU_PLATFORM_NAME,
    )


@app.route("/api/system-info")
def system_info():
    return jsonify({
        "cpu_count": mp.cpu_count(),
        "gpu": {
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
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    filepath = data.get("filepath", "").strip()
    mode = data.get("mode", "cpu").strip().lower()
    cores = int(data.get("cores", 0))
    work_group_size = int(data.get("work_group_size", 64))

    if not filepath:
        return jsonify({"error": "No se especificó la ruta del archivo."}), 400

    path = Path(filepath).expanduser().resolve()

    if not path.exists():
        return jsonify({"error": f"Archivo no encontrado: {path}"}), 404
    if not path.is_file():
        return jsonify({"error": f"La ruta no es un archivo: {path}"}), 400

    if mode == "gpu" and not GPU_AVAILABLE:
        return jsonify({"error": "No se detectó una GPU OpenCL compatible."}), 400

    if mode == "gpu" and (work_group_size < 1 or work_group_size > GPU_MAX_WORK_GROUP):
        work_group_size = min(64, GPU_MAX_WORK_GROUP)

    file_size = path.stat().st_size
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "filename": path.name,
        "filepath": str(path),
        "file_size": file_size,
        "mode": mode,
        "status": "queued",
        "events": [],
        "total_errors": 0,
        "error_details": [],
        "elapsed": 0,
        "core_load": {},
    }

    if mode == "gpu":
        t = threading.Thread(
            target=run_gpu_analysis,
            args=(job_id, path, work_group_size),
            daemon=True,
        )
    else:
        t = threading.Thread(
            target=run_cpu_analysis,
            args=(job_id, path, cores),
            daemon=True,
        )
    t.start()

    return jsonify({
        "job_id": job_id,
        "filename": path.name,
        "file_size": file_size,
        "mode": mode,
    })


@app.route("/api/browse", methods=["POST"])
def browse():
    data = request.get_json()
    dir_path = data.get("path", "~")
    path = Path(dir_path).expanduser().resolve()

    if not path.exists() or not path.is_dir():
        path = Path.home()

    entries = []
    try:
        if path.parent != path:
            entries.append({"name": "..", "path": str(path.parent), "is_dir": True, "size": 0})
        for item in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.name.startswith("."):
                continue
            try:
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                })
            except PermissionError:
                continue
    except PermissionError:
        pass

    return jsonify({"current": str(path), "entries": entries})


@app.route("/api/compare", methods=["POST"])
def compare():
    data = request.get_json()
    filepath_a = data.get("filepath_a", "").strip()
    filepath_b = data.get("filepath_b", "").strip()
    mode = data.get("mode", "cpu").strip().lower()
    cores = int(data.get("cores", 0))
    work_group_size = int(data.get("work_group_size", 64))

    if not filepath_a or not filepath_b:
        return jsonify({"error": "Se requieren ambos archivos."}), 400

    path_a = Path(filepath_a).expanduser().resolve()
    path_b = Path(filepath_b).expanduser().resolve()

    for p, name in [(path_a, "A"), (path_b, "B")]:
        if not p.exists():
            return jsonify({"error": f"Archivo {name} no encontrado: {p}"}), 404
        if not p.is_file():
            return jsonify({"error": f"La ruta {name} no es un archivo: {p}"}), 400

    if mode == "gpu" and not GPU_AVAILABLE:
        return jsonify({"error": "No se detectó una GPU OpenCL compatible."}), 400

    if mode == "gpu" and (work_group_size < 1 or work_group_size > GPU_MAX_WORK_GROUP):
        work_group_size = min(64, GPU_MAX_WORK_GROUP)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "filename_a": path_a.name, "filename_b": path_b.name,
        "mode": mode, "status": "queued", "events": [], "elapsed": 0,
    }

    if mode == "gpu":
        t = threading.Thread(target=run_gpu_compare, args=(job_id, path_a, path_b, work_group_size), daemon=True)
    else:
        t = threading.Thread(target=run_cpu_compare, args=(job_id, path_a, path_b, cores), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "filename_a": path_a.name, "filename_b": path_b.name, "mode": mode})


@app.route("/api/stream/<job_id>")
def stream(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    def event_stream():
        idx = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                break
            while idx < len(job["events"]):
                yield f"data: {json.dumps(job['events'][idx])}\n\n"
                idx += 1
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.3)

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    print(f"\n🧬 DNA Unified Checker v4.0")
    print(f"   CPU: {mp.cpu_count()} núcleos")
    print(f"   GPU: {'✅ ' + GPU_NAME if GPU_AVAILABLE else '❌ ' + GPU_NAME}")
    if GPU_AVAILABLE:
        print(f"   Plataforma: {GPU_PLATFORM_NAME}")
        print(f"   Compute Units: {GPU_COMPUTE_UNITS}")
        print(f"   Memoria: {GPU_GLOBAL_MEM / (1024**2):.0f} MB")
    print()
    app.run(debug=True, port=5000, threaded=True)
