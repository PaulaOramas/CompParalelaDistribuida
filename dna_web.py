"""
DNA Parallel Checker v2.0 — Web Interface
Flask backend that processes DNA files using multiprocessing
and streams real-time progress via Server-Sent Events.
Supports local file path selection (no upload size limit).
"""

import json
import multiprocessing as mp
import os
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

# ─── In-memory job store ────────────────────────────────────────────────────

jobs: dict = {}

# ─── DNA Processing Logic ──────────────────────────────────────────────────

VALID = frozenset("ATCGNatcgn")


def process_chunk(args):
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


def count_lines(path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def auto_chunk_size(total_lines: int, cores: int) -> int:
    """Automatically determine a good chunk size based on file and core count."""
    # Aim for ~4 flushes per core, with bounds
    ideal = max(total_lines // (cores * 4), 5_000)
    return min(ideal, 200_000)


def run_analysis(job_id: str, filepath: Path, cores: int):
    """Run the full analysis in a background thread."""
    job = jobs[job_id]
    job["status"] = "counting"
    job["events"].append({"type": "status", "data": "counting"})

    total_lines = count_lines(filepath)
    job["total_lines"] = total_lines
    job["events"].append({"type": "total_lines", "data": total_lines})

    available = mp.cpu_count()
    cores = available if cores == 0 or cores > available else cores
    job["cores_used"] = cores

    chunk_size = auto_chunk_size(total_lines, cores)

    job["events"].append(
        {
            "type": "info",
            "data": {
                "cores": cores,
                "chunk_size": chunk_size,
                "total_lines": total_lines,
            },
        }
    )

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

            job_results = pool.map(process_chunk, [(sc,) for sc in sub_chunks])

            for transformed_lines, errors, error_details, pid, count in job_results:
                for line in transformed_lines:
                    fout.write(line + "\n")
                total_errors += errors
                rows_processed += count

                # Only keep first 200 error details for the UI
                if len(all_error_details) < 200:
                    all_error_details.extend(
                        error_details[: 200 - len(all_error_details)]
                    )

                pid_str = str(pid)
                core_load[pid_str] = core_load.get(pid_str, 0) + count

            fout.flush()

            elapsed = time.time() - start_time
            job["events"].append(
                {
                    "type": "progress",
                    "data": {
                        "rows_processed": rows_processed,
                        "total_lines": total_lines,
                        "errors": total_errors,
                        "elapsed": round(elapsed, 2),
                        "core_load": dict(core_load),
                    },
                }
            )

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

    job["events"].append(
        {
            "type": "done",
            "data": {
                "total_errors": total_errors,
                "elapsed": round(elapsed, 2),
                "core_load": core_load,
                "rows_processed": rows_processed,
                "error_details": all_error_details[:200],
                "output_file": str(out_path),
            },
        }
    )


# ─── Routes ─────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html", cpu_count=mp.cpu_count())


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Start analysis from a local file path."""
    data = request.get_json()
    filepath = data.get("filepath", "").strip()
    cores = int(data.get("cores", 0))

    if not filepath:
        return jsonify({"error": "No se especificó la ruta del archivo."}), 400

    path = Path(filepath).expanduser().resolve()

    if not path.exists():
        return jsonify({"error": f"Archivo no encontrado: {path}"}), 404

    if not path.is_file():
        return jsonify({"error": f"La ruta no es un archivo: {path}"}), 400

    file_size = path.stat().st_size

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "filename": path.name,
        "filepath": str(path),
        "file_size": file_size,
        "status": "queued",
        "events": [],
        "event_index": 0,
        "total_errors": 0,
        "error_details": [],
        "elapsed": 0,
        "core_load": {},
    }

    # Start processing in background thread
    t = threading.Thread(
        target=run_analysis,
        args=(job_id, path, cores),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "filename": path.name, "file_size": file_size})


@app.route("/api/browse", methods=["POST"])
def browse():
    """List files in a directory for the file browser."""
    data = request.get_json()
    dir_path = data.get("path", "~")

    path = Path(dir_path).expanduser().resolve()

    if not path.exists() or not path.is_dir():
        path = Path.home()

    entries = []
    try:
        # Add parent directory entry
        if path.parent != path:
            entries.append(
                {
                    "name": "..",
                    "path": str(path.parent),
                    "is_dir": True,
                    "size": 0,
                }
            )

        for item in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.name.startswith("."):
                continue
            try:
                entries.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "is_dir": item.is_dir(),
                        "size": item.stat().st_size if item.is_file() else 0,
                    }
                )
            except PermissionError:
                continue
    except PermissionError:
        pass

    return jsonify({"current": str(path), "entries": entries})


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
                evt = job["events"][idx]
                yield f"data: {json.dumps(evt)}\n\n"
                idx += 1

            if job["status"] == "done":
                break

            time.sleep(0.3)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/info")
def info():
    return jsonify({"cpu_count": mp.cpu_count()})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
