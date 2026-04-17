"""
DNA Distributed Coordinator v2.0
=================================
Coordinador principal que distribuye la carga de comparación de ADN
entre nodos workers conectados. Incluye interfaz web Flask.

Características:
  - Lectura de archivos con CPU, procesamiento con GPU (OpenCL)
  - Distribución configurable: todos los nodos, nodo específico, excluir maestro
  - Tolerancia a fallos: redistribución automática si un nodo cae
  - Failover del maestro: elección automática vía Bully algorithm
  - Gestión de nodos: activar/desactivar, eliminar nodos
  - Configuración GPU: elegir compute units y work group size
  - Interfaz web con monitoreo en tiempo real (SSE)

Uso:
    python dna_distributed_coordinator.py
    python dna_distributed_coordinator.py --port 5555 --web-port 5000
"""

import argparse
import json
import multiprocessing as mp
import os
import signal
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import zmq
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
_gpu_device = None

try:
    import pyopencl as cl
    import numpy as np

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
HEARTBEAT_TIMEOUT = 30.0    # seconds before considering a node dead
CHUNK_SIZE = 50_000          # lines per chunk default
BROADCAST_INTERVAL = 2.0     # seconds between coordinator heartbeats
WINDOW_SIZE = 3              # max chunks in-flight per worker (flow control)
ACTIVE_COORD_FILE = Path(__file__).parent / ".active_coordinator"
UDP_DISCOVERY_PORT = 5599    # UDP port for LAN coordinator discovery

# ─── Flask App ──────────────────────────────────────────────────────────────

app = Flask(__name__)

# ─── Worker Registry ───────────────────────────────────────────────────────

class WorkerInfo:
    """Info about a connected worker node."""
    def __init__(self, node_id: str, node_name: str, hostname: str,
                 pid: int, cpu_count: int, gpu_info: dict = None):
        self.node_id = node_id
        self.node_name = node_name
        self.hostname = hostname
        self.pid = pid
        self.cpu_count = cpu_count
        self.last_heartbeat = time.time()
        self.registered_at = time.time()
        self.connected = True
        self.enabled = True          # Can be toggled on/off from UI
        self.processing = False
        self.current_chunk = None
        self.chunks_processed = 0
        self.total_lines = 0
        self.total_matches = 0
        self.total_compared = 0

        # GPU info from worker
        self.gpu_info = gpu_info or {}
        self.gpu_available = self.gpu_info.get("available", False)
        self.gpu_name = self.gpu_info.get("name", "No disponible")
        self.gpu_compute_units = self.gpu_info.get("compute_units", 0)
        self.gpu_max_work_group = self.gpu_info.get("max_work_group_size", 0)
        self.gpu_global_mem = self.gpu_info.get("global_memory", 0)

        # Configurable GPU settings (can be set per-node from UI)
        self.gpu_work_group_size = min(64, self.gpu_max_work_group) if self.gpu_max_work_group > 0 else 64
        self.gpu_compute_units_to_use = self.gpu_compute_units  # use all by default

        # CPU config (can be set per-node from UI)
        self.cpu_cores_to_use = self.cpu_count  # use all by default

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "hostname": self.hostname,
            "pid": self.pid,
            "cpu_count": self.cpu_count,
            "last_heartbeat": self.last_heartbeat,
            "registered_at": self.registered_at,
            "connected": self.connected,
            "enabled": self.enabled,
            "processing": self.processing,
            "current_chunk": self.current_chunk,
            "chunks_processed": self.chunks_processed,
            "total_lines": self.total_lines,
            "total_matches": self.total_matches,
            "total_compared": self.total_compared,
            "time_since_heartbeat": round(time.time() - self.last_heartbeat, 1),
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_compute_units": self.gpu_compute_units,
            "gpu_max_work_group": self.gpu_max_work_group,
            "gpu_global_mem": self.gpu_global_mem,
            "gpu_work_group_size": self.gpu_work_group_size,
            "gpu_compute_units_to_use": self.gpu_compute_units_to_use,
            "cpu_cores_to_use": self.cpu_cores_to_use,
        }


# ─── Distributed Coordinator ──────────────────────────────────────────────

class DistributedCoordinator:
    def __init__(self, zmq_port: int = 5555, web_port: int = 5000):
        self.zmq_port = zmq_port
        self.web_port = web_port
        self.coordinator_id = str(uuid.uuid4())[:8]
        self.hostname = socket.gethostname()
        self.local_ip = self._get_local_ip()

        # Worker registry
        self.workers: dict[str, WorkerInfo] = {}
        self.workers_lock = threading.RLock()

        # Job management
        self.jobs: dict = {}
        self.jobs_lock = threading.RLock()

        # ZMQ context
        self.context = zmq.Context()
        self.running = False

        # Pending chunks for reassignment
        self.pending_chunks: dict[str, dict] = {}
        self.completed_chunks: dict[str, dict] = {}
        self.chunk_assignments: dict[str, str] = {}  # chunk_id -> node_id

        # Flow control: queue of chunks not yet sent
        self.chunks_queue: dict[str, list] = {}      # job_id -> [(chunk_id, chunk_data)]
        self.in_flight: dict[str, int] = {}           # node_id -> count of chunks in flight
        self._queue_lock = threading.Lock()
        self._router_lock = threading.Lock()          # Protect ROUTER socket (multi-thread sends)

        # State for failover propagation
        self.state_version = 0

        print(f"\n{'='*60}")
        print(f"  🧬 DNA Distributed Coordinator v2.0")
        print(f"  ID:        {self.coordinator_id}")
        print(f"  Host:      {self.hostname}")
        print(f"  IP Local:  {self.local_ip}")
        print(f"  ZMQ Port:  {zmq_port}")
        print(f"  Web Port:  {web_port}")
        print(f"  CPUs:      {mp.cpu_count()}")
        print(f"  GPU:       {'✅ ' + GPU_NAME if GPU_AVAILABLE else '❌ ' + GPU_NAME}")
        if GPU_AVAILABLE:
            print(f"  GPU CUs:   {GPU_COMPUTE_UNITS}")
            print(f"  GPU Mem:   {GPU_GLOBAL_MEM / (1024**2):.0f} MB")
            print(f"  Max WG:    {GPU_MAX_WORK_GROUP}")
        print(f"{'='*60}\n")

    def _get_local_ip(self):
        """Get the local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def start(self):
        """Start the coordinator."""
        self.running = True

        # ROUTER socket for worker communication
        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.bind(f"tcp://*:{self.zmq_port}")

        # PUB socket for broadcasting
        self.pub = self.context.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(f"tcp://*:{self.zmq_port + 1}")

        print(f"  ✅ ZMQ ROUTER en puerto {self.zmq_port}")
        print(f"  ✅ ZMQ PUB en puerto {self.zmq_port + 1}")

        # Write active coordinator address to file
        try:
            ACTIVE_COORD_FILE.write_text(json.dumps({
                "addr": f"{self.local_ip}:{self.zmq_port}",
                "web": f"http://{self.local_ip}:{self.web_port}",
                "pid": os.getpid(),
                "started": time.time(),
            }))
        except Exception:
            pass

        # Start background threads
        self.msg_thread = threading.Thread(
            target=self._message_loop, daemon=True
        )
        self.msg_thread.start()

        self.health_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self.health_thread.start()

        self.broadcast_thread = threading.Thread(
            target=self._broadcast_loop, daemon=True
        )
        self.broadcast_thread.start()

        # Self-demotion thread: detect if another coordinator exists
        self._demotion_thread = threading.Thread(
            target=self._self_demotion_loop, daemon=True
        )
        self._demotion_thread.start()

        print(f"  ✅ Hilos de control iniciados")
        print(f"\n  🌐 Interfaz web: http://{self.local_ip}:{self.web_port}")
        print(f"  📡 Workers deben conectar a: {self.local_ip}:{self.zmq_port}")
        print(f"\n  Esperando workers...\n")

    def _self_demotion_loop(self):
        """Periodically check if this coordinator should demote itself.
        If we have zero workers for a while and detect another coordinator
        on the network, we should step down and become a worker."""
        time.sleep(15)  # Grace period at startup
        while self.running:
            time.sleep(10)
            # Only check if we have no workers
            with self.workers_lock:
                connected_count = sum(
                    1 for w in self.workers.values() if w.connected
                )
            if connected_count > 0:
                continue

            # Listen for UDP broadcasts from another coordinator
            try:
                udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                udp_sock.settimeout(5.0)
                udp_sock.bind(("", UDP_DISCOVERY_PORT))

                data, sender_addr = udp_sock.recvfrom(4096)
                msg = json.loads(data.decode())
                udp_sock.close()

                if (msg.get("type") == "COORDINATOR_ANNOUNCE" and
                        msg.get("coordinator_id") != self.coordinator_id):
                    other_addr = msg.get("addr", "")
                    print(f"\n  ⚠ Otro coordinador detectado en {other_addr}")
                    print(f"  🔄 Este coordinador no tiene workers → demotándose a worker...")
                    self.stop()

                    # Relaunch as worker
                    from dna_distributed_node import WorkerNode
                    node = WorkerNode(other_addr, f"ex-coord-{self.hostname}")
                    node.start()
                    return
            except socket.timeout:
                pass
            except OSError:
                pass
            except Exception:
                pass

    def _message_loop(self):
        """Main loop to receive messages from workers."""
        poller = zmq.Poller()
        poller.register(self.router, zmq.POLLIN)

        while self.running:
            try:
                events = dict(poller.poll(500))
                if self.router in events:
                    frames = self.router.recv_multipart()
                    if len(frames) < 2:
                        continue
                    identity = frames[0]
                    msg = json.loads(frames[-1])
                    self._handle_message(identity.decode(), msg)
            except zmq.ZMQError:
                time.sleep(0.1)
            except Exception as e:
                import traceback
                print(f"  ❌ Message loop error: {e}", flush=True)
                traceback.print_exc()
                time.sleep(0.1)

    def _handle_message(self, identity: str, msg: dict):
        """Handle incoming message from a worker."""
        msg_type = msg.get("type", "")
        node_id = msg.get("node_id", identity)
        node_name = msg.get("node_name", node_id)
        data = msg.get("data", {})

        # ANY message from a worker proves it's alive — refresh heartbeat
        with self.workers_lock:
            if node_id in self.workers:
                self.workers[node_id].last_heartbeat = time.time()
                # Auto-reconnect if it was marked dead but is sending messages
                if not self.workers[node_id].connected:
                    self.workers[node_id].connected = True
                    print(f"\n  🔄 Worker {self.workers[node_id].node_name} reconectado automáticamente")

        if msg_type == "REGISTER":
            self._register_worker(node_id, node_name, data)

        elif msg_type == "HEARTBEAT":
            self._update_heartbeat(node_id, node_name, data)

        elif msg_type == "RESULT":
            # Process results in a separate thread to avoid blocking heartbeat reception
            threading.Thread(
                target=self._handle_result, args=(node_id, data),
                daemon=True
            ).start()

        elif msg_type == "VALIDATE_RESULT":
            threading.Thread(
                target=self._handle_validate_result, args=(node_id, data),
                daemon=True
            ).start()

        elif msg_type == "UNREGISTER":
            self._unregister_worker(node_id)

        elif msg_type == "PONG":
            with self.workers_lock:
                if node_id in self.workers:
                    self.workers[node_id].last_heartbeat = time.time()

        elif msg_type == "ELECTION_RESPONSE":
            pass  # Handled by workers during election

        elif msg_type == "VALIDATE_RESULT":
            self._handle_validate_result(node_id, data)

    def _register_worker(self, node_id: str, node_name: str, data: dict):
        """Register a new worker."""
        gpu_info = data.get("gpu_info", {})
        with self.workers_lock:
            self.workers[node_id] = WorkerInfo(
                node_id=node_id,
                node_name=node_name,
                hostname=data.get("hostname", "unknown"),
                pid=data.get("pid", 0),
                cpu_count=data.get("cpu_count", 1),
                gpu_info=gpu_info,
            )
        gpu_status = f"GPU: {gpu_info.get('name', 'N/A')}" if gpu_info.get("available") else "Sin GPU"
        print(f"  ✅ Worker registrado: {node_name} ({node_id}) "
              f"desde {data.get('hostname', '?')} — {gpu_status}", flush=True)

        self._broadcast_peer_list()

        # If there's an active job with queued chunks, send work to the new worker
        self._assign_queued_work_to_worker(node_id)

    def _update_heartbeat(self, node_id: str, node_name: str, data: dict):
        """Update heartbeat for a worker."""
        with self.workers_lock:
            if node_id in self.workers:
                w = self.workers[node_id]
                w.last_heartbeat = time.time()
                w.connected = True
                w.processing = data.get("processing", False)
                w.current_chunk = data.get("current_chunk")
                w.chunks_processed = data.get("chunks_processed", 0)
                w.total_lines = data.get("total_lines", 0)
                w.total_matches = data.get("total_matches", 0)
                w.total_compared = data.get("total_compared", 0)
                # Update GPU info if provided
                if "gpu_info" in data:
                    gi = data["gpu_info"]
                    w.gpu_available = gi.get("available", w.gpu_available)
                    w.gpu_name = gi.get("name", w.gpu_name)
                    w.gpu_compute_units = gi.get("compute_units", w.gpu_compute_units)
                    w.gpu_max_work_group = gi.get("max_work_group_size", w.gpu_max_work_group)
                    w.gpu_global_mem = gi.get("global_memory", w.gpu_global_mem)
            else:
                self._register_worker(node_id, node_name, data)

    def _handle_result(self, node_id: str, data: dict):
        """Handle a result from a worker."""
        chunk_id = data.get("chunk_id", "")
        job_id = data.get("job_id", "")

        with self.jobs_lock:
            if job_id not in self.jobs:
                return
            job = self.jobs[job_id]

        # Remove from assignments
        if chunk_id in self.chunk_assignments:
            del self.chunk_assignments[chunk_id]

        # Store completed chunk result
        if job_id not in self.completed_chunks:
            self.completed_chunks[job_id] = {}
        self.completed_chunks[job_id][chunk_id] = data

        # Remove from pending
        if job_id in self.pending_chunks and chunk_id in self.pending_chunks[job_id]:
            del self.pending_chunks[job_id][chunk_id]

        # Aggregate results
        chunk_index = data.get("chunk_index", 0)
        matches = data.get("matches", 0)
        compared = data.get("compared", 0)
        elapsed = data.get("elapsed", 0)
        lines_processed = data.get("lines_processed", 0)
        gpu_metrics = data.get("gpu_metrics", {})

        with self.jobs_lock:
            job["total_matches"] += matches
            job["total_compared"] += compared
            job["chunks_completed"] += 1
            job["lines_processed"] += lines_processed

            # Per-node tracking
            if node_id not in job["node_stats"]:
                job["node_stats"][node_id] = {
                    "chunks": 0, "lines": 0, "matches": 0,
                    "compared": 0, "time": 0, "gpu_metrics": {},
                }
            job["node_stats"][node_id]["chunks"] += 1
            job["node_stats"][node_id]["lines"] += lines_processed
            job["node_stats"][node_id]["matches"] += matches
            job["node_stats"][node_id]["compared"] += compared
            job["node_stats"][node_id]["time"] += elapsed
            if gpu_metrics:
                existing = job["node_stats"][node_id].get("gpu_metrics", {})
                existing["total_kernel_ms"] = existing.get("total_kernel_ms", 0) + gpu_metrics.get("kernel_time_ms", 0)
                existing["total_h2d_ms"] = existing.get("total_h2d_ms", 0) + gpu_metrics.get("transfer_h2d_ms", 0)
                existing["total_d2h_ms"] = existing.get("total_d2h_ms", 0) + gpu_metrics.get("transfer_d2h_ms", 0)
                job["node_stats"][node_id]["gpu_metrics"] = existing

            # Progress
            total_chunks = job.get("total_chunks", 1)
            sim_pct = round(
                job["total_matches"] / max(job["total_compared"], 1) * 100, 4
            )
            job_elapsed = time.time() - job["start_time"]

            # Get node names for stats
            node_stats_named = {}
            with self.workers_lock:
                for nid, stats in job["node_stats"].items():
                    w = self.workers.get(nid)
                    name = w.node_name if w else nid
                    node_stats_named[name] = stats

            event = {
                "type": "progress",
                "data": {
                    "chunks_completed": job["chunks_completed"],
                    "total_chunks": total_chunks,
                    "lines_processed": job["lines_processed"],
                    "total_lines": job.get("total_lines", 0),
                    "total_matches": job["total_matches"],
                    "total_compared": job["total_compared"],
                    "similarity": sim_pct,
                    "elapsed": round(job_elapsed, 2),
                    "node_stats": node_stats_named,
                    "processing_node": node_id,
                    "chunk_elapsed": elapsed,
                },
            }
            job["events"].append(event)

            # Check if job is complete
            if job["chunks_completed"] >= total_chunks:
                job["status"] = "done"
                job["elapsed"] = round(job_elapsed, 2)
                job["similarity"] = sim_pct

                all_details = []
                for cid in sorted(self.completed_chunks.get(job_id, {}).keys()):
                    cdata = self.completed_chunks[job_id][cid]
                    all_details.extend(cdata.get("line_details", []))

                done_event = {
                    "type": "done",
                    "data": {
                        "similarity": sim_pct,
                        "total_matches": job["total_matches"],
                        "total_compared": job["total_compared"],
                        "total_mismatches": job["total_compared"] - job["total_matches"],
                        "lines_a": job.get("lines_a", 0),
                        "lines_b": job.get("lines_b", 0),
                        "lines_compared": job.get("total_lines", 0),
                        "elapsed": round(job_elapsed, 2),
                        "node_stats": node_stats_named,
                        "line_details": all_details[:100],
                        "distribution_mode": job.get("distribution_mode", "all"),
                    },
                }
                job["events"].append(done_event)
                print(f"\n  ✅ Job {job_id[:8]} completado: "
                      f"{sim_pct:.2f}% coincidencia en {job_elapsed:.1f}s")

        # Flow control: decrement in-flight and send next chunk
        with self._queue_lock:
            self.in_flight[node_id] = max(0, self.in_flight.get(node_id, 1) - 1)
        self._send_next_chunk(job_id, node_id)

    def _unregister_worker(self, node_id: str):
        """Unregister a worker."""
        with self.workers_lock:
            if node_id in self.workers:
                name = self.workers[node_id].node_name
                self.workers[node_id].connected = False
                print(f"  ⚠ Worker desconectado: {name} ({node_id})")

        self._reassign_chunks(node_id)

    def remove_worker(self, node_id: str) -> bool:
        """Remove a worker completely from registry."""
        with self.workers_lock:
            if node_id in self.workers:
                name = self.workers[node_id].node_name
                # Send kill signal to worker
                try:
                    self.router.send_multipart([
                        node_id.encode(),
                        json.dumps({"type": "KILL", "data": {"reason": "Removed by coordinator"}}).encode(),
                    ])
                except zmq.ZMQError:
                    pass
                del self.workers[node_id]
                print(f"  🗑️ Worker eliminado: {name} ({node_id})")
                self._broadcast_peer_list()
                return True
        return False

    def toggle_worker(self, node_id: str, enabled: bool) -> bool:
        """Enable or disable a worker."""
        with self.workers_lock:
            if node_id in self.workers:
                self.workers[node_id].enabled = enabled
                name = self.workers[node_id].node_name
                state = "activado" if enabled else "desactivado"
                print(f"  {'✅' if enabled else '⏸️'} Worker {state}: {name} ({node_id})")

                # Notify worker of its status
                try:
                    msg = {"type": "STATUS_CHANGE", "data": {"enabled": enabled}}
                    self.router.send_multipart([
                        node_id.encode(),
                        json.dumps(msg).encode(),
                    ])
                except zmq.ZMQError:
                    pass
                return True
        return False

    def configure_worker_gpu(self, node_id: str, work_group_size: int,
                              compute_units: int) -> bool:
        """Configure GPU settings for a worker."""
        with self.workers_lock:
            if node_id in self.workers:
                w = self.workers[node_id]
                if work_group_size > 0:
                    w.gpu_work_group_size = min(work_group_size, w.gpu_max_work_group)
                if compute_units > 0:
                    w.gpu_compute_units_to_use = min(compute_units, w.gpu_compute_units)

                # Notify worker of GPU config
                try:
                    msg = {
                        "type": "GPU_CONFIG",
                        "data": {
                            "work_group_size": w.gpu_work_group_size,
                            "compute_units": w.gpu_compute_units_to_use,
                        },
                    }
                    self.router.send_multipart([
                        node_id.encode(),
                        json.dumps(msg).encode(),
                    ])
                except zmq.ZMQError:
                    pass

                name = w.node_name
                print(f"  ⚙️ GPU config actualizada para {name}: "
                      f"WG={w.gpu_work_group_size}, CUs={w.gpu_compute_units_to_use}")
                return True
        return False

    def configure_worker_cpu(self, node_id: str, cpu_cores: int) -> bool:
        """Configure CPU cores for a worker."""
        with self.workers_lock:
            if node_id in self.workers:
                w = self.workers[node_id]
                if 0 < cpu_cores <= w.cpu_count:
                    w.cpu_cores_to_use = cpu_cores
                else:
                    w.cpu_cores_to_use = w.cpu_count

                # Notify worker of CPU config
                try:
                    msg = {
                        "type": "CPU_CONFIG",
                        "data": {"cpu_cores": w.cpu_cores_to_use},
                    }
                    self.router.send_multipart([
                        node_id.encode(),
                        json.dumps(msg).encode(),
                    ])
                except zmq.ZMQError:
                    pass

                name = w.node_name
                print(f"  ⚙️ CPU config actualizada para {name}: "
                      f"Cores={w.cpu_cores_to_use}/{w.cpu_count}")
                return True
        return False

    def _health_check_loop(self):
        """Periodically check worker health and reassign chunks if needed."""
        while self.running:
            time.sleep(3)
            now = time.time()
            dead_workers = []

            with self.workers_lock:
                for node_id, worker in self.workers.items():
                    if worker.connected and (now - worker.last_heartbeat) > HEARTBEAT_TIMEOUT:
                        # Send a PING before declaring dead
                        try:
                            ping_msg = {
                                "type": "PING",
                                "data": {"ping_id": str(uuid.uuid4())[:8]},
                            }
                            with self._router_lock:
                                self.router.send_multipart([
                                    node_id.encode(),
                                    json.dumps(ping_msg).encode(),
                                ])
                        except zmq.ZMQError:
                            pass
                        # Give extra grace period after ping
                        if (now - worker.last_heartbeat) > HEARTBEAT_TIMEOUT + 5:
                            worker.connected = False
                            dead_workers.append(node_id)
                            print(f"\n  💀 Worker {worker.node_name} ({node_id}) "
                                  f"sin heartbeat por {now - worker.last_heartbeat:.0f}s — marcado como caído")

            for node_id in dead_workers:
                self._reassign_chunks(node_id)

    def _reassign_chunks(self, dead_node_id: str):
        """Reassign chunks from a dead node: put them back in the queue."""
        chunks_to_reassign = []
        for chunk_id, assigned_node in list(self.chunk_assignments.items()):
            if assigned_node == dead_node_id:
                chunks_to_reassign.append(chunk_id)

        if not chunks_to_reassign:
            return

        print(f"  🔄 Reponiendo {len(chunks_to_reassign)} chunks del nodo caído en la cola")

        # Put chunks back in their job's queue
        for chunk_id in chunks_to_reassign:
            del self.chunk_assignments[chunk_id]
            for job_id, pending in self.pending_chunks.items():
                if chunk_id in pending:
                    chunk_data = pending[chunk_id]
                    with self._queue_lock:
                        if job_id not in self.chunks_queue:
                            self.chunks_queue[job_id] = []
                        # Put at front of queue (priority)
                        self.chunks_queue[job_id].insert(0, (chunk_id, chunk_data))
                    break

        # Decrement in-flight for dead node
        with self._queue_lock:
            self.in_flight[dead_node_id] = 0

        # Try to send queued chunks to other available workers
        with self.workers_lock:
            alive_workers = [
                nid for nid, w in self.workers.items()
                if w.connected and w.enabled and nid != dead_node_id
            ]

        for wid in alive_workers:
            for job_id in list(self.chunks_queue.keys()):
                with self._queue_lock:
                    in_flight = self.in_flight.get(wid, 0)
                spots = WINDOW_SIZE - in_flight
                for _ in range(spots):
                    if not self._send_next_chunk(job_id, wid):
                        break

        remaining = sum(
            len(q) for q in self.chunks_queue.values()
        )
        if remaining > 0 and not alive_workers:
            print(f"  ⚠ {remaining} chunks en cola esperando un worker")
        elif remaining > 0:
            print(f"  📦 {remaining} chunks aún en cola, se enviarán bajo demanda")
        else:
            print(f"  ✅ Todos los chunks reasignados exitosamente")

    def _assign_queued_work_to_worker(self, node_id: str):
        """Send queued chunks to a newly connected worker (dynamic join)."""
        with self._queue_lock:
            active_jobs = [
                jid for jid, q in self.chunks_queue.items() if q
            ]

        if not active_jobs:
            return

        with self.workers_lock:
            w = self.workers.get(node_id)
            if not w or not w.connected or not w.enabled:
                return
            name = w.node_name

        for job_id in active_jobs:
            with self._queue_lock:
                self.in_flight[node_id] = 0

            sent = 0
            for _ in range(WINDOW_SIZE):
                if self._send_next_chunk(job_id, node_id):
                    sent += 1
                else:
                    break

            if sent > 0:
                print(f"  🆕 Worker {name} se unió al job {job_id[:8]} — "
                      f"recibió {sent} chunks")

    def _broadcast_loop(self):
        """Send periodic broadcasts to workers."""
        while self.running:
            try:
                # Coordinator heartbeat with state for failover
                self.state_version += 1
                state_data = self._get_coordinator_state()

                self.pub.send_json({
                    "type": "COORDINATOR_HEARTBEAT",
                    "coordinator_id": self.coordinator_id,
                    "timestamp": time.time(),
                    "state_version": self.state_version,
                    "coordinator_addr": f"{self.local_ip}:{self.zmq_port}",
                    "web_port": self.web_port,
                    "state": state_data,
                })
            except zmq.ZMQError:
                pass

            # UDP LAN broadcast for discovery by other coordinators
            try:
                udp_msg = json.dumps({
                    "type": "COORDINATOR_ANNOUNCE",
                    "addr": f"{self.local_ip}:{self.zmq_port}",
                    "web": f"http://{self.local_ip}:{self.web_port}",
                    "coordinator_id": self.coordinator_id,
                }).encode()
                udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                udp_sock.sendto(udp_msg, ("<broadcast>", UDP_DISCOVERY_PORT))
                udp_sock.close()
            except Exception:
                pass

            time.sleep(BROADCAST_INTERVAL)

    def _get_coordinator_state(self) -> dict:
        """Get coordinator state for failover propagation."""
        with self.workers_lock:
            workers_state = {}
            for nid, w in self.workers.items():
                workers_state[nid] = {
                    "node_name": w.node_name,
                    "hostname": w.hostname,
                    "connected": w.connected,
                    "enabled": w.enabled,
                    "gpu_work_group_size": w.gpu_work_group_size,
                    "gpu_compute_units_to_use": w.gpu_compute_units_to_use,
                }

        with self.jobs_lock:
            jobs_summary = {}
            for jid, j in self.jobs.items():
                if j["status"] in ("processing", "distributing"):
                    completed_chunk_ids = list(
                        self.completed_chunks.get(jid, {}).keys()
                    )
                    jobs_summary[jid] = {
                        "status": j["status"],
                        "type": j.get("type", "compare"),
                        "filepath_a": j.get("filepath_a", ""),
                        "filepath_b": j.get("filepath_b", ""),
                        "filepath": j.get("filepath", ""),
                        "distribution_mode": j.get("distribution_mode", "all"),
                        "chunk_size": CHUNK_SIZE,
                        "total_chunks": j.get("total_chunks", 0),
                        "chunks_completed": j.get("chunks_completed", 0),
                        "total_matches": j.get("total_matches", 0),
                        "total_compared": j.get("total_compared", 0),
                        "total_errors": j.get("total_errors", 0),
                        "lines_processed": j.get("lines_processed", 0),
                        "total_lines": j.get("total_lines", 0),
                        "completed_chunk_ids": completed_chunk_ids,
                        "start_time": j.get("start_time", 0),
                    }

        return {
            "workers": workers_state,
            "active_jobs": jobs_summary,
            "pending_chunk_ids": list(self.chunk_assignments.keys()),
        }

    def _broadcast_peer_list(self):
        """Broadcast the current peer list to all workers."""
        with self.workers_lock:
            peers = {
                nid: {
                    "name": w.node_name,
                    "hostname": w.hostname,
                    "connected": w.connected,
                    "enabled": w.enabled,
                }
                for nid, w in self.workers.items()
            }
        try:
            self.pub.send_json({
                "type": "PEER_LIST",
                "data": {"peers": peers},
            })
        except zmq.ZMQError:
            pass

    def _get_available_worker(self, exclude: list = None,
                              target_node: str = None) -> str | None:
        """Get an available worker node ID (only enabled & connected)."""
        exclude = exclude or []
        with self.workers_lock:
            if target_node:
                if (target_node in self.workers and
                        self.workers[target_node].connected and
                        self.workers[target_node].enabled and
                        target_node not in exclude):
                    return target_node
                for nid, w in self.workers.items():
                    if (w.node_name == target_node and w.connected
                            and w.enabled and nid not in exclude):
                        return nid
                return None

            # Get least-busy connected AND enabled worker
            candidates = [
                (nid, w) for nid, w in self.workers.items()
                if w.connected and w.enabled and nid not in exclude
            ]
            if not candidates:
                return None

            candidates.sort(key=lambda x: (x[1].processing, x[1].chunks_processed))
            return candidates[0][0]

    def _send_chunk_to_worker(self, node_id: str, chunk_data: dict):
        """Send a chunk to a specific worker via ROUTER socket."""
        # Attach GPU config from worker settings
        with self.workers_lock:
            w = self.workers.get(node_id)
            if w:
                chunk_data["data"]["gpu_config"] = {
                    "work_group_size": w.gpu_work_group_size,
                    "compute_units": w.gpu_compute_units_to_use,
                }

        try:
            with self._router_lock:
                self.router.send_multipart([
                    node_id.encode(),
                    json.dumps(chunk_data).encode(),
                ])
        except zmq.ZMQError as e:
            print(f"  ❌ Error enviando chunk a {node_id}: {e}")

    def get_active_workers(self) -> list:
        """Get list of active workers as dicts."""
        with self.workers_lock:
            return [w.to_dict() for w in self.workers.values() if w.connected]

    def get_all_workers(self) -> list:
        """Get list of all workers as dicts."""
        with self.workers_lock:
            return [w.to_dict() for w in self.workers.values()]

    # ─── Job Management ────────────────────────────────────────────────────

    def start_compare_job(self, filepath_a: str, filepath_b: str,
                          distribution_mode: str = "all",
                          target_node: str = None,
                          exclude_master: bool = False,
                          chunk_size: int = CHUNK_SIZE) -> dict:
        """
        Start a DNA comparison job.
        File reading is done with CPU, processing distributed to workers (GPU).

        distribution_mode:
          - "all": distribute to all enabled workers
          - "specific": send all to target_node
          - "exclude_master": distribute to all except the first registered node
        """
        path_a = Path(filepath_a).expanduser().resolve()
        path_b = Path(filepath_b).expanduser().resolve()

        for p, name in [(path_a, "A"), (path_b, "B")]:
            if not p.exists():
                raise FileNotFoundError(f"Archivo {name} no encontrado: {p}")
            if not p.is_file():
                raise ValueError(f"La ruta {name} no es un archivo: {p}")

        # Get available workers (connected AND enabled)
        with self.workers_lock:
            active = [
                (nid, w) for nid, w in self.workers.items()
                if w.connected and w.enabled
            ]

        if not active:
            raise RuntimeError("No hay workers conectados y habilitados")

        if distribution_mode == "specific" and target_node:
            available_ids = []
            for nid, w in active:
                if nid == target_node or w.node_name == target_node:
                    available_ids.append(nid)
            if not available_ids:
                raise RuntimeError(f"Nodo '{target_node}' no encontrado o no habilitado")
        elif distribution_mode == "exclude_master":
            if len(active) < 2:
                raise RuntimeError("Se necesitan al menos 2 workers habilitados para excluir el maestro")
            sorted_active = sorted(active, key=lambda x: x[1].registered_at)
            master_id = sorted_active[0][0]
            available_ids = [nid for nid, _ in active if nid != master_id]
        else:
            available_ids = [nid for nid, _ in active]

        job_id = str(uuid.uuid4())

        job = {
            "id": job_id,
            "type": "compare",
            "filepath_a": str(path_a),
            "filepath_b": str(path_b),
            "filename_a": path_a.name,
            "filename_b": path_b.name,
            "distribution_mode": distribution_mode,
            "target_node": target_node,
            "status": "reading",
            "events": [],
            "total_matches": 0,
            "total_compared": 0,
            "chunks_completed": 0,
            "total_chunks": 0,
            "lines_processed": 0,
            "node_stats": {},
            "start_time": time.time(),
            "elapsed": 0,
            "similarity": 0,
        }

        with self.jobs_lock:
            self.jobs[job_id] = job

        job["events"].append({"type": "status", "data": "reading"})

        # Start processing in background
        t = threading.Thread(
            target=self._run_compare_job,
            args=(job_id, path_a, path_b, available_ids, chunk_size),
            daemon=True,
        )
        t.start()

        return {
            "job_id": job_id,
            "filename_a": path_a.name,
            "filename_b": path_b.name,
            "distribution_mode": distribution_mode,
            "workers": available_ids,
        }

    def _run_compare_job(self, job_id: str, path_a: Path, path_b: Path,
                         worker_ids: list, chunk_size: int):
        """Background thread to run a comparison job.
        FILE READING is done with CPU here on the coordinator.
        COMPARISON is sent to workers for GPU processing.
        """
        with self.jobs_lock:
            job = self.jobs[job_id]

        try:
            # CPU reads DNA lines
            print(f"  📖 Leyendo archivos con CPU...")
            read_start = time.time()
            lines_a = self._read_dna_lines(path_a)
            lines_b = self._read_dna_lines(path_b)
            read_elapsed = time.time() - read_start
            print(f"  ✅ Lectura completada en {read_elapsed:.2f}s (CPU)")

            total_a = len(lines_a)
            total_b = len(lines_b)
            min_lines = min(total_a, total_b)

            job["lines_a"] = total_a
            job["lines_b"] = total_b
            job["total_lines"] = min_lines

            job["events"].append({
                "type": "info",
                "data": {
                    "lines_a": total_a,
                    "lines_b": total_b,
                    "lines_compared": min_lines,
                    "workers": worker_ids,
                    "read_time": round(read_elapsed, 2),
                    "read_mode": "CPU",
                },
            })

            if min_lines == 0:
                job["status"] = "error"
                job["events"].append({
                    "type": "error",
                    "data": "Uno de los archivos no tiene líneas de ADN.",
                })
                return

            # Create chunks
            chunks = []
            for i in range(0, min_lines, chunk_size):
                end = min(i + chunk_size, min_lines)
                chunk_id = str(uuid.uuid4())[:12]
                chunk_data = {
                    "type": "CHUNK_COMPARE",
                    "data": {
                        "chunk_id": chunk_id,
                        "job_id": job_id,
                        "chunk_index": len(chunks),
                        "lines_a": lines_a[i:end],
                        "lines_b": lines_b[i:end],
                        "start_line": i,
                        "end_line": end,
                        "total_chunks": 0,
                    },
                }
                chunks.append((chunk_id, chunk_data))

            total_chunks = len(chunks)
            for _, cd in chunks:
                cd["data"]["total_chunks"] = total_chunks

            job["total_chunks"] = total_chunks
            job["status"] = "distributing"
            job["events"].append({
                "type": "status",
                "data": "distributing",
            })

            # Store pending chunks
            self.pending_chunks[job_id] = {}
            for chunk_id, chunk_data in chunks:
                self.pending_chunks[job_id][chunk_id] = chunk_data

            # Distribute chunks with flow control (only WINDOW_SIZE per worker at a time)
            job["status"] = "processing"
            job["events"].append({"type": "status", "data": "processing"})

            # Store ALL chunks in queue
            with self._queue_lock:
                self.chunks_queue[job_id] = list(chunks)  # [(chunk_id, chunk_data), ...]
                # Initialize in-flight counts
                for wid in worker_ids:
                    self.in_flight[wid] = 0

            # Send initial batch (WINDOW_SIZE per worker)
            sent = 0
            for wid in worker_ids:
                for _ in range(WINDOW_SIZE):
                    if self._send_next_chunk(job_id, wid):
                        sent += 1
                    else:
                        break

            print(f"\n  📤 Enviados {sent} chunks iniciales (ventana={WINDOW_SIZE}) "
                  f"para job {job_id[:8]}. Resto se envía bajo demanda.")

        except Exception as e:
            job["status"] = "error"
            job["events"].append({"type": "error", "data": str(e)})
            print(f"  ❌ Error en job {job_id[:8]}: {e}")
            import traceback
            traceback.print_exc()

    def _send_next_chunk(self, job_id: str, node_id: str) -> bool:
        """Send the next pending chunk from the queue to a specific worker.
        Returns True if a chunk was sent, False if queue is empty."""
        with self._queue_lock:
            if job_id not in self.chunks_queue or not self.chunks_queue[job_id]:
                return False

            # Check worker is alive
            with self.workers_lock:
                w = self.workers.get(node_id)
                if not w or not w.connected or not w.enabled:
                    return False

            chunk_id, chunk_data = self.chunks_queue[job_id].pop(0)
            self.in_flight[node_id] = self.in_flight.get(node_id, 0) + 1

        self.chunk_assignments[chunk_id] = node_id
        self._send_chunk_to_worker(node_id, chunk_data)

        with self.workers_lock:
            w = self.workers.get(node_id)
            name = w.node_name if w else node_id

        remaining = 0
        with self._queue_lock:
            remaining = len(self.chunks_queue.get(job_id, []))

        with self.jobs_lock:
            total = self.jobs.get(job_id, {}).get("total_chunks", 0)

        print(f"  📦 Chunk → {name} (en vuelo: {self.in_flight.get(node_id, 0)}, "
              f"pendientes: {remaining}/{total})")
        return True

    def _read_dna_lines(self, filepath: Path) -> list:
        """Read DNA lines from file using CPU, skipping headers and empty lines."""
        lines = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n\r")
                if line.lstrip().startswith(">"):
                    continue
                if not line.strip():
                    continue
                lines.append(line)
        return lines

    def _read_dna_lines_with_rows(self, filepath: Path) -> list:
        """Read DNA lines with row numbers from file using CPU."""
        lines = []
        row = 0
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                row += 1
                line = raw_line.rstrip("\n\r")
                if line.lstrip().startswith(">"):
                    continue
                if not line.strip():
                    continue
                lines.append((row, line))
        return lines

    def _handle_validate_result(self, node_id: str, data: dict):
        """Handle a validation result from a worker."""
        chunk_id = data.get("chunk_id", "")
        job_id = data.get("job_id", "")

        with self.jobs_lock:
            if job_id not in self.jobs:
                return
            job = self.jobs[job_id]

        # Remove from assignments
        if chunk_id in self.chunk_assignments:
            del self.chunk_assignments[chunk_id]

        # Store completed chunk result
        if job_id not in self.completed_chunks:
            self.completed_chunks[job_id] = {}
        self.completed_chunks[job_id][chunk_id] = data

        # Remove from pending
        if job_id in self.pending_chunks and chunk_id in self.pending_chunks[job_id]:
            del self.pending_chunks[job_id][chunk_id]

        chunk_index = data.get("chunk_index", 0)
        total_errors = data.get("total_errors", 0)
        error_details = data.get("error_details", [])
        elapsed = data.get("elapsed", 0)
        lines_processed = data.get("lines_processed", 0)
        gpu_metrics = data.get("gpu_metrics", {})

        with self.jobs_lock:
            job["total_errors"] = job.get("total_errors", 0) + total_errors
            job["chunks_completed"] += 1
            job["lines_processed"] += lines_processed

            # Accumulate error details (limit to 500)
            existing_details = job.get("all_error_details", [])
            remaining = 500 - len(existing_details)
            if remaining > 0:
                existing_details.extend(error_details[:remaining])
            job["all_error_details"] = existing_details

            # Per-node tracking
            if node_id not in job["node_stats"]:
                job["node_stats"][node_id] = {
                    "chunks": 0, "lines": 0, "errors": 0,
                    "time": 0, "gpu_metrics": {},
                }
            job["node_stats"][node_id]["chunks"] += 1
            job["node_stats"][node_id]["lines"] += lines_processed
            job["node_stats"][node_id]["errors"] += total_errors
            job["node_stats"][node_id]["time"] += elapsed
            if gpu_metrics:
                existing = job["node_stats"][node_id].get("gpu_metrics", {})
                existing["total_kernel_ms"] = existing.get("total_kernel_ms", 0) + gpu_metrics.get("kernel_time_ms", 0)
                job["node_stats"][node_id]["gpu_metrics"] = existing

            total_chunks = job.get("total_chunks", 1)
            job_elapsed = time.time() - job["start_time"]

            # Get node names
            node_stats_named = {}
            with self.workers_lock:
                for nid, stats in job["node_stats"].items():
                    w = self.workers.get(nid)
                    name = w.node_name if w else nid
                    node_stats_named[name] = stats

            event = {
                "type": "progress",
                "data": {
                    "chunks_completed": job["chunks_completed"],
                    "total_chunks": total_chunks,
                    "lines_processed": job["lines_processed"],
                    "total_lines": job.get("total_lines", 0),
                    "total_errors": job["total_errors"],
                    "elapsed": round(job_elapsed, 2),
                    "node_stats": node_stats_named,
                },
            }
            job["events"].append(event)

            # Check if job complete
            if job["chunks_completed"] >= total_chunks:
                job["status"] = "done"
                job["elapsed"] = round(job_elapsed, 2)

                # Collect all error details sorted by row
                all_details = []
                for cid in sorted(self.completed_chunks.get(job_id, {}).keys()):
                    cdata = self.completed_chunks[job_id][cid]
                    all_details.extend(cdata.get("error_details", []))
                all_details.sort(key=lambda x: (x.get("row", 0), x.get("col", 0)))

                done_event = {
                    "type": "done",
                    "data": {
                        "job_type": "analyze",
                        "total_errors": job["total_errors"],
                        "total_lines": job.get("total_lines", 0),
                        "lines_processed": job["lines_processed"],
                        "elapsed": round(job_elapsed, 2),
                        "node_stats": node_stats_named,
                        "error_details": all_details[:500],
                    },
                }
                job["events"].append(done_event)
                print(f"\n  ✅ Validación job {job_id[:8]} completada: "
                      f"{job['total_errors']} errores en {job_elapsed:.1f}s")

        # Flow control: decrement in-flight and send next chunk
        with self._queue_lock:
            self.in_flight[node_id] = max(0, self.in_flight.get(node_id, 1) - 1)
        self._send_next_chunk(job_id, node_id)

    # ─── Analyze Job ─────────────────────────────────────────────────────────

    def start_analyze_job(self, filepath: str,
                          distribution_mode: str = "all",
                          target_node: str = None,
                          chunk_size: int = CHUNK_SIZE) -> dict:
        """
        Start a DNA validation/analysis job.
        Detects invalid characters (≠ ACGTN) and reports row/col positions.
        File reading is done with CPU, processing distributed to workers (GPU).
        """
        path = Path(filepath).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
        if not path.is_file():
            raise ValueError(f"La ruta no es un archivo: {path}")

        # Get available workers
        with self.workers_lock:
            active = [
                (nid, w) for nid, w in self.workers.items()
                if w.connected and w.enabled
            ]

        if not active:
            raise RuntimeError("No hay workers conectados y habilitados")

        if distribution_mode == "specific" and target_node:
            available_ids = []
            for nid, w in active:
                if nid == target_node or w.node_name == target_node:
                    available_ids.append(nid)
            if not available_ids:
                raise RuntimeError(f"Nodo '{target_node}' no encontrado")
        elif distribution_mode == "exclude_master":
            sorted_active = sorted(active, key=lambda x: x[1].registered_at)
            master_id = sorted_active[0][0]
            available_ids = [nid for nid, _ in active if nid != master_id]
        else:
            available_ids = [nid for nid, _ in active]

        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "type": "analyze",
            "filepath": str(path),
            "filename": path.name,
            "distribution_mode": distribution_mode,
            "status": "reading",
            "events": [],
            "total_errors": 0,
            "all_error_details": [],
            "chunks_completed": 0,
            "total_chunks": 0,
            "lines_processed": 0,
            "node_stats": {},
            "start_time": time.time(),
            "elapsed": 0,
        }

        with self.jobs_lock:
            self.jobs[job_id] = job

        job["events"].append({"type": "status", "data": "reading"})

        t = threading.Thread(
            target=self._run_analyze_job,
            args=(job_id, path, available_ids, chunk_size),
            daemon=True,
        )
        t.start()

        return {
            "job_id": job_id,
            "filename": path.name,
            "distribution_mode": distribution_mode,
            "workers": available_ids,
        }

    def _run_analyze_job(self, job_id: str, filepath: Path,
                         worker_ids: list, chunk_size: int):
        """Background thread to run a validation/analysis job.
        FILE READING is done with CPU.
        VALIDATION is sent to workers for GPU processing.
        """
        with self.jobs_lock:
            job = self.jobs[job_id]

        try:
            print(f"  📖 Leyendo archivo con CPU para validación...")
            read_start = time.time()
            lines_with_rows = self._read_dna_lines_with_rows(filepath)
            read_elapsed = time.time() - read_start
            total_lines = len(lines_with_rows)
            print(f"  ✅ Lectura completada en {read_elapsed:.2f}s — {total_lines} líneas")

            job["total_lines"] = total_lines
            job["events"].append({
                "type": "info",
                "data": {
                    "total_lines": total_lines,
                    "filename": filepath.name,
                    "workers": worker_ids,
                    "read_time": round(read_elapsed, 2),
                    "read_mode": "CPU",
                },
            })

            if total_lines == 0:
                job["status"] = "error"
                job["events"].append({"type": "error", "data": "El archivo no tiene líneas de ADN."})
                return

            # Create chunks
            chunks = []
            for i in range(0, total_lines, chunk_size):
                end = min(i + chunk_size, total_lines)
                chunk_id = str(uuid.uuid4())[:12]
                batch = lines_with_rows[i:end]
                row_numbers = [r for r, _ in batch]
                lines = [l for _, l in batch]
                chunk_data = {
                    "type": "CHUNK_VALIDATE",
                    "data": {
                        "chunk_id": chunk_id,
                        "job_id": job_id,
                        "chunk_index": len(chunks),
                        "lines": lines,
                        "row_numbers": row_numbers,
                        "start_line": i,
                        "end_line": end,
                        "total_chunks": 0,
                    },
                }
                chunks.append((chunk_id, chunk_data))

            total_chunks = len(chunks)
            for _, cd in chunks:
                cd["data"]["total_chunks"] = total_chunks

            job["total_chunks"] = total_chunks
            job["status"] = "processing"
            job["events"].append({"type": "status", "data": "processing"})

            # Store pending chunks
            self.pending_chunks[job_id] = {}
            for chunk_id, chunk_data in chunks:
                self.pending_chunks[job_id][chunk_id] = chunk_data

            # Distribute with flow control (same as compare job)
            with self._queue_lock:
                self.chunks_queue[job_id] = list(chunks)
                for wid in worker_ids:
                    self.in_flight[wid] = 0

            # Send initial batch
            sent = 0
            for wid in worker_ids:
                for _ in range(WINDOW_SIZE):
                    if self._send_next_chunk(job_id, wid):
                        sent += 1
                    else:
                        break

            print(f"\n  📤 Enviados {sent} chunks de validación iniciales (ventana={WINDOW_SIZE}) "
                  f"para job {job_id[:8]}. Resto bajo demanda.")

        except Exception as e:
            job["status"] = "error"
            job["events"].append({"type": "error", "data": str(e)})
            print(f"  ❌ Error en validación job {job_id[:8]}: {e}")

    def restore_state(self, state_file: str):
        """Restore coordinator state from a failover state file.
        Re-reads original files and redistributes only pending chunks.
        """
        try:
            with open(state_file, "r") as f:
                saved = json.load(f)
            state = saved.get("state", {})
            workers_state = state.get("workers", {})
            active_jobs = state.get("active_jobs", {})
            print(f"  🔄 Restaurando estado de failover...")
            print(f"  💾 {len(workers_state)} workers previos")
            print(f"  💾 {len(active_jobs)} jobs activos a recuperar")
            print(f"  💾 Promovido por nodo: {saved.get('promoted_by', 'unknown')}")

            self._failover_state = state
            Path(state_file).unlink(missing_ok=True)

            if active_jobs:
                # Start recovery in a thread (wait for workers to connect first)
                t = threading.Thread(
                    target=self._recover_jobs, args=(active_jobs,),
                    daemon=True
                )
                t.start()
                print(f"  ⏳ Recuperación de jobs iniciada (esperando workers...)")

        except Exception as e:
            print(f"  ⚠ Error restaurando estado: {e}")
            import traceback
            traceback.print_exc()

    def _recover_jobs(self, active_jobs: dict):
        """Background thread: wait for workers, then re-read files and redistribute."""
        # Wait up to 30s for at least one worker to connect
        for i in range(30):
            time.sleep(1)
            with self.workers_lock:
                connected = [
                    nid for nid, w in self.workers.items()
                    if w.connected and w.enabled
                ]
            if connected:
                print(f"\n  ✅ {len(connected)} worker(s) reconectados — iniciando recuperación")
                break
        else:
            print(f"  ❌ No hay workers después de 30s — recuperación abortada")
            return

        for job_id, job_state in active_jobs.items():
            try:
                self._recover_single_job(job_id, job_state)
            except Exception as e:
                print(f"  ❌ Error recuperando job {job_id[:8]}: {e}")
                import traceback
                traceback.print_exc()

    def _resolve_file(self, filepath: str) -> Path | None:
        """Resolve a file path for failover recovery.
        Tries: 1) absolute path, 2) by filename in project dir, 3) recursive search.
        Handles cross-platform paths (Windows paths on Linux and vice versa)."""
        p = Path(filepath)
        if p.exists():
            return p

        # Extract filename handling both Windows and Unix separators
        # On Linux, Path("C:\\Users\\file.fna").name returns the whole string
        filename = filepath.replace("\\", "/").split("/")[-1]

        # Try by filename in this project's directory
        project_dir = Path(__file__).parent
        local = project_dir / filename
        if local.exists():
            print(f"  📂 Archivo encontrado en directorio local: {local}")
            return local

        # Recursive search in project dir (1 level deep)
        for child in project_dir.iterdir():
            if child.is_file() and child.name == filename:
                print(f"  📂 Archivo encontrado: {child}")
                return child
            if child.is_dir():
                candidate = child / filename
                if candidate.exists():
                    print(f"  📂 Archivo encontrado: {candidate}")
                    return candidate

        return None

    def _recover_single_job(self, old_job_id: str, job_state: dict):
        """Re-read files, create chunks, skip completed, distribute remaining."""
        job_type = job_state.get("type", "compare")
        completed_ids = set(job_state.get("completed_chunk_ids", []))
        chunks_completed = job_state.get("chunks_completed", 0)
        chunk_size = job_state.get("chunk_size", CHUNK_SIZE)

        print(f"\n  🔄 Recuperando job {old_job_id[:8]} (tipo: {job_type})")
        print(f"  📊 Progreso anterior: {chunks_completed} chunks completados")

        if job_type == "compare":
            filepath_a = job_state.get("filepath_a", "")
            filepath_b = job_state.get("filepath_b", "")
            if not filepath_a or not filepath_b:
                print(f"  ❌ Rutas de archivo no disponibles")
                return

            print(f"  📖 Re-leyendo archivos...")
            print(f"     A: {filepath_a}")
            print(f"     B: {filepath_b}")

            path_a = self._resolve_file(filepath_a)
            path_b = self._resolve_file(filepath_b)
            if not path_a or not path_b:
                missing = []
                if not path_a: missing.append(f"A: {filepath_a}")
                if not path_b: missing.append(f"B: {filepath_b}")
                print(f"  ❌ Archivos no encontrados (ni ruta absoluta ni en directorio del proyecto):")
                for m in missing:
                    print(f"     {m}")
                print(f"  💡 Tip: copia los archivos al directorio del proyecto para que el failover los encuentre")
                return

            # Re-read and re-chunk
            lines_a = self._read_dna_lines(path_a)
            lines_b = self._read_dna_lines(path_b)
            total_lines = min(len(lines_a), len(lines_b))

            chunks = []
            for i in range(0, total_lines, chunk_size):
                end = min(i + chunk_size, total_lines)
                chunk_id = str(uuid.uuid4())[:12]
                chunk_data = {
                    "type": "CHUNK_COMPARE",
                    "data": {
                        "chunk_id": chunk_id,
                        "job_id": old_job_id,
                        "chunk_index": len(chunks),
                        "lines_a": lines_a[i:end],
                        "lines_b": lines_b[i:end],
                        "start_line": i,
                        "end_line": end,
                        "total_chunks": 0,
                    },
                }
                chunks.append((chunk_id, chunk_data))

            total_chunks = len(chunks)
            for _, cd in chunks:
                cd["data"]["total_chunks"] = total_chunks

            # Only distribute chunks that were NOT completed
            # Since chunk IDs are new, we skip by index
            pending_chunks = chunks[chunks_completed:]

            print(f"  📊 Total chunks: {total_chunks}, ya completados: {chunks_completed}, "
                  f"pendientes: {len(pending_chunks)}")

            # Create recovered job
            job = {
                "id": old_job_id,
                "type": "compare",
                "filepath_a": str(path_a),
                "filepath_b": str(path_b),
                "filename_a": path_a.name,
                "filename_b": path_b.name,
                "distribution_mode": job_state.get("distribution_mode", "all"),
                "status": "processing",
                "events": [{"type": "status", "data": "recovered"}],
                "total_matches": job_state.get("total_matches", 0),
                "total_compared": job_state.get("total_compared", 0),
                "chunks_completed": chunks_completed,
                "total_chunks": total_chunks,
                "lines_processed": job_state.get("lines_processed", 0),
                "total_lines": total_lines,
                "node_stats": {},
                "start_time": job_state.get("start_time", time.time()),
                "elapsed": 0,
                "similarity": 0,
            }

            with self.jobs_lock:
                self.jobs[old_job_id] = job

            # Store pending chunks
            self.pending_chunks[old_job_id] = {}
            for chunk_id, chunk_data in pending_chunks:
                self.pending_chunks[old_job_id][chunk_id] = chunk_data

            # Get worker IDs
            with self.workers_lock:
                worker_ids = [
                    nid for nid, w in self.workers.items()
                    if w.connected and w.enabled
                ]

            # Distribute via flow control
            with self._queue_lock:
                self.chunks_queue[old_job_id] = list(pending_chunks)
                for wid in worker_ids:
                    self.in_flight[wid] = 0

            sent = 0
            for wid in worker_ids:
                for _ in range(WINDOW_SIZE):
                    if self._send_next_chunk(old_job_id, wid):
                        sent += 1
                    else:
                        break

            print(f"  ✅ Job {old_job_id[:8]} recuperado: {sent} chunks enviados, "
                  f"{len(pending_chunks) - sent} en cola")

        elif job_type == "analyze":
            filepath = job_state.get("filepath", "")
            if not filepath:
                print(f"  ❌ Ruta de archivo no disponible")
                return

            path = self._resolve_file(filepath)
            if not path:
                print(f"  ❌ Archivo no encontrado: {filepath}")
                print(f"  💡 Tip: copia el archivo al directorio del proyecto")
                return

            print(f"  📖 Re-leyendo archivo: {filepath}")
            lines_with_rows = []
            row = 0
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    row += 1
                    line = raw_line.rstrip("\n\r")
                    if line.lstrip().startswith(">"):
                        continue
                    if not line.strip():
                        continue
                    lines_with_rows.append((row, line))

            total_lines = len(lines_with_rows)
            chunks = []
            for i in range(0, total_lines, chunk_size):
                end = min(i + chunk_size, total_lines)
                chunk_id = str(uuid.uuid4())[:12]
                batch = lines_with_rows[i:end]
                row_numbers = [r for r, _ in batch]
                lines = [l for _, l in batch]
                chunk_data = {
                    "type": "CHUNK_VALIDATE",
                    "data": {
                        "chunk_id": chunk_id,
                        "job_id": old_job_id,
                        "chunk_index": len(chunks),
                        "lines": lines,
                        "row_numbers": row_numbers,
                        "start_line": i,
                        "end_line": end,
                        "total_chunks": 0,
                    },
                }
                chunks.append((chunk_id, chunk_data))

            total_chunks = len(chunks)
            for _, cd in chunks:
                cd["data"]["total_chunks"] = total_chunks

            pending_chunks = chunks[chunks_completed:]

            print(f"  📊 Total chunks: {total_chunks}, completados: {chunks_completed}, "
                  f"pendientes: {len(pending_chunks)}")

            job = {
                "id": old_job_id,
                "type": "analyze",
                "filepath": str(path),
                "filename": path.name,
                "status": "processing",
                "events": [{"type": "status", "data": "recovered"}],
                "total_errors": job_state.get("total_errors", 0),
                "chunks_completed": chunks_completed,
                "total_chunks": total_chunks,
                "lines_processed": job_state.get("lines_processed", 0),
                "total_lines": total_lines,
                "node_stats": {},
                "start_time": job_state.get("start_time", time.time()),
                "elapsed": 0,
            }

            with self.jobs_lock:
                self.jobs[old_job_id] = job

            self.pending_chunks[old_job_id] = {}
            for chunk_id, chunk_data in pending_chunks:
                self.pending_chunks[old_job_id][chunk_id] = chunk_data

            with self.workers_lock:
                worker_ids = [
                    nid for nid, w in self.workers.items()
                    if w.connected and w.enabled
                ]

            with self._queue_lock:
                self.chunks_queue[old_job_id] = list(pending_chunks)
                for wid in worker_ids:
                    self.in_flight[wid] = 0

            sent = 0
            for wid in worker_ids:
                for _ in range(WINDOW_SIZE):
                    if self._send_next_chunk(old_job_id, wid):
                        sent += 1
                    else:
                        break

            print(f"  ✅ Job validación {old_job_id[:8]} recuperado: {sent} chunks enviados")

    def stop(self):
        """Stop the coordinator."""
        self.running = False
        # Send SHUTDOWN multiple times to ensure delivery
        for _ in range(3):
            try:
                self.pub.send_json({"type": "SHUTDOWN"})
                time.sleep(0.3)
            except Exception:
                break
        time.sleep(0.5)  # Let messages flush
        try:
            self.router.close()
            self.pub.close()
            self.context.term()
        except Exception:
            pass
        # Clean up active coordinator file (only if we own it)
        try:
            if ACTIVE_COORD_FILE.exists():
                info = json.loads(ACTIVE_COORD_FILE.read_text())
                if info.get("pid") == os.getpid():
                    ACTIVE_COORD_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"\n  🛑 Coordinador detenido")


# ─── Global coordinator instance ──────────────────────────────────────────

coordinator: DistributedCoordinator | None = None


# ─── Flask Routes ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "distributed_index.html",
        coordinator_id=coordinator.coordinator_id if coordinator else "N/A",
        hostname=coordinator.hostname if coordinator else "N/A",
        local_ip=coordinator.local_ip if coordinator else "N/A",
        zmq_port=coordinator.zmq_port if coordinator else 0,
        cpu_count=mp.cpu_count(),
        gpu_available=GPU_AVAILABLE,
        gpu_name=GPU_NAME,
        gpu_compute_units=GPU_COMPUTE_UNITS,
        gpu_max_work_group=GPU_MAX_WORK_GROUP,
        gpu_global_mem=GPU_GLOBAL_MEM,
    )


@app.route("/api/system-info")
def system_info():
    return jsonify({
        "coordinator_id": coordinator.coordinator_id,
        "hostname": coordinator.hostname,
        "local_ip": coordinator.local_ip,
        "zmq_port": coordinator.zmq_port,
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
        },
    })


@app.route("/api/workers")
def get_workers():
    if not coordinator:
        return jsonify({"workers": []})
    return jsonify({
        "workers": coordinator.get_all_workers(),
        "active_count": len(coordinator.get_active_workers()),
    })


@app.route("/api/workers/<node_id>/toggle", methods=["POST"])
def toggle_worker(node_id):
    """Enable or disable a worker node."""
    if not coordinator:
        return jsonify({"error": "Coordinator not initialized"}), 500
    data = request.get_json()
    enabled = data.get("enabled", True)
    if coordinator.toggle_worker(node_id, enabled):
        return jsonify({"ok": True, "node_id": node_id, "enabled": enabled})
    return jsonify({"error": f"Worker {node_id} not found"}), 404


@app.route("/api/workers/<node_id>/remove", methods=["DELETE"])
def remove_worker(node_id):
    """Remove a worker node completely."""
    if not coordinator:
        return jsonify({"error": "Coordinator not initialized"}), 500
    if coordinator.remove_worker(node_id):
        return jsonify({"ok": True, "node_id": node_id})
    return jsonify({"error": f"Worker {node_id} not found"}), 404


@app.route("/api/workers/<node_id>/gpu-config", methods=["POST"])
def configure_gpu(node_id):
    """Configure GPU settings for a worker."""
    if not coordinator:
        return jsonify({"error": "Coordinator not initialized"}), 500
    data = request.get_json()
    wg_size = int(data.get("work_group_size", 0))
    cu = int(data.get("compute_units", 0))
    if coordinator.configure_worker_gpu(node_id, wg_size, cu):
        return jsonify({"ok": True, "node_id": node_id})
    return jsonify({"error": f"Worker {node_id} not found"}), 404


@app.route("/api/workers/<node_id>/cpu-config", methods=["POST"])
def configure_cpu(node_id):
    """Configure CPU cores for a worker."""
    if not coordinator:
        return jsonify({"error": "Coordinator not initialized"}), 500
    data = request.get_json()
    cores = int(data.get("cpu_cores", 0))
    if coordinator.configure_worker_cpu(node_id, cores):
        return jsonify({"ok": True, "node_id": node_id})
    return jsonify({"error": f"Worker {node_id} not found"}), 404


@app.route("/api/compare", methods=["POST"])
def compare():
    if not coordinator:
        return jsonify({"error": "Coordinator not initialized"}), 500

    data = request.get_json()
    filepath_a = data.get("filepath_a", "").strip()
    filepath_b = data.get("filepath_b", "").strip()
    distribution_mode = data.get("distribution_mode", "all").strip()
    target_node = data.get("target_node", "").strip() or None
    chunk_size = int(data.get("chunk_size", CHUNK_SIZE))

    if not filepath_a or not filepath_b:
        return jsonify({"error": "Se requieren ambos archivos."}), 400

    try:
        result = coordinator.start_compare_job(
            filepath_a, filepath_b,
            distribution_mode=distribution_mode,
            target_node=target_node,
            chunk_size=chunk_size,
        )
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Start a distributed DNA validation job (detect invalid characters)."""
    if not coordinator:
        return jsonify({"error": "Coordinator not initialized"}), 500

    data = request.get_json()
    filepath = data.get("filepath", "").strip()
    distribution_mode = data.get("distribution_mode", "all").strip()
    target_node = data.get("target_node", "").strip() or None
    chunk_size = int(data.get("chunk_size", CHUNK_SIZE))

    if not filepath:
        return jsonify({"error": "Se requiere la ruta del archivo."}), 400

    try:
        result = coordinator.start_analyze_job(
            filepath,
            distribution_mode=distribution_mode,
            target_node=target_node,
            chunk_size=chunk_size,
        )
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400


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
            entries.append({
                "name": "..", "path": str(path.parent),
                "is_dir": True, "size": 0,
            })
        for item in sorted(path.iterdir(),
                           key=lambda x: (not x.is_dir(), x.name.lower())):
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


@app.route("/api/stream/<job_id>")
def stream(job_id):
    if not coordinator:
        return jsonify({"error": "Coordinator not initialized"}), 500

    with coordinator.jobs_lock:
        if job_id not in coordinator.jobs:
            return jsonify({"error": "Job not found"}), 404

    def event_stream():
        idx = 0
        while True:
            with coordinator.jobs_lock:
                job = coordinator.jobs.get(job_id)
                if not job:
                    break
                events = job["events"]
                status = job["status"]

            while idx < len(events):
                yield f"data: {json.dumps(events[idx])}\n\n"
                idx += 1

            if status in ("done", "error"):
                break
            time.sleep(0.3)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/api/workers/stream")
def stream_workers():
    """SSE endpoint for real-time worker status updates."""
    if not coordinator:
        return jsonify({"error": "Coordinator not initialized"}), 500

    def worker_stream():
        while True:
            workers = coordinator.get_all_workers()
            active = len([w for w in workers if w["connected"] and w["enabled"]])
            total = len([w for w in workers if w["connected"]])
            yield f"data: {json.dumps({'workers': workers, 'active_count': active, 'total_connected': total})}\n\n"
            time.sleep(1.5)

    return Response(worker_stream(), mimetype="text/event-stream")


# ─── Main ──────────────────────────────────────────────────────────────────

def _check_existing_coordinator(zmq_port: int) -> str | None:
    """Check if there's already an active coordinator.
    Uses: 1) local file, 2) UDP LAN broadcast discovery.
    Returns its address if found, None otherwise."""

    # --- Method 1: Check local file ---
    if ACTIVE_COORD_FILE.exists():
        try:
            info = json.loads(ACTIVE_COORD_FILE.read_text())
            addr = info.get("addr", "")
            pid = info.get("pid", 0)

            if pid:
                try:
                    os.kill(pid, 0)
                except OSError:
                    ACTIVE_COORD_FILE.unlink(missing_ok=True)
                    addr = ""  # Don't return, try UDP too

            if addr:
                # Verify it's actually alive via ZMQ
                ctx = zmq.Context()
                sub = ctx.socket(zmq.SUB)
                sub.setsockopt(zmq.SUBSCRIBE, b"")
                sub.setsockopt(zmq.RCVTIMEO, 2000)
                host, port = addr.rsplit(":", 1)
                sub.connect(f"tcp://{host}:{int(port) + 1}")
                try:
                    msg = sub.recv_json()
                    if msg.get("type") == "COORDINATOR_HEARTBEAT":
                        sub.close()
                        ctx.term()
                        return addr
                except zmq.Again:
                    pass
                sub.close()
                ctx.term()
        except Exception:
            pass

    # --- Method 2: UDP LAN broadcast discovery ---
    print("  🔍 Buscando coordinador existente en la red (4s)...")
    try:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_sock.bind(("", UDP_DISCOVERY_PORT))
        udp_sock.settimeout(4.0)

        deadline = time.time() + 4.0
        while time.time() < deadline:
            try:
                data, sender_addr = udp_sock.recvfrom(4096)
                msg = json.loads(data.decode())
                if msg.get("type") == "COORDINATOR_ANNOUNCE":
                    addr = msg.get("addr", "")
                    if addr:
                        print(f"  📡 Coordinador encontrado vía UDP: {addr} "
                              f"(desde {sender_addr[0]})")
                        udp_sock.close()
                        return addr
            except socket.timeout:
                break
            except Exception:
                continue

        udp_sock.close()
    except OSError as e:
        # Port might be in use by the coordinator itself
        if "Address already in use" in str(e):
            pass
        else:
            print(f"  ⚠ Error en descubrimiento UDP: {e}")
    except Exception:
        pass

    print("  ✅ No se encontró coordinador existente")
    return None


def main():
    global coordinator

    parser = argparse.ArgumentParser(
        prog="dna_distributed_coordinator",
        description="Coordinador distribuido para comparación de ADN v2.0",
    )
    parser.add_argument(
        "--port", "-p", default=5555, type=int,
        help="Puerto ZMQ para comunicación con workers (default: 5555)",
    )
    parser.add_argument(
        "--web-port", "-w", default=5000, type=int,
        help="Puerto para la interfaz web Flask (default: 5000)",
    )
    parser.add_argument(
        "--restore-state", default=None, type=str,
        help="Ruta al archivo de estado para restaurar después de failover",
    )
    args = parser.parse_args()

    # Check if there's already a coordinator running
    existing = _check_existing_coordinator(args.port)
    if existing:
        print(f"\n{'='*60}")
        print(f"  ⚠️  Ya existe un coordinador activo en {existing}")
        print(f"  🔄 Iniciando como WORKER en vez de coordinador...")
        print(f"{'='*60}\n")

        # Import and run as worker
        from dna_distributed_node import WorkerNode
        node = WorkerNode(existing, f"ex-coord-{socket.gethostname()}")

        def signal_handler(sig, frame):
            print("\n  🛑 Señal de interrupción recibida")
            node.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        node.start()
        return

    coordinator = DistributedCoordinator(
        zmq_port=args.port,
        web_port=args.web_port,
    )

    # Restore state from failover if provided
    if args.restore_state:
        coordinator.restore_state(args.restore_state)

    def signal_handler(sig, frame):
        print("\n  🛑 Señal de interrupción recibida")
        coordinator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    coordinator.start()

    # Start Flask
    app.run(
        host="0.0.0.0",
        port=args.web_port,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()