#!/usr/bin/env python3
"""
Script de prueba rápida para el sistema distribuido de ADN.
Genera archivos de prueba y muestra instrucciones.

Uso:
    python test_distributed.py --generate       # Genera archivos de prueba
    python test_distributed.py --start-coord    # Inicia el coordinador
    python test_distributed.py --start-worker   # Inicia un worker
"""

import argparse
import os
import random
import sys
from pathlib import Path

BASES = "ATCG"
BASES_WITH_ERRORS = "ATCGXYZ123"  # Include some invalid chars


def generate_test_files(size_mb: int = 10, output_dir: str = "."):
    """Generate two DNA test files with controlled similarity."""
    output = Path(output_dir)
    file_a = output / "test_dna_a.fna"
    file_b = output / "test_dna_b.fna"

    chars_per_line = 80
    lines_needed = (size_mb * 1024 * 1024) // chars_per_line

    print(f"\n  🧬 Generando archivos de prueba ({size_mb} MB cada uno)")
    print(f"  Líneas por archivo: {lines_needed:,}")

    random.seed(42)

    with open(file_a, "w") as fa, open(file_b, "w") as fb:
        fa.write(">Test DNA Sequence A\n")
        fb.write(">Test DNA Sequence B\n")

        for i in range(lines_needed):
            # Generate base line
            line = "".join(random.choice(BASES) for _ in range(chars_per_line))

            # Create variant with ~85% similarity
            variant = list(line)
            num_mutations = random.randint(5, 20)
            for _ in range(num_mutations):
                pos = random.randint(0, chars_per_line - 1)
                variant[pos] = random.choice(BASES)

            fa.write(line + "\n")
            fb.write("".join(variant) + "\n")

            if (i + 1) % 100000 == 0:
                pct = (i + 1) / lines_needed * 100
                print(f"    {pct:.0f}% ({i+1:,} líneas)")

    size_a = file_a.stat().st_size / (1024 * 1024)
    size_b = file_b.stat().st_size / (1024 * 1024)

    print(f"\n  ✅ Archivos generados:")
    print(f"    {file_a} ({size_a:.1f} MB)")
    print(f"    {file_b} ({size_b:.1f} MB)")
    return str(file_a), str(file_b)


def show_instructions():
    """Show usage instructions."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           🧬 DNA Distributed Checker — Instrucciones            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. GENERAR ARCHIVOS DE PRUEBA:                                  ║
║     python test_distributed.py --generate --size 10              ║
║                                                                  ║
║  2. INICIAR COORDINADOR (Terminal 1):                            ║
║     python dna_distributed_coordinator.py                        ║
║                                                                  ║
║  3. INICIAR WORKERS (Terminales 2, 3, etc.):                    ║
║     python dna_distributed_node.py -c 127.0.0.1:5555 -n nodo1   ║
║     python dna_distributed_node.py -c 127.0.0.1:5555 -n nodo2   ║
║     python dna_distributed_node.py -c 127.0.0.1:5555 -n nodo3   ║
║                                                                  ║
║  4. ABRIR INTERFAZ WEB:                                         ║
║     http://localhost:5000                                        ║
║                                                                  ║
║  MODOS DE DISTRIBUCIÓN:                                          ║
║     • Todos los nodos: reparte chunks entre todos               ║
║     • Nodo específico: envía todo a un solo nodo                ║
║     • Excluir maestro: omite el primer worker registrado        ║
║                                                                  ║
║  TOLERANCIA A FALLOS:                                            ║
║     • Si un worker se desconecta, sus chunks se redistribuyen   ║
║     • Si el coordinador cae, los workers inician elección       ║
║     • Los procesos en curso no se interrumpen                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Script de prueba para DNA Distributed Checker",
    )
    parser.add_argument("--generate", action="store_true",
                        help="Generar archivos de prueba")
    parser.add_argument("--size", type=int, default=10,
                        help="Tamaño de los archivos de prueba en MB (default: 10)")
    parser.add_argument("--output", type=str, default=".",
                        help="Directorio de salida para archivos de prueba")
    args = parser.parse_args()

    if args.generate:
        generate_test_files(args.size, args.output)
    else:
        show_instructions()


if __name__ == "__main__":
    main()
