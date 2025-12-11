// Author: William Selby

#!/usr/bin/env python3
"""
Tiny Qiskit CLI to run a few example quantum circuits.

Examples:
  python3 quantum_cli.py --circuit bell
  python3 quantum_cli.py --circuit ghz --qubits 4 --shots 4096
  python3 quantum_cli.py --circuit super --shots 200
"""

import argparse

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ----- Circuit builders ----- #

def build_bell() -> QuantumCircuit:
    """2-qubit Bell state."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_ghz(num_qubits: int = 3) -> QuantumCircuit:
    """N-qubit GHZ state: (|000...0> + |111...1>)/sqrt(2)."""
    if num_qubits < 2:
        raise ValueError("GHZ circuit needs at least 2 qubits.")

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def build_superposition() -> QuantumCircuit:
    """Single-qubit Hadamard superposition."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


# Map CLI names to builder functions
CIRCUITS = {
    "bell": build_bell,
    "ghz": build_ghz,
    "super": build_superposition,
}


# ----- Runner ----- #

def run_circuit(qc: QuantumCircuit, shots: int = 1024, draw_style: str = "text"):
    """Simulate a circuit and print ASCII diagram + measurement counts."""
    simulator = AerSimulator()

    # Transpile for the backend
    compiled = transpile(qc, simulator)

    job = simulator.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    print("\nQuantum circuit:\n")
    print(qc.draw(draw_style))

    print(f"\nShots: {shots}")
    print("Measurement counts:")
    for bitstring, count in sorted(counts.items(), key=lambda x: x[0]):
        print(f"  {bitstring}: {count}")


# ----- CLI ----- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiny Qiskit CLI to run some example quantum circuits."
    )
    parser.add_argument(
        "-c",
        "--circuit",
        choices=sorted(CIRCUITS.keys()),
        default="bell",
        help="Which circuit to run.",
    )
    parser.add_argument(
        "-s",
        "--shots",
        type=int,
        default=1024,
        help="Number of shots (repetitions) to run on the simulator.",
    )
    parser.add_argument(
        "-q",
        "--qubits",
        type=int,
        default=3,
        help="Number of qubits for the GHZ circuit (ignored for others).",
    )
    parser.add_argument(
        "-d",
        "--draw",
        choices=["text", "unicode"],
        default="text",
        help="How to render the circuit diagram in the terminal.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build selected circuit
    if args.circuit == "ghz":
        qc = build_ghz(args.qubits)
    else:
        qc = CIRCUITS[args.circuit]()

    print(f"\n[+] Running circuit: {args.circuit}")
    if args.circuit == "ghz":
        print(f"    Qubits: {args.qubits}")
    print(f"    Shots : {args.shots}\n")

    run_circuit(qc, shots=args.shots, draw_style=args.draw)


if __name__ == "__main__":
    main()
