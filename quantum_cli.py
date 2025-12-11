// Author: William Selby

#!/usr/bin/env python3
"""
Tiny Qiskit CLI to run example quantum circuits *and* algorithms.

Basic circuits:
  python3 quantum_cli.py --circuit bell
  python3 quantum_cli.py --circuit ghz --qubits 4 --shots 4096
  python3 quantum_cli.py --circuit super --shots 200

Algorithms:
  # Deutsch–Jozsa (constant oracle)
  python3 quantum_cli.py --algorithm dj_constant --qubits 3

  # Deutsch–Jozsa (balanced oracle)
  python3 quantum_cli.py --algorithm dj_balanced --qubits 4

  # Bernstein–Vazirani with secret string 1011
  python3 quantum_cli.py --algorithm bv --secret 1011

  # Grover search on 2 qubits, marking |11>
  python3 quantum_cli.py --algorithm grover2 --target 11
"""

import argparse

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# Basic circuit demos
# ---------------------------------------------------------------------

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


CIRCUITS = {
    "bell": build_bell,
    "ghz": build_ghz,
    "super": build_superposition,
}


# ---------------------------------------------------------------------
# Algorithm demos
# ---------------------------------------------------------------------

ALGORITHMS = [
    "dj_constant",   # Deutsch–Jozsa with constant oracle
    "dj_balanced",   # Deutsch–Jozsa with balanced oracle
    "bv",            # Bernstein–Vazirani
    "grover2",       # Grover on 2 qubits
]


def build_deutsch_jozsa(num_qubits: int = 3, balanced: bool = True) -> QuantumCircuit:
    """
    Deutsch–Jozsa algorithm.

    - num_qubits input qubits (x)
    - 1 ancilla qubit (y)
    - If balanced=True: oracle flips output for half of inputs (simple balanced oracle)
    - If balanced=False: oracle is constant (does nothing)
    """
    if num_qubits < 1:
        raise ValueError("Deutsch–Jozsa needs at least 1 input qubit.")

    n = num_qubits
    qc = QuantumCircuit(n + 1, n)

    # Prepare ancilla in |1>
    qc.x(n)

    # Hadamard on all qubits
    for i in range(n + 1):
        qc.h(i)

    # Oracle Uf
    if balanced:
        # Simple balanced oracle:
        # flip ancilla if any input qubit is 1:
        for i in range(n):
            qc.cx(i, n)
    else:
        # Constant f(x) = 0 oracle: do nothing
        pass

    # Hadamard on input qubits again
    for i in range(n):
        qc.h(i)

    # Measure only the input qubits
    qc.measure(range(n), range(n))
    return qc


def build_bernstein_vazirani(secret: str = "1011") -> QuantumCircuit:
    """
    Bernstein–Vazirani algorithm.

    Secret string s (e.g. '1011') is encoded in the oracle.
    Measurement reveals s in a single query (ignoring noise).
    """
    if not secret or any(c not in "01" for c in secret):
        raise ValueError("Secret must be a non-empty bitstring like '1011'.")

    n = len(secret)
    qc = QuantumCircuit(n + 1, n)

    # Prepare ancilla |1>
    qc.x(n)

    # Hadamard on all qubits
    for i in range(n + 1):
        qc.h(i)

    # Oracle Uf_s : flips ancilla if dot(x, s) = 1 (mod 2)
    for i, bit in enumerate(secret):
        if bit == "1":
            qc.cx(i, n)

    # Hadamard on input qubits
    for i in range(n):
        qc.h(i)

    # Measure inputs → should give secret (up to bit order conventions)
    qc.measure(range(n), range(n))
    return qc


def build_grover_2qubit(target: str = "11") -> QuantumCircuit:
    """
    Grover's algorithm on 2 qubits for a single marked element.

    target ∈ {'00', '01', '10', '11'} specifies which basis state is "marked".
    """
    if target not in {"00", "01", "10", "11"}:
        raise ValueError("Grover target must be one of: 00, 01, 10, 11.")

    qc = QuantumCircuit(2, 2)

    # 1) Start in uniform superposition
    qc.h([0, 1])

    # 2) Oracle: phase flip on |target>
    if target == "11":
        qc.cz(0, 1)
    elif target == "00":
        qc.x([0, 1])
        qc.cz(0, 1)
        qc.x([0, 1])
    elif target == "01":
        qc.x(0)
        qc.cz(0, 1)
        qc.x(0)
    elif target == "10":
        qc.x(1)
        qc.cz(0, 1)
        qc.x(1)

    # 3) Diffusion (inversion about the mean) for 2 qubits
    qc.h([0, 1])
    qc.x([0, 1])
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)
    qc.x([0, 1])
    qc.h([0, 1])

    # 4) Measure
    qc.measure([0, 1], [0, 1])
    return qc


def build_algorithm_circuit(args) -> QuantumCircuit:
    """Dispatch helper for algorithm mode."""
    if args.algorithm == "dj_constant":
        return build_deutsch_jozsa(num_qubits=args.qubits, balanced=False)
    if args.algorithm == "dj_balanced":
        return build_deutsch_jozsa(num_qubits=args.qubits, balanced=True)
    if args.algorithm == "bv":
        return build_bernstein_vazirani(secret=args.secret)
    if args.algorithm == "grover2":
        return build_grover_2qubit(target=args.target)

    raise ValueError(f"Unknown algorithm: {args.algorithm}")


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run_circuit(qc: QuantumCircuit, shots: int = 1024, draw_style: str = "text"):
    """Simulate a circuit and print ASCII diagram + measurement counts."""
    simulator = AerSimulator()
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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiny Qiskit CLI to run example quantum circuits and algorithms."
    )

    # Either pick a basic circuit *or* an algorithm
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-c",
        "--circuit",
        choices=sorted(CIRCUITS.keys()),
        help="Which basic circuit demo to run.",
    )
    group.add_argument(
        "-a",
        "--algorithm",
        choices=ALGORITHMS,
        help="Which algorithm demo to run.",
    )

    # Defaults
    parser.set_defaults(circuit="bell", algorithm=None)

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
        help="Number of qubits for GHZ/Deutsch–Jozsa (ignored by others).",
    )
    parser.add_argument(
        "-d",
        "--draw",
        choices=["text", "unicode"],
        default="text",
        help="How to render the circuit diagram in the terminal.",
    )
    parser.add_argument(
        "--secret",
        type=str,
        default="1011",
        help="Secret bitstring for Bernstein–Vazirani (e.g. 1011).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="11",
        help="Target basis state for Grover 2-qubit (00, 01, 10, 11).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.algorithm:
        qc = build_algorithm_circuit(args)
        mode = "algorithm"
        label = args.algorithm
    else:
        circ_name = args.circuit
        if circ_name == "ghz":
            qc = build_ghz(args.qubits)
        else:
            qc = CIRCUITS[circ_name]()
        mode = "circuit"
        label = circ_name

    print(f"\n[+] Running {mode}: {label}")
    if args.algorithm:
        if label.startswith("dj_"):
            print(f"    Type     : Deutsch–Jozsa ({'balanced' if 'balanced' in label else 'constant'} oracle)")
            print(f"    Qubits   : {args.qubits} inputs + 1 ancilla")
        elif label == "bv":
            print(f"    Algorithm: Bernstein–Vazirani")
            print(f"    Secret   : {args.secret}")
        elif label == "grover2":
            print(f"    Algorithm: Grover (2-qubit)")
            print(f"    Target   : {args.target}")
    else:
        if label == "ghz":
            print(f"    Qubits   : {args.qubits}")
        print(f"    Shots    : {args.shots}")

    run_circuit(qc, shots=args.shots, draw_style=args.draw)


if __name__ == "__main__":
    main()
