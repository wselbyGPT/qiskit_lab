// Author: William Selby

#!/usr/bin/env python3

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def main():
    # 2 qubits (for quantum state), 2 classical bits (for measurement)
    qc = QuantumCircuit(2, 2)

    # Step 1: Put qubit 0 into superposition with a Hadamard gate
    qc.h(0)

    # Step 2: Entangle qubit 1 with qubit 0 using a CNOT gate
    qc.cx(0, 1)

    # Step 3: Measure both qubits into the classical bits
    qc.measure([0, 1], [0, 1])

    # Show the circuit as ASCII art
    print("Quantum circuit:\n")
    print(qc.draw("text"))

    # Use the Aer simulator to run the circuit
    simulator = AerSimulator()

    # Run with 1024 shots (repetitions)
    job = simulator.run(qc, shots=1024)
    result = job.result()
    counts = result.get_counts()

    print("\nMeasurement counts:")
    for bitstring, count in sorted(counts.items()):
        print(f"  {bitstring}: {count}")


if __name__ == "__main__":
    main()
