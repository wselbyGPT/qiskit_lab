// Author: William Selby

#!/usr/bin/env python3
"""
shor_qiskit_generic.py

Generic Shor's algorithm demo in Qiskit for small odd composite N
(e.g., 15, 21, 33, 35, ...), with an optional curses "scope" TUI.

Modes:
  - Normal mode (default): run Shor once, print factors.
  - Scope mode (--scope): repeatedly run the circuit and display an
    ASCII histogram of measurement results, approximate orders r, and
    candidate factors in a live TUI.

For larger N this becomes slow / memory-heavy because modular exponentiation
is implemented as a full permutation matrix on the work register.
"""

import argparse
import math
import random
from math import gcd
from fractions import Fraction
from collections import Counter
import time
import curses

import numpy as np

from qiskit import QuantumCircuit
from qiskit.synthesis import synth_qft_full
from qiskit.circuit.library import UnitaryGate

# Try to use the modern AerSimulator if available, otherwise fall back
# to the older Aer + execute pattern.
try:
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    USE_AER_SIMULATOR = True
except ImportError:
    from qiskit import Aer, execute

    USE_AER_SIMULATOR = False


# ---------------------------------------------------------------------------
# Small helper: primality check
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for k in range(3, r + 1, 2):
        if n % k == 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Quantum circuit building blocks
# ---------------------------------------------------------------------------

def initialize_registers(qc: QuantumCircuit, n_count: int, n_work: int) -> None:
    """
    Prepare:
        - Counting register (qubits 0 .. n_count-1) in uniform superposition.
        - Work register   (qubits n_count .. n_count+n_work-1) in state |1>.
          (integer '1' in little-endian binary).
    """
    # H on all counting qubits
    qc.h(range(n_count))

    # Work register: set LSB of work register to |1>, others stay |0>.
    # Qubit index n_count is the LSB of the work register.
    qc.x(n_count)


def make_modexp_unitary_gate(N: int, a: int, power: int, n_work: int) -> UnitaryGate:
    """
    Build a UnitaryGate on n_work qubits implementing:

        |x> -> | (a^power * x) mod N >  for 0 <= x < N
        |x> -> | x >                    for x >= N  (leave unused states alone)

    where x is interpreted as an integer encoded in little-endian binary
    across the n_work qubits.

    This is a 2^n_work x 2^n_work permutation matrix.
    """
    dim = 2 ** n_work
    U = np.zeros((dim, dim), dtype=complex)

    mul = pow(a, power, N)

    for x in range(dim):
        if x < N:
            y = (mul * x) % N
        else:
            # For states representing integers >= N, just leave them unchanged
            y = x
        U[y, x] = 1.0

    label = f"{a}^{power} mod {N}"
    return UnitaryGate(U, label=label)


def apply_modular_exponentiation(qc: QuantumCircuit, N: int, a: int,
                                 n_count: int, n_work: int) -> None:
    """
    Apply controlled-U operations that implement:

        |x>|y>  ->  |x>| y * a^x mod N >

    using controlled-U^(2^j) for each counting bit j.
    """
    work_qubits = list(range(n_count, n_count + n_work))

    for j in range(n_count):
        power = 2 ** j
        base_gate = make_modexp_unitary_gate(N, a, power, n_work)
        controlled_gate = base_gate.control(1)
        control_qubit = j
        qc.append(controlled_gate, [control_qubit] + work_qubits)


def apply_inverse_qft(qc: QuantumCircuit, n_count: int) -> None:
    """
    Apply the inverse Quantum Fourier Transform to the counting register.

    We synthesize the IQFT directly using the new synthesis API, with
    do_swaps=False so we handle bit order ourselves (via bitstring[::-1]).
    """
    iqft_circ = synth_qft_full(
        num_qubits=n_count,
        do_swaps=False,
        inverse=True,
    )
    qc.compose(iqft_circ, qubits=range(n_count), inplace=True)


def build_shor_circuit(N: int, a: int, n_count: int = None):
    """
    Assemble the full Shor circuit for factoring N with base a.

    - n_work = ceil(log2(N))
    - n_count defaults to 2 * n_work (typical PE choice).

    Layout:
      - Qubits 0 .. n_count-1: counting register
      - Qubits n_count .. n_count+n_work-1: work register
    """
    if N <= 3:
        raise ValueError("N must be > 3.")

    n_work = math.ceil(math.log2(N))

    if n_count is None:
        n_count = 2 * n_work

    total_qubits = n_count + n_work
    qc = QuantumCircuit(total_qubits, n_count)

    initialize_registers(qc, n_count, n_work)
    qc.barrier()

    apply_modular_exponentiation(qc, N, a, n_count, n_work)
    qc.barrier()

    apply_inverse_qft(qc, n_count)

    # Measure the counting register only
    qc.measure(range(n_count), range(n_count))

    return qc, n_count


# ---------------------------------------------------------------------------
# Simulation + classical post-processing
# ---------------------------------------------------------------------------

def run_circuit_and_get_counts(qc: QuantumCircuit, shots: int):
    """Run the circuit on a simulator and return the counts dictionary."""
    if USE_AER_SIMULATOR:
        backend = AerSimulator()
        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()
    else:
        backend = Aer.get_backend("qasm_simulator")
        result = execute(qc, backend=backend, shots=shots).result()
        counts = result.get_counts(qc)
    return counts


def extract_factors_from_counts(counts, a: int, N: int, n_count: int):
    """
    Given measurement counts from the Shor circuit, try to recover non-trivial
    factors of N.

    Steps:
      - Interpret each bitstring as an integer x (with bit-reversal to match
        do_swaps=False).
      - Compute phase = x / 2^n_count.
      - Use continued fractions to approximate phase ~= s / r with
        denominator r <= N.
      - Check if r is a valid order: a^r mod N == 1.
      - If so, compute gcd(a^(r/2) ± 1, N) to get candidate factors.
    """
    candidate_factors = set()
    denom_limit = N  # reasonable limit for order

    for bitstring, freq in counts.items():
        x = int(bitstring[::-1], 2)
        if x == 0:
            continue

        phase = x / (2 ** n_count)
        frac = Fraction(phase).limit_denominator(denom_limit)
        r = frac.denominator

        if r == 0 or r % 2 == 1:
            continue

        if pow(a, r, N) != 1:
            continue

        xr = pow(a, r // 2, N)

        if xr == 1 or xr == N - 1:
            continue

        f1 = gcd(xr - 1, N)
        f2 = gcd(xr + 1, N)

        for f in (f1, f2):
            if 1 < f < N:
                candidate_factors.add(f)

    return sorted(candidate_factors)


# ---------------------------------------------------------------------------
# Curses "scope" UI
# ---------------------------------------------------------------------------

def draw_scope(stdscr, aggregated, factors, N, a, n_count, shots_per_round, iteration):
    """
    Render the ShorScope TUI: title, stats, and an ASCII histogram of the
    most frequent bitstrings, with approximate r estimates.
    """
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    # Title / header
    try:
        title = (
            f"ShorScope  N={N}  a={a}  n_count={n_count}  "
            f"shots/round={shots_per_round}   [q=quit, r=reset]"
        )
        stdscr.addnstr(0, 0, title, max_x - 1, curses.color_pair(1))
    except curses.error:
        pass

    total_shots = sum(aggregated.values())
    info_line = f"Iter={iteration}  total_shots={total_shots}"
    try:
        stdscr.addnstr(1, 0, info_line, max_x - 1, curses.color_pair(3))
        factors_str = ", ".join(str(f) for f in factors) if factors else "None yet"
        stdscr.addnstr(2, 0, f"Candidate factors: {factors_str}", max_x - 1, curses.color_pair(3))
    except curses.error:
        pass

    # Histogram area
    hist_top = 4
    available_rows = max_y - hist_top
    if available_rows <= 0:
        stdscr.refresh()
        return

    if not aggregated:
        try:
            stdscr.addnstr(hist_top, 0, "No data yet...", max_x - 1)
        except curses.error:
            pass
        stdscr.refresh()
        return

    items = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
    items = items[:available_rows]
    max_freq = items[0][1]

    # bitstring + spaces + freq + " r~xxx"
    label_width = n_count + 3 + 6 + 6
    bar_width = max_x - label_width - 1
    if bar_width < 1:
        bar_width = 1

    for i, (bitstring, freq) in enumerate(items):
        row = hist_top + i
        if row >= max_y:
            break

        bar_len = int(freq / max_freq * bar_width) if max_freq > 0 else 0

        # Approximate r for this peak (just for display)
        x = int(bitstring[::-1], 2)
        phase = x / (2 ** n_count)
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator

        label = f"{bitstring:>{n_count}} {freq:6d} r~{r:>3}"

        try:
            stdscr.addnstr(row, 0, label[:max_x - 1])
        except curses.error:
            continue

        if bar_len > 0 and max_x > label_width:
            bar_start = label_width
            bartext = "#" * min(bar_len, max_x - bar_start - 1)
            try:
                stdscr.addnstr(
                    row,
                    bar_start,
                    bartext,
                    max_x - bar_start - 1,
                    curses.color_pair(2),
                )
            except curses.error:
                pass

    stdscr.refresh()


def curses_main(stdscr, args, qc, n_count, N, a):
    """
    Main loop for ShorScope.
    Repeatedly runs the Shor circuit, aggregates counts, and updates the TUI.
    """
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)   # title
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # bars
        curses.init_pair(3, curses.COLOR_YELLOW, -1) # HUD

    shots_per_round = args.scope_shots or args.shots
    aggregated = Counter()
    iteration = 0

    while True:
        iteration += 1

        # Run one round of Shor and merge counts
        counts = run_circuit_and_get_counts(qc, shots_per_round)
        aggregated.update(counts)

        # Extract candidate factors from the aggregated counts
        factors = extract_factors_from_counts(aggregated, a, N, n_count)

        # Draw TUI
        draw_scope(stdscr, aggregated, factors, N, a, n_count, shots_per_round, iteration)

        # Handle keys
        try:
            key = stdscr.getch()
        except curses.error:
            key = -1

        if key == ord('q'):
            break
        elif key == ord('r'):
            aggregated.clear()
            iteration = 0

        time.sleep(args.scope_refresh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic Shor's algorithm demo in Qiskit for small N, with optional curses scope."
    )
    parser.add_argument(
        "-N",
        "--number",
        type=int,
        default=21,
        help="Number to factor (small odd composite, e.g. 15, 21, 33, 35).",
    )
    parser.add_argument(
        "-a",
        "--base",
        type=int,
        default=None,
        help="Base 'a' used in f(x) = a^x mod N.  "
             "If omitted, a random 2 <= a < N with gcd(a, N) = 1 is chosen.",
    )
    parser.add_argument(
        "--ncount",
        type=int,
        default=None,
        help="Number of counting qubits. Default: 2 * ceil(log2(N)).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Number of measurement shots on the simulator (per run in non-scope mode; per round in scope mode if --scope-shots not set).",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Print an ASCII drawing of the quantum circuit (non-scope mode).",
    )
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Print raw measurement counts (non-scope mode).",
    )
    parser.add_argument(
        "--scope",
        action="store_true",
        help="Run curses-based ShorScope TUI (ASCII histogram).",
    )
    parser.add_argument(
        "--scope-refresh",
        type=float,
        default=0.2,
        help="Delay between scope iterations in seconds.",
    )
    parser.add_argument(
        "--scope-shots",
        type=int,
        default=None,
        help="Number of shots per scope iteration (default: --shots).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    N = args.number

    if N <= 3:
        raise SystemExit("N must be > 3.")
    if N % 2 == 0:
        raise SystemExit("N must be odd (even numbers are trivially factored).")
    if is_prime(N):
        raise SystemExit(f"N = {N} is prime; nothing to factor.")

    # Choose base a
    if args.base is None:
        # Pick random a with gcd(a, N) = 1
        while True:
            a = random.randrange(2, N)
            if gcd(a, N) == 1:
                break
    else:
        a = args.base
        if not (2 <= a < N):
            raise SystemExit(f"Base 'a' must satisfy 2 <= a < N (got a={a}).")
        g = gcd(a, N)
        if g != 1 and g != N:
            # Classical shortcut
            print(f"gcd({a}, {N}) = {g} gives a trivial factor without any quantum magic.")
            print(f"Factors: {g} and {N // g}")
            return

    if args.shots <= 0:
        raise SystemExit("Shots must be a positive integer.")

    # Build circuit once (reused in scope mode)
    qc, n_count = build_shor_circuit(N, a, n_count=args.ncount)

    if args.scope:
        # ShorScope TUI mode
        curses.wrapper(curses_main, args, qc, n_count, N, a)
        return

    # One-shot, non-scope mode
    print(f"Running Shor's algorithm demo for N = {N} using base a = {a}")
    print(f"Counting qubits (precision): {n_count}")
    print(f"Shots: {args.shots}")
    print()

    if args.draw:
        print("=== Quantum circuit ===")
        print(qc.draw("text"))
        print()

    counts = run_circuit_and_get_counts(qc, shots=args.shots)

    if args.show_counts:
        print("=== Measurement counts (bitstring: frequency) ===")
        for bitstring, freq in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{bitstring} : {freq}")
        print()

    factors = extract_factors_from_counts(counts, a, N, n_count)

    if factors:
        print(f"Non-trivial factors of {N} found by Shor's algorithm: {factors}")
    else:
        print("No non-trivial factors found in this run.")
        print("Try increasing --shots or tweaking --ncount, or choosing a different base with -a.")


if __name__ == "__main__":
    main()
