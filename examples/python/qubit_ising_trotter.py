"""Trotterised dynamics of a transverse-field Ising chain, and a sparse many-qubit state (spex alone).

H = -J sum_i Z_i Z_{i+1} - h sum_i X_i on an open chain. One Trotter step of size dt is a product of
exp(-i dt c P) factors; apply_exp_pauli(angle) is exp(-i angle/2 P), so angle = 2 * c * dt.
"""
import math

import spex_tequila as spex

N, J, h, T = 4, 1.0, 0.7, 1.0


def exp_pauli(paulis, angle):
    term = spex.ExpPauliTerm()
    term.pauli_map = paulis
    term.angle = angle
    return term


zz = [(exp_pauli({i: "Z", i + 1: "Z"}, 0.0), -J) for i in range(N - 1)]
x = [(exp_pauli({i: "X"}, 0.0), -h) for i in range(N)]
H = zz + x


def trotter(state, steps):
    dt = T / steps
    step = [exp_pauli(t.pauli_map, 2 * c * dt) for t, c in H]
    for _ in range(steps):
        state = spex.apply_U(step, state, 1e-14, N)
    return state


# Compare against a fine-step reference: first-order Trotter makes the infidelity fall like 1/steps^2
start = spex.initialize_zero_state(N)
reference = trotter(start, 4000)
previous = 1.0
for steps in (10, 100, 1000):
    end = trotter(start, steps)
    infidelity = 1 - abs(spex.inner_product(reference, end)) ** 2
    print(f"{steps:5d} steps: {len(end):2d} amplitudes, infidelity = {infidelity:.2e}")
    assert abs(spex.inner_product(end, end) - 1.0) < 1e-10  # each factor is unitary
    assert infidelity < previous / 10
    previous = infidelity

# Sparse states: 30 qubits would need 2^30 amplitudes densely, but 10 rotations create only 2^10
M, angle = 30, 0.6
state = spex.initialize_zero_state(M)
for q in range(10):
    state = spex.apply_exp_pauli(exp_pauli({q: "X"}, angle), state, 1e-14, M)
z0 = spex.expectation_value(state, state, [(exp_pauli({0: "Z"}, 0.0), 1.0)], M)
print(f"{M} qubits: {len(state)} amplitudes, <Z_0> = {z0.real:.6f} (cos(angle) = {math.cos(angle):.6f})")
assert len(state) == 2**10
assert abs(z0 - math.cos(angle)) < 1e-12
