"""Sparse Pauli states with spex alone (no tequila needed).

A state is a dict {basis_state: amplitude}. On the Pauli side qubit q is bit num_qubits-1-q of the
integer (qubit 0 is the leftmost character in state_to_string), so for 2 qubits 0b10 is |q0=1, q1=0>.
apply_exp_pauli(term, state, threshold, num_qubits) applies exp(-i * angle/2 * P) like a Pauli
rotation gate. Amplitudes below `threshold` are dropped; a negative threshold keeps everything.
"""
import math

import spex_tequila as spex

N = 2


def exp_pauli(paulis, angle):
    term = spex.ExpPauliTerm()
    term.pauli_map = paulis  # {qubit: "X" | "Y" | "Z"}
    term.angle = angle
    return term


def close(a, b):
    return abs(a - b) < 1e-12


# 1. Bell-like state: exp(-i pi/4 X0 X1)|00> = (|00> - i|11>)/sqrt(2)
zero = spex.initialize_zero_state(N)
bell = spex.apply_exp_pauli(exp_pauli({0: "X", 1: "X"}, math.pi / 2), zero, 1e-12, N)
print(spex.state_to_string(bell, N), end="")
s2 = 1 / math.sqrt(2)
assert close(bell[0b00], s2) and close(bell[0b11], -1j * s2)

# 2. Expectation values <bell|P|bell>: expectation_value(phi, psi, [(term, coefficient)], num_qubits)
for paulis, expected in [({0: "Z", 1: "Z"}, 1.0), ({0: "X", 1: "X"}, 0.0), ({0: "X", 1: "Y"}, -1.0)]:
    value = spex.expectation_value(bell, bell, [(exp_pauli(paulis, 0.0), 1.0)], N)
    print(f"<{''.join(f'{p}{q}' for q, p in paulis.items())}> = {value.real:+.3f}")
    assert close(value, expected)

# 3. A Hamiltonian is a list of (term, coefficient): H = 0.5 ZZ + 0.25 (ZI + IZ)
H = [
    (exp_pauli({0: "Z", 1: "Z"}, 0.0), 0.5),
    (exp_pauli({0: "Z"}, 0.0), 0.25),
    (exp_pauli({1: "Z"}, 0.0), 0.25),
]
assert close(spex.expectation_value(bell, bell, H, N), 0.5)
assert close(spex.expectation_value_parallel(bell, bell, H, N), 0.5)  # same, multi-threaded

# 4. Inner product <psi|phi>: the overlap of the Bell-like state with |00>
overlap = spex.inner_product(bell, zero)
print("<bell|00> =", overlap)
assert close(overlap, s2)

# 5. apply_U applies a sequence of rotations: Ry on qubit 0, then Rz on qubit 1
U = [exp_pauli({0: "Y"}, math.pi / 2), exp_pauli({1: "Z"}, 0.4)]
out = spex.apply_U(U, zero, 1e-12, N)
assert close(spex.inner_product(out, out), 1.0)  # unitary: the norm stays 1

# 6. The threshold removes numerical noise: a full pi rotation leaves cos(pi/2) ~ 6e-17 behind
flip = exp_pauli({0: "X"}, math.pi)
noisy = spex.apply_exp_pauli(flip, zero, -1.0, N)
clean = spex.apply_exp_pauli(flip, zero, 1e-12, N)
print(f"entries without threshold: {len(noisy)}, with threshold: {len(clean)}")
assert len(noisy) == 2 and len(clean) == 1
