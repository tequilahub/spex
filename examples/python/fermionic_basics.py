"""Fock states and fermionic expectation values with spex alone (no tequila needed).

A state is a plain dict {basis_state: amplitude}. Bit k of the integer is the
occupation of spin-orbital k, so fock(0, 2) = 0b101 means orbitals 0 and 2 are occupied.
"""
import math

import spex_tequila as spex


def fock(*orbitals):
    return sum(1 << orb for orb in orbitals)


def close(a, b):
    return abs(a - b) < 1e-12


# 1. <phi| a†_0 a_1 |psi>: move one electron from orbital 1 to orbital 0
psi = {fock(1): 1.0}
phi = {fock(0): 1.0}
hop = spex.FermionTerm([0], [1], 1.0)  # FermionTerm(creation_idx, annihilation_idx, weight)
print("operator:", hop.to_fermion_string())
print("<0|a†_0 a_1|1> =", spex.expectation_value_fermionic_term(phi, psi, hop))
assert close(spex.expectation_value_fermionic_term(phi, psi, hop), 1.0)

# 2. The same operator written as an OpenFermion-style string
assert close(spex.expectation_value_fermionic_term(phi, psi, spex.parse_fermion_string("0^ 1")), 1.0)

# 3. Fermionic signs: a†_1 a†_0 |vac> = -|11>
vac, both = {fock(): 1.0}, {fock(0, 1): 1.0}
pair = spex.FermionTerm([0, 1], [], 1.0)
print("<11|a†_1 a†_0|vac> =", spex.expectation_value_fermionic_term(both, vac, pair))
assert close(spex.expectation_value_fermionic_term(both, vac, pair), -1.0)

# 4. Occupation numbers of a superposition: sqrt(0.7)|orbital 0> + sqrt(0.3)|orbital 1>
state = {fock(0): math.sqrt(0.7), fock(1): math.sqrt(0.3)}
for p in range(2):
    n_p = spex.parse_fermion_string(f"{p}^ {p}")
    print(f"<n_{p}> =", spex.expectation_value_fermionic_term(state, state, n_p).real)

# 5. A sum of terms. Careful with raw index lists: FermionTerm([c0, c1], [a0, a1]) is the operator
# a†_c1 a†_c0 a_a1 a_a0 (the lists are in the order spex applies them), so
# FermionTerm([0, 1], [0, 1]) would be -n_0 n_1. parse_fermion_string reads strings literally.
# A sum of terms with expectation_value_fermionic: H = -t (a†_0 a_1 + a†_1 a_0) + U n_0 n_1
t, U = 1.0, 4.0
H = [
    spex.FermionTerm([0], [1], -t),
    spex.FermionTerm([1], [0], -t),
    spex.parse_fermion_string("0^ 1^ 1 0", U),  # n_0 n_1, read literally like OpenFermion
]
print("<state|H|state> =", spex.expectation_value_fermionic(state, state, H))
# hopping gives -2t sqrt(0.7*0.3); n_0 n_1 vanishes on one-electron states
assert close(spex.expectation_value_fermionic(state, state, H), -2 * t * math.sqrt(0.7 * 0.3))
print("<11|H|11> =", spex.expectation_value_fermionic(both, both, H))
assert close(spex.expectation_value_fermionic(both, both, H), U)
