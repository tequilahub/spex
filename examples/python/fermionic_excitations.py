"""Fermionic excitations, fSWAP and abstract generators on sparse Fock states, with spex alone.

apply_fermion_excitation(state, term, theta) applies exp(-i*theta/2 * (w*A + conj(w)*A†)) with
A = a†... a... given by the FermionTerm. With weight w = 1j this is a real rotation, the
usual unitary-coupled-cluster building block exp(theta/2 * (A - A†)).
"""
import math

import spex_tequila as spex


def fock(*orbitals):
    return sum(1 << orb for orb in orbitals)


def show(title, state, n_orbitals=5):
    print(title)
    print(spex.state_to_string(state, n_orbitals).rstrip() or "  (empty)")


def close(state, expected):
    return all(abs(state.get(k, 0) - expected.get(k, 0)) < 1e-12 for k in set(state) | set(expected))


# 1. Single excitation 0 -> 1: theta = pi/2 splits the electron 50/50, theta = pi moves it
single = spex.FermionTerm([1], [0], 1.0j)  # creation_idx, annihilation_idx, weight
s2 = 1 / math.sqrt(2)
half = spex.apply_fermion_excitation({fock(0): 1.0}, single, math.pi / 2)
show("single excitation, theta = pi/2:", half)
assert close(half, {fock(0): s2, fock(1): s2})
assert close(spex.apply_fermion_excitation({fock(0): 1.0}, single, math.pi), {fock(1): 1.0})

# 2. Double excitation (0, 1) -> (2, 3) on the closed-shell state; orbital 4 is a spectator.
# Multi-index terms are easiest to write as strings: spex reads them literally like OpenFermion.
double = spex.parse_fermion_string("2^ 3^ 1 0", 1.0j)
hf = {fock(0, 1, 4): 1.0}
res = spex.apply_fermion_excitation(hf, double, math.pi / 2)
show("double excitation, theta = pi/2:", res)
assert close(res, {fock(0, 1, 4): s2, fock(2, 3, 4): s2})

# 3. Pauli blocking and empty orbitals: nothing happens
assert close(spex.apply_fermion_excitation({fock(0, 1): 1.0}, single, math.pi / 2), {fock(0, 1): 1.0})
assert close(spex.apply_fermion_excitation({fock(2): 1.0}, single, math.pi / 2), {fock(2): 1.0})

# 4. Jumping over an occupied orbital costs a sign: a†_2 a_0 passes orbital 1
jump = spex.FermionTerm([2], [0], 1.0j)
assert close(spex.apply_fermion_excitation({fock(0): 1.0}, jump, math.pi), {fock(2): 1.0})
assert close(spex.apply_fermion_excitation({fock(0, 1): 1.0}, jump, math.pi), {fock(1, 2): -1.0})

# 5. fSWAP exchanges two orbitals; it picks up -1 only if both are occupied
assert close(spex.apply_fswap({fock(0): 1.0}, 0, 2), {fock(2): 1.0})
assert close(spex.apply_fswap({fock(0, 2): 1.0}, 0, 2), {fock(0, 2): -1.0})
once = spex.apply_fswap({fock(0, 1): 0.6, fock(2): 0.8}, 0, 2)
twice = spex.apply_fswap(once, 0, 2)
assert close(twice, {fock(0, 1): 0.6, fock(2): 0.8})  # self-inverse

# 6. Abstract generator: exp(-i*theta*(sum of terms)) in one call, valid when the terms act on
# disjoint orbitals. Here the alpha (0 -> 2) and beta (1 -> 3) excitations commute.
terms = [spex.FermionTerm([2], [0], 1.0j), spex.FermionTerm([3], [1], 1.0j)]
state = {fock(0, 1): 1.0}
one_call = spex.apply_abstract_generator(state, terms, 0.3)
sequential = state
for term in terms:
    sequential = spex.apply_fermion_excitation(sequential, term, 0.3)
show("abstract generator, theta = 0.3:", one_call, 4)
assert close(one_call, sequential)
norm = spex.inner_product(one_call, one_call)  # unitary: the norm stays 1
assert abs(norm - 1) < 1e-12
