"""Cross-check spex against Tequila: same circuits, same amplitudes, same energies.

This is the only example that needs Tequila (pip install tequila-basic, plus pyscf for the molecule),
and it shows how to move a Tequila/OpenFermion problem into spex.

Qubit ordering: Tequila's integer keys have qubit k at bit k, and so do spex's fermionic functions
(spin-orbital k = bit k). spex's Pauli functions instead put qubit q at bit n-1-q, so below the Pauli
qubit indices are mirrored with `mirror` to make the amplitudes line up.
"""
import math

import tequila as tq
from openfermion import get_fermion_operator

import spex_tequila as spex


def fock(*orbitals):
    return sum(1 << orb for orb in orbitals)


def max_difference(tq_wfn, spex_state):
    tq_state = {int(k): complex(v) for k, v in tq_wfn.items()}
    return max(abs(tq_state.get(k, 0) - spex_state.get(k, 0)) for k in set(tq_state) | set(spex_state))


# --- Fermionic: UCCD double excitation on H2 -------------------------------------------------
mol = tq.Molecule("H 0 0 0\nH 0 0 0.7414", "sto-3g", units="angstrom")

# Tequila/OpenFermion operator -> spex terms: each term is a tuple of (orbital, is_creation) pairs
H_openfermion = get_fermion_operator(mol.make_molecular_hamiltonian())
H = [
    spex.parse_fermion_string(" ".join(f"{i}^" if creation else f"{i}" for i, creation in term), weight)
    for term, weight in H_openfermion.terms.items()  # the empty term is the nuclear repulsion
]

hf = {fock(0, 1): 1.0}
double = spex.FermionTerm([0, 1], [2, 3], 1.0j)
U = tq.gates.X([0, 1]) + mol.make_excitation_gate(indices=[(1, 3), (0, 2)], angle="theta")
E = tq.ExpectationValue(H=mol.make_hamiltonian(), U=U)

print("theta     |amplitude diff|   E(spex)         E(tequila)")
for theta in (0.0, 0.3, -0.226, 1.2):
    state = spex.apply_fermion_excitation(hf, double, theta)
    diff = max_difference(tq.simulate(U, variables={"theta": theta}), state)
    e_spex = spex.expectation_value_fermionic(state, state, H).real
    e_tq = tq.simulate(E, variables={"theta": theta})
    print(f"{theta:6.3f}    {diff:.1e}            {e_spex:.10f}  {e_tq:.10f}")
    assert diff < 1e-9 and abs(e_spex - e_tq) < 1e-9

# Minimise with spex (a scan is enough for one parameter), then ask Tequila for the same point and for FCI
best = min((k * 5e-4 for k in range(-1000, 1001)),
           key=lambda t: spex.expectation_value_fermionic(*[spex.apply_fermion_excitation(hf, double, t)] * 2, H).real)
e_fci = mol.compute_energy("fci")
print(f"best theta = {best:.4f}: E(tequila) = {tq.simulate(E, variables={'theta': best}):.8f}, E(FCI) = {e_fci:.8f}")
assert abs(tq.simulate(E, variables={"theta": best}) - e_fci) < 1e-6

# --- Qubit: a Pauli rotation and a Hamiltonian -----------------------------------------------
theta = 0.8
U = tq.gates.ExpPauli(paulistring="X(0)X(1)", angle=theta) + tq.gates.Ry(angle=0.4, target=0)
pauli_Hamiltonian = 0.5 * tq.paulis.Z(0) * tq.paulis.Z(1) + 0.25 * tq.paulis.X(0) + 0.25 * tq.paulis.Y(1)

n = 2


def mirror(qubit):
    return n - 1 - qubit


xx = spex.ExpPauliTerm()
xx.pauli_map, xx.angle = {mirror(0): "X", mirror(1): "X"}, theta
ry = spex.ExpPauliTerm()
ry.pauli_map, ry.angle = {mirror(0): "Y"}, 0.4  # Ry(a) = exp(-i a/2 Y), the same convention as spex
state = spex.apply_U([xx, ry], spex.initialize_zero_state(n), 1e-14, n)

terms = []
for pauli_string in pauli_Hamiltonian.paulistrings:
    term = spex.ExpPauliTerm()
    term.pauli_map = {mirror(q): p for q, p in pauli_string.items()}  # {qubit: "X" | "Y" | "Z"}
    terms.append((term, complex(pauli_string.coeff)))

diff = max_difference(tq.simulate(U), state)
e_spex = spex.expectation_value(state, state, terms, n).real
e_tq = tq.simulate(tq.ExpectationValue(H=pauli_Hamiltonian, U=U))
print(f"qubit circuit: amplitude diff {diff:.1e}, E(spex) = {e_spex:.10f}, E(tequila) = {e_tq:.10f}")
assert diff < 1e-9 and abs(e_spex - e_tq) < 1e-9
