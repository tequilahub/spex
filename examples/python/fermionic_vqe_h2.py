"""H2 (STO-3G, 0.7414 Angstrom) with a one-parameter UCCD ansatz, using spex alone.

The molecular-orbital integrals below were generated once with PySCF (RHF/STO-3G); nothing
here needs tequila, PySCF or numpy. Spin-orbital 2*i + s belongs to spatial orbital i
(s = 0 alpha, s = 1 beta), so the Hartree-Fock state occupies spin-orbitals 0 and 1.
"""
import math

import spex_tequila as spex

E_NUC = 0.7137539936876182
H1 = {0: -1.2524635735648986, 1: -0.4759487152209647}  # h_ii, the only non-zero one-electron integrals
# Two-electron integrals (pq|rs) in chemists' notation; the other index permutations are equal by symmetry
ERI = {
    (0, 0, 0, 0): 0.6744887663568376, (1, 1, 1, 1): 0.6973937674230251,
    (0, 0, 1, 1): 0.6634680964235672, (1, 1, 0, 0): 0.6634680964235672,
    (0, 1, 0, 1): 0.1812888082114959, (0, 1, 1, 0): 0.1812888082114959,
    (1, 0, 0, 1): 0.1812888082114959, (1, 0, 1, 0): 0.1812888082114959,
}
E_HF_PYSCF, E_FCI_PYSCF = -1.116684387085341, -1.137270174660904  # reference values from PySCF


def fock(*orbitals):
    return sum(1 << orb for orb in orbitals)


# H = sum_{p,s} h_pp a†_ps a_ps + 1/2 sum (pq|rs) a†_p,s a†_r,t a_s,t a_q,s   (without E_NUC)
H = []
for p, h in H1.items():
    for s in range(2):
        H.append(spex.parse_fermion_string(f"{2 * p + s}^ {2 * p + s}", h))
for (p, q, r, s), g in ERI.items():
    for sigma in range(2):
        for tau in range(2):
            a, b, c, d = 2 * p + sigma, 2 * r + tau, 2 * s + tau, 2 * q + sigma
            if a != b and c != d:  # a†_a a†_a = 0 and a_c a_c = 0
                H.append(spex.parse_fermion_string(f"{a}^ {b}^ {c} {d}", 0.5 * g))


def energy(state):
    return spex.expectation_value_fermionic(state, state, H).real + E_NUC


hf = {fock(0, 1): 1.0}
print(f"E(HF)  = {energy(hf):.10f}   (PySCF {E_HF_PYSCF:.10f})")
assert abs(energy(hf) - E_HF_PYSCF) < 1e-9

# UCCD: |psi(theta)> = exp(theta/2 (A - A†)) |HF> with A = a†_2 a†_3 a_1 a_0
double = spex.parse_fermion_string("2^ 3^ 1 0", 1.0j)


def vqe_energy(theta):
    return energy(spex.apply_fermion_excitation(hf, double, theta))


# Crude scan over theta, then parabola through the best three points
step = 2 * math.pi / 720
best = min(range(-360, 361), key=lambda k: vqe_energy(k * step))
e0, e1, e2 = (vqe_energy((best + k) * step) for k in (-1, 0, 1))
theta = (best + 0.5 * (e0 - e2) / (e0 - 2 * e1 + e2)) * step
print(f"theta* = {theta:.6f}, E(UCCD) = {vqe_energy(theta):.10f}")

# Exact diagonalisation in the two-determinant space {|HF>, |double>}, using spex for the matrix elements
dets = [{fock(0, 1): 1.0}, {fock(2, 3): 1.0}]
hh = [[spex.expectation_value_fermionic(a, b, H).real + E_NUC * (a == b) for b in dets] for a in dets]
mean, half_gap = (hh[0][0] + hh[1][1]) / 2, (hh[0][0] - hh[1][1]) / 2
e_exact = mean - math.hypot(half_gap, hh[0][1])
print(f"E(FCI) = {e_exact:.10f}   (PySCF {E_FCI_PYSCF:.10f})")
assert abs(e_exact - E_FCI_PYSCF) < 1e-9
assert abs(vqe_energy(theta) - e_exact) < 1e-8  # UCCD is exact for H2 in a minimal basis
