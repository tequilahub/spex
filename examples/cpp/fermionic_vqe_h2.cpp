// H2 (STO-3G, 0.7414 Angstrom) with a one-parameter UCCD ansatz, using the spex C++ core alone.
// Build: g++ -std=c++17 -O2 -pthread fermionic_vqe_h2.cpp -o fermionic_vqe_h2
//
// The molecular-orbital integrals below were generated once with PySCF (RHF/STO-3G).
// Spin-orbital 2*i + s belongs to spatial orbital i (s = 0 alpha, s = 1 beta), so the
// Hartree-Fock state occupies spin-orbitals 0 and 1.
#define SPEX_NO_PYTHON
#include "../../spex.cpp"

#include <array>
#include <cassert>
#include <cstdio>
#include <map>

const double E_NUC = 0.7137539936876182;
const double H1[2] = {-1.2524635735648986, -0.4759487152209647};  // h_ii, the only non-zero one-electron integrals
// Two-electron integrals (pq|rs) in chemists' notation; the other index permutations are equal by symmetry
const std::map<std::array<int, 4>, double> ERI = {
    {{0, 0, 0, 0}, 0.6744887663568376}, {{1, 1, 1, 1}, 0.6973937674230251},
    {{0, 0, 1, 1}, 0.6634680964235672}, {{1, 1, 0, 0}, 0.6634680964235672},
    {{0, 1, 0, 1}, 0.1812888082114959}, {{0, 1, 1, 0}, 0.1812888082114959},
    {{1, 0, 0, 1}, 0.1812888082114959}, {{1, 0, 1, 0}, 0.1812888082114959},
};
const double E_HF_PYSCF = -1.116684387085341, E_FCI_PYSCF = -1.137270174660904;  // reference values from PySCF

uint64_t fock(std::initializer_list<int> orbitals) {
    uint64_t b = 0;
    for (int o : orbitals) b |= 1ULL << o;
    return b;
}

std::vector<FermionTerm> H;

double energy(const State& state) {
    return expectation_value_fermionic(state, state, H).real() + E_NUC;
}

int main() {
    // H = sum_{p,s} h_pp a†_ps a_ps + 1/2 sum (pq|rs) a†_p,s a†_r,t a_s,t a_q,s   (without E_NUC)
    for (int p = 0; p < 2; ++p)
        for (int s = 0; s < 2; ++s) {
            std::string i = std::to_string(2 * p + s);
            H.push_back(parse_fermion_string(i + "^ " + i, H1[p]));
        }
    for (const auto& [idx, g] : ERI) {
        auto [p, q, r, s] = idx;
        for (int sigma = 0; sigma < 2; ++sigma)
            for (int tau = 0; tau < 2; ++tau) {
                int a = 2 * p + sigma, b = 2 * r + tau, c = 2 * s + tau, d = 2 * q + sigma;
                if (a != b && c != d)  // a†_a a†_a = 0 and a_c a_c = 0
                    H.push_back(parse_fermion_string(std::to_string(a) + "^ " + std::to_string(b) + "^ " +
                                                     std::to_string(c) + " " + std::to_string(d), 0.5 * g));
            }
    }

    State hf{{fock({0, 1}), 1.0}};
    std::printf("E(HF)  = %.10f   (PySCF %.10f)\n", energy(hf), E_HF_PYSCF);
    assert(std::abs(energy(hf) - E_HF_PYSCF) < 1e-9);

    // UCCD: |psi(theta)> = exp(theta/2 (A - A†)) |HF> with A = a†_2 a†_3 a_1 a_0
    FermionTerm dbl = parse_fermion_string("2^ 3^ 1 0", std::complex<double>(0, 1));
    auto vqe_energy = [&](double theta) { return energy(apply_fermionic_excitation(hf, dbl, theta)); };

    // Crude scan over theta, then parabola through the best three points
    const double step = 2 * std::acos(-1.0) / 720;
    int best = -360;
    for (int k = -360; k <= 360; ++k)
        if (vqe_energy(k * step) < vqe_energy(best * step)) best = k;
    double e0 = vqe_energy((best - 1) * step), e1 = vqe_energy(best * step), e2 = vqe_energy((best + 1) * step);
    double theta = (best + 0.5 * (e0 - e2) / (e0 - 2 * e1 + e2)) * step;
    std::printf("theta* = %.6f, E(UCCD) = %.10f\n", theta, vqe_energy(theta));

    // Exact diagonalisation in the two-determinant space {|HF>, |double>}, using spex for the matrix elements
    std::array<State, 2> dets = {State{{fock({0, 1}), 1.0}}, State{{fock({2, 3}), 1.0}}};
    double hh[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            hh[i][j] = expectation_value_fermionic(dets[i], dets[j], H).real() + (i == j ? E_NUC : 0.0);
    double e_exact = (hh[0][0] + hh[1][1]) / 2 - std::hypot((hh[0][0] - hh[1][1]) / 2, hh[0][1]);
    std::printf("E(FCI) = %.10f   (PySCF %.10f)\n", e_exact, E_FCI_PYSCF);
    assert(std::abs(e_exact - E_FCI_PYSCF) < 1e-9);
    assert(std::abs(vqe_energy(theta) - e_exact) < 1e-8);  // UCCD is exact for H2 in a minimal basis
}
