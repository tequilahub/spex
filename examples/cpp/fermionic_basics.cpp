// Fock states and fermionic expectation values with the spex C++ core (no Python needed).
// Build: g++ -std=c++17 -O2 -pthread fermionic_basics.cpp -o fermionic_basics
//
// A State is a sparse map {basis_state: amplitude}. Bit k of the integer is the occupation of
// spin-orbital k, so fock({0, 2}) = 0b101 means orbitals 0 and 2 are occupied.
#define SPEX_NO_PYTHON
#include "../../spex.cpp"

#include <cassert>

uint64_t fock(std::initializer_list<int> orbitals) {
    uint64_t b = 0;
    for (int o : orbitals) b |= 1ULL << o;
    return b;
}

bool close(std::complex<double> a, std::complex<double> b) { return std::abs(a - b) < 1e-12; }

int main() {
    // 1. <phi| a†_0 a_1 |psi>: move one electron from orbital 1 to orbital 0
    State psi{{fock({1}), 1.0}}, phi{{fock({0}), 1.0}};
    FermionTerm hop{{0}, {1}, 1.0};  // creation_idx, annihilation_idx, weight
    std::cout << "operator: " << hop.to_fermion_string() << "\n";
    std::cout << "<0|a†_0 a_1|1> = " << expectation_value_fermionic_term(phi, psi, hop) << "\n";
    assert(close(expectation_value_fermionic_term(phi, psi, hop), 1.0));

    // 2. The same operator written as an OpenFermion-style string
    assert(close(expectation_value_fermionic_term(phi, psi, parse_fermion_string("0^ 1")), 1.0));

    // 3. Fermionic signs: a†_1 a†_0 |vac> = -|11>
    State vac{{fock({}), 1.0}}, both{{fock({0, 1}), 1.0}};
    FermionTerm pair{{0, 1}, {}, 1.0};
    std::cout << "<11|a†_1 a†_0|vac> = " << expectation_value_fermionic_term(both, vac, pair) << "\n";
    assert(close(expectation_value_fermionic_term(both, vac, pair), -1.0));

    // 4. Occupation numbers of a superposition: sqrt(0.7)|orbital 0> + sqrt(0.3)|orbital 1>
    State state{{fock({0}), std::sqrt(0.7)}, {fock({1}), std::sqrt(0.3)}};
    for (int p = 0; p < 2; ++p) {
        auto n_p = parse_fermion_string(std::to_string(p) + "^ " + std::to_string(p));
        std::cout << "<n_" << p << "> = " << expectation_value_fermionic_term(state, state, n_p).real() << "\n";
    }

    // 5. A sum of terms: H = -t (a†_0 a_1 + a†_1 a_0) + U n_0 n_1
    // Careful with raw index vectors: FermionTerm({c0, c1}, {a0, a1}) is the operator
    // a†_c1 a†_c0 a_a1 a_a0 (the vectors are in the order spex applies them), so
    // FermionTerm({0, 1}, {0, 1}) would be -n_0 n_1. parse_fermion_string reads strings literally.
    const double t = 1.0, U = 4.0;
    std::vector<FermionTerm> H = {
        FermionTerm{{0}, {1}, -t},
        FermionTerm{{1}, {0}, -t},
        parse_fermion_string("0^ 1^ 1 0", U),  // n_0 n_1
    };
    std::cout << "<state|H|state> = " << expectation_value_fermionic(state, state, H) << "\n";
    assert(close(expectation_value_fermionic(state, state, H), -2 * t * std::sqrt(0.7 * 0.3)));
    assert(close(expectation_value_fermionic(both, both, H), U));
}
