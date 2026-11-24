// Fermionic excitations, fSWAP and abstract generators on sparse Fock states, with the spex C++ core.
// Build: g++ -std=c++17 -O2 -pthread fermionic_excitations.cpp -o fermionic_excitations
//
// apply_fermionic_excitation(state, term, theta) applies exp(-i*theta/2 * (w*A + conj(w)*A†)) with
// A = a†... a... given by the FermionTerm. With weight w = 1i this is a real rotation, the
// usual unitary-coupled-cluster building block exp(theta/2 * (A - A†)).
#define SPEX_NO_PYTHON
#include "../../spex.cpp"

#include <cassert>

using namespace std::complex_literals;

uint64_t fock(std::initializer_list<int> orbitals) {
    uint64_t b = 0;
    for (int o : orbitals) b |= 1ULL << o;
    return b;
}

bool close(const State& a, const State& b) {
    for (const auto& [k, v] : a) {
        auto it = b.find(k);
        if (std::abs(v - (it == b.end() ? 0.0 : it->second)) > 1e-12) return false;
    }
    for (const auto& [k, v] : b)
        if (a.find(k) == a.end() && std::abs(v) > 1e-12) return false;
    return true;
}

int main() {
    const double pi = std::acos(-1.0), s2 = 1 / std::sqrt(2.0);

    // 1. Single excitation 0 -> 1: theta = pi/2 splits the electron 50/50, theta = pi moves it
    FermionTerm single{{1}, {0}, 1i};  // creation_idx, annihilation_idx, weight
    State half = apply_fermionic_excitation({{fock({0}), 1.0}}, single, pi / 2);
    std::cout << "single excitation, theta = pi/2:\n" << state_to_string(half, 5);
    assert(close(half, {{fock({0}), s2}, {fock({1}), s2}}));
    assert(close(apply_fermionic_excitation({{fock({0}), 1.0}}, single, pi), {{fock({1}), 1.0}}));

    // 2. Double excitation (0, 1) -> (2, 3) on the closed-shell state; orbital 4 is a spectator.
    // Multi-index terms are easiest to write as strings: spex reads them literally like OpenFermion.
    FermionTerm dbl = parse_fermion_string("2^ 3^ 1 0", 1i);
    State res = apply_fermionic_excitation({{fock({0, 1, 4}), 1.0}}, dbl, pi / 2);
    std::cout << "double excitation, theta = pi/2:\n" << state_to_string(res, 5);
    assert(close(res, {{fock({0, 1, 4}), s2}, {fock({2, 3, 4}), s2}}));

    // 3. Pauli blocking and empty orbitals: nothing happens
    assert(close(apply_fermionic_excitation({{fock({0, 1}), 1.0}}, single, pi / 2), {{fock({0, 1}), 1.0}}));
    assert(close(apply_fermionic_excitation({{fock({2}), 1.0}}, single, pi / 2), {{fock({2}), 1.0}}));

    // 4. Jumping over an occupied orbital costs a sign: a†_2 a_0 passes orbital 1
    FermionTerm jump{{2}, {0}, 1i};
    assert(close(apply_fermionic_excitation({{fock({0}), 1.0}}, jump, pi), {{fock({2}), 1.0}}));
    assert(close(apply_fermionic_excitation({{fock({0, 1}), 1.0}}, jump, pi), {{fock({1, 2}), -1.0}}));

    // 5. fSWAP exchanges two orbitals; it picks up -1 only if both are occupied
    assert(close(apply_fswap({{fock({0}), 1.0}}, 0, 2), {{fock({2}), 1.0}}));
    assert(close(apply_fswap({{fock({0, 2}), 1.0}}, 0, 2), {{fock({0, 2}), -1.0}}));
    State input{{fock({0, 1}), 0.6}, {fock({2}), 0.8}};
    assert(close(apply_fswap(apply_fswap(input, 0, 2), 0, 2), input));  // self-inverse

    // 6. Abstract generator: exp(-i*theta*(sum of terms)) in one call, valid when the terms act on
    // disjoint orbitals. Here the alpha (0 -> 2) and beta (1 -> 3) excitations commute.
    std::vector<FermionTerm> terms = {FermionTerm{{2}, {0}, 1i}, FermionTerm{{3}, {1}, 1i}};
    State state{{fock({0, 1}), 1.0}};
    State one_call = apply_abstract_generator(state, terms, 0.3);
    State sequential = state;
    for (const auto& term : terms) sequential = apply_fermionic_excitation(sequential, term, 0.3);
    std::cout << "abstract generator, theta = 0.3:\n" << state_to_string(one_call, 4);
    assert(close(one_call, sequential));
    assert(std::abs(inner_product(one_call, one_call) - 1.0) < 1e-12);  // unitary: the norm stays 1
}
