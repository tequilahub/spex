// Sparse Pauli states with the spex C++ core (no Python needed).
// Build: g++ -std=c++17 -O2 -pthread qubit_basics.cpp -o qubit_basics
//
// A State is a sparse map {basis_state: amplitude}. On the Pauli side qubit q is bit num_qubits-1-q of the
// integer (qubit 0 is the leftmost character in state_to_string), so for 2 qubits 0b10 is |q0=1, q1=0>.
// apply_exp_pauli(term, state, threshold, num_qubits) applies exp(-i * angle/2 * P) like a Pauli
// rotation gate. Amplitudes below `threshold` are dropped; a negative threshold keeps everything.
#define SPEX_NO_PYTHON
#include "../../spex.cpp"

#include <cassert>

using namespace std::complex_literals;

const int N = 2;

ExpPauliTerm exp_pauli(std::initializer_list<std::pair<int, char>> paulis, double angle) {
    ExpPauliTerm term;
    for (auto [qubit, pauli] : paulis) term.pauli_map[qubit] = pauli;  // 'X' | 'Y' | 'Z'
    term.angle = angle;
    return term;
}

bool close(std::complex<double> a, std::complex<double> b) { return std::abs(a - b) < 1e-12; }

int main() {
    const double pi = std::acos(-1.0), s2 = 1 / std::sqrt(2.0);

    // 1. Bell-like state: exp(-i pi/4 X0 X1)|00> = (|00> - i|11>)/sqrt(2)
    State zero = initialize_zero_state(N);
    State bell = apply_exp_pauli(exp_pauli({{0, 'X'}, {1, 'X'}}, pi / 2), zero, 1e-12, N);
    std::cout << state_to_string(bell, N);
    assert(close(bell[0b00], s2) && close(bell[0b11], -1i * s2));

    // 2. Expectation values <bell|P|bell>: expectation_value(phi, psi, {(term, coefficient)}, num_qubits)
    using Hamiltonian = std::vector<std::pair<ExpPauliTerm, std::complex<double>>>;
    Hamiltonian zz = {{exp_pauli({{0, 'Z'}, {1, 'Z'}}, 0.0), 1.0}};
    Hamiltonian xx = {{exp_pauli({{0, 'X'}, {1, 'X'}}, 0.0), 1.0}};
    Hamiltonian xy = {{exp_pauli({{0, 'X'}, {1, 'Y'}}, 0.0), 1.0}};
    std::cout << "<Z0Z1> = " << expectation_value(bell, bell, zz, N).real() << "\n";
    std::cout << "<X0X1> = " << expectation_value(bell, bell, xx, N).real() << "\n";
    std::cout << "<X0Y1> = " << expectation_value(bell, bell, xy, N).real() << "\n";
    assert(close(expectation_value(bell, bell, zz, N), 1.0));
    assert(close(expectation_value(bell, bell, xx, N), 0.0));
    assert(close(expectation_value(bell, bell, xy, N), -1.0));

    // 3. A Hamiltonian is a list of (term, coefficient): H = 0.5 ZZ + 0.25 (ZI + IZ)
    Hamiltonian H = {
        {exp_pauli({{0, 'Z'}, {1, 'Z'}}, 0.0), 0.5},
        {exp_pauli({{0, 'Z'}}, 0.0), 0.25},
        {exp_pauli({{1, 'Z'}}, 0.0), 0.25},
    };
    assert(close(expectation_value(bell, bell, H, N), 0.5));
    assert(close(expectation_value_parallel(bell, bell, H, N), 0.5));  // same, multi-threaded

    // 4. Inner product <psi|phi>: the overlap of the Bell-like state with |00>
    std::complex<double> overlap = inner_product(bell, zero);
    std::cout << "<bell|00> = " << overlap << "\n";
    assert(close(overlap, s2));

    // 5. apply_U applies a sequence of rotations: Ry on qubit 0, then Rz on qubit 1
    std::vector<ExpPauliTerm> U = {exp_pauli({{0, 'Y'}}, pi / 2), exp_pauli({{1, 'Z'}}, 0.4)};
    State out = apply_U(U, zero, 1e-12, N);
    assert(close(inner_product(out, out), 1.0));  // unitary: the norm stays 1

    // 6. The threshold removes numerical noise: a full pi rotation leaves cos(pi/2) ~ 6e-17 behind
    ExpPauliTerm flip = exp_pauli({{0, 'X'}}, pi);
    State noisy = apply_exp_pauli(flip, zero, -1.0, N);
    State clean = apply_exp_pauli(flip, zero, 1e-12, N);
    std::cout << "entries without threshold: " << noisy.size() << ", with threshold: " << clean.size() << "\n";
    assert(noisy.size() == 2 && clean.size() == 1);
}
