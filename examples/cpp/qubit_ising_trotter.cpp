// Trotterised dynamics of a transverse-field Ising chain, and a sparse many-qubit state (spex C++ core).
// Build: g++ -std=c++17 -O2 -pthread qubit_ising_trotter.cpp -o qubit_ising_trotter
//
// H = -J sum_i Z_i Z_{i+1} - h sum_i X_i on an open chain. One Trotter step of size dt is a product of
// exp(-i dt c P) factors; apply_exp_pauli(angle) is exp(-i angle/2 P), so angle = 2 * c * dt.
#define SPEX_NO_PYTHON
#include "../../spex.cpp"

#include <cassert>
#include <cstdio>

using Hamiltonian = std::vector<std::pair<ExpPauliTerm, std::complex<double>>>;

const int N = 4;
const double J = 1.0, h = 0.7, T = 1.0;

ExpPauliTerm exp_pauli(std::initializer_list<std::pair<int, char>> paulis, double angle) {
    ExpPauliTerm term;
    for (auto [qubit, pauli] : paulis) term.pauli_map[qubit] = pauli;
    term.angle = angle;
    return term;
}

Hamiltonian H;

State trotter(State state, int steps) {
    const double dt = T / steps;
    std::vector<ExpPauliTerm> step;
    for (const auto& [term, c] : H) {
        step.push_back(term);
        step.back().angle = 2 * c.real() * dt;
    }
    for (int i = 0; i < steps; ++i) state = apply_U(step, state, 1e-14, N);
    return state;
}

int main() {
    for (int i = 0; i < N - 1; ++i) H.push_back({exp_pauli({{i, 'Z'}, {i + 1, 'Z'}}, 0.0), -J});
    for (int i = 0; i < N; ++i) H.push_back({exp_pauli({{i, 'X'}}, 0.0), -h});

    // Compare against a fine-step reference: first-order Trotter makes the infidelity fall like 1/steps^2
    State start = initialize_zero_state(N);
    State reference = trotter(start, 4000);
    double previous = 1.0;
    for (int steps : {10, 100, 1000}) {
        State end = trotter(start, steps);
        double infidelity = 1 - std::pow(std::abs(inner_product(reference, end)), 2);
        std::printf("%5d steps: %2zu amplitudes, infidelity = %.2e\n", steps, end.size(), infidelity);
        assert(std::abs(inner_product(end, end) - 1.0) < 1e-10);  // each factor is unitary
        assert(infidelity < previous / 10);
        previous = infidelity;
    }

    // Sparse states: 30 qubits would need 2^30 amplitudes densely, but 10 rotations create only 2^10
    const int M = 30;
    const double angle = 0.6;
    State state = initialize_zero_state(M);
    for (int q = 0; q < 10; ++q) state = apply_exp_pauli(exp_pauli({{q, 'X'}}, angle), state, 1e-14, M);
    Hamiltonian z0 = {{exp_pauli({{0, 'Z'}}, 0.0), 1.0}};
    std::complex<double> value = expectation_value(state, state, z0, M);
    std::printf("%d qubits: %zu amplitudes, <Z_0> = %.6f (cos(angle) = %.6f)\n", M, state.size(), value.real(), std::cos(angle));
    assert(state.size() == (1u << 10));
    assert(std::abs(value - std::cos(angle)) < 1e-12);
}
