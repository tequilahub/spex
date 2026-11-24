# spex examples

Self-contained scripts that need only spex: no Tequila, Sunrise, PySCF or NumPy. Every example exists in Python and in C++ with the same content and the same output, and checks its own results with asserts. The one exception is the Tequila comparison at the bottom, which is Python only.

| Topic | Python | C++ |
| --- | --- | --- |
| Fock states, `FermionTerm`, fermion strings, fermionic expectation values | [fermionic_basics.py](python/fermionic_basics.py) | [fermionic_basics.cpp](cpp/fermionic_basics.cpp) |
| Excitations, fSWAP, abstract generators | [fermionic_excitations.py](python/fermionic_excitations.py) | [fermionic_excitations.cpp](cpp/fermionic_excitations.cpp) |
| H2 with a UCCD ansatz, checked against exact diagonalisation | [fermionic_vqe_h2.py](python/fermionic_vqe_h2.py) | [fermionic_vqe_h2.cpp](cpp/fermionic_vqe_h2.cpp) |
| Pauli rotations, expectation values, inner products, thresholds | [qubit_basics.py](python/qubit_basics.py) | [qubit_basics.cpp](cpp/qubit_basics.cpp) |
| Trotterised Ising dynamics and a 30-qubit sparse state | [qubit_ising_trotter.py](python/qubit_ising_trotter.py) | [qubit_ising_trotter.cpp](cpp/qubit_ising_trotter.cpp) |
| Same circuits and energies in Tequila and spex, moving an OpenFermion Hamiltonian into spex (needs `tequila-basic` and `pyscf`) | [compare_with_tequila.py](python/compare_with_tequila.py) | n/a |

## Running them

Python (after `pip install spex-tequila`, or `pip install -e .` in a clone):

```
python examples/python/fermionic_basics.py
```

C++ (the examples include `spex.cpp` directly with `SPEX_NO_PYTHON` defined, so no Python or pybind11 is needed, only a C++17 compiler):

```
g++ -std=c++17 -O2 -pthread examples/cpp/fermionic_basics.cpp -o fermionic_basics && ./fermionic_basics
```

Qubit ordering differs between the two sides: fermionic functions put spin-orbital `k` at bit `k`, while Pauli functions put qubit `q` at bit `num_qubits-1-q`. `compare_with_tequila.py` shows how to line them up with Tequila.

CI runs all of them on every push.
