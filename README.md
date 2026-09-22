# SPEX
spex is a sparse-state simulator for [Tequila](https://github.com/tequilahub/tequila), implemented in C++ using Pybind11. It provides

- fermionic excitations, fSWAP and abstract generators on sparse Fock states,
- fermionic expectation values via `FermionTerm` and `expectation_value_fermionic`,
- expectation values, inner products and exponential Pauli operators on sparse Pauli states.

Python 3.10 or newer is required.

# Install
From PyPI:
```
pip install spex-tequila
```

From source with test dependencies (for development):
```
git clone https://github.com/tequilahub/spex.git
cd spex
pip install -e ".[test]"
```