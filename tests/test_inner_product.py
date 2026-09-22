import unittest

import spex_tequila as spex


class TestInnerProduct(unittest.TestCase):
    """<psi|phi> must conjugate the bra, whichever state holds fewer entries."""

    def _exact(self, psi, phi):
        return sum(v.conjugate() * phi[k] for k, v in psi.items() if k in phi)

    def test_bra_smaller_than_ket(self):
        psi = {0: 1 + 2j, 1: 3 - 1j}
        phi = {0: 0.5 - 1j, 1: 2 + 0.5j, 2: 1 + 0j}
        self.assertAlmostEqual(spex.inner_product(psi, phi), self._exact(psi, phi))

    def test_bra_larger_than_ket(self):
        psi = {0: 1 + 2j, 1: 3 - 1j, 2: 1 + 0j}
        phi = {0: 0.5 - 1j, 1: 2 + 0.5j}
        self.assertAlmostEqual(spex.inner_product(psi, phi), self._exact(psi, phi))

    def test_equal_sizes(self):
        psi = {0: 1 + 2j, 1: 3 - 1j}
        phi = {0: 0.5 - 1j, 1: 2 + 0.5j}
        self.assertAlmostEqual(spex.inner_product(psi, phi), self._exact(psi, phi))

    def test_conjugate_symmetry(self):
        psi = {0: 1 + 2j, 1: 3 - 1j, 2: 0.25j}
        phi = {0: 0.5 - 1j, 1: 2 + 0.5j}
        self.assertAlmostEqual(
            spex.inner_product(psi, phi), spex.inner_product(phi, psi).conjugate()
        )

    def test_expectation_value_matches_manual_sum(self):
        psi = {0b01: 1 + 1j, 0b10: 0.5 - 2j}
        phi = {0b01: 2 - 0.5j, 0b10: 1 + 3j}
        term = spex.ExpPauliTerm()
        term.pauli_map = {0: "Z"}
        got = spex.expectation_value(phi, psi, [(term, 1.0 + 0j)], 2)
        applied = {}
        for basis, amp in psi.items():
            new_basis, phase = spex.apply_Pk(term.pauli_map, basis, 2)
            applied[new_basis] = applied.get(new_basis, 0j) + phase * amp
        self.assertAlmostEqual(got, self._exact(phi, applied))


if __name__ == "__main__":
    unittest.main()
