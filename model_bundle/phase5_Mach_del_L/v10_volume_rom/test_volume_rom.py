"""Numerical tests for the Gram formulation and whole-condition isolation."""
from __future__ import annotations

import unittest

import numpy as np

from volume_rom import (fit_from_gram, project_coefficients, quadratic_error,
                        snapshot_weights)
from train_volume import build_grams


class VolumeRomMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(721)
        cls.F = 2 + rng.normal(size=(7, 3)) @ rng.normal(size=(3, 91)) * .1
        cls.weights = rng.uniform(.1, 2., 91)
        cls.weights /= cls.weights.sum()
        cls.X = np.column_stack((np.linspace(0, 100, 7), np.linspace(1.1e5, 3e5, 7)))
        cls.G = (cls.F * cls.weights) @ cls.F.T
        cls.train = np.array([0, 1, 2, 3, 5, 6])
        cls.model = fit_from_gram(cls.X[cls.train], cls.G[np.ix_(cls.train, cls.train)])

    def test_training_basis_matches_direct_weighted_svd(self):
        F = self.F[self.train]
        mean = self.model["mean_snapshot_weights"] @ F
        modes = self.model["mode_snapshot_weights"] @ F
        np.testing.assert_allclose(mean, F.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose((modes * self.weights) @ modes.T, np.eye(3), atol=3e-12)
        direct_singular = np.linalg.svd((F - mean) * np.sqrt(self.weights), compute_uv=False)
        np.testing.assert_allclose(self.model["details"]["singular_values"][:3], direct_singular[:3], atol=1e-12)
        reconstructed = mean + self.model["training_coefficients"] @ modes
        np.testing.assert_allclose(reconstructed, F, atol=1e-11)

    def test_projection_and_quadratic_error_equal_explicit_field_error(self):
        test = np.array([4])
        coefficients = project_coefficients(self.model, self.G[np.ix_(self.train, test)])
        weights = snapshot_weights(self.model, coefficients)[0]
        prediction = weights @ self.F[self.train]
        modes = self.model["mode_snapshot_weights"] @ self.F[self.train]
        mean = self.model["mean_snapshot_weights"] @ self.F[self.train]
        direct = mean + ((self.F[test[0]] - mean) * self.weights) @ modes.T @ modes
        np.testing.assert_allclose(prediction, direct, atol=2e-12)
        # A deliberately imperfect prediction also exercises nonzero error.
        weights = snapshot_weights(self.model, coefficients * .5)[0]
        expanded = np.zeros(len(self.F))
        expanded[self.train] = weights
        explicit_error = weights @ self.F[self.train] - self.F[test[0]]
        expected = np.sqrt(np.dot(explicit_error ** 2, self.weights))
        computed = quadratic_error(self.G, int(test[0]), expanded)["weighted_rmse_mach"]
        self.assertAlmostEqual(computed, expected, places=12)

    def test_heldout_values_cannot_enter_training_gram(self):
        altered = self.F.copy()
        altered[4] = altered[4] * 1000 + 70
        changed_gram = (altered * self.weights) @ altered.T
        np.testing.assert_array_equal(changed_gram[np.ix_(self.train, self.train)],
                                      self.G[np.ix_(self.train, self.train)])

    def test_block_grams_match_direct_regional_integrals(self):
        regions = {"all": np.ones(self.F.shape[1], dtype=bool),
                   "half": np.arange(self.F.shape[1]) < 40}
        grams, totals = build_grams(self.F, self.weights, regions, block_size=17)
        np.testing.assert_allclose(grams["all"], self.G, atol=2e-14)
        chosen = regions["half"]
        expected = (self.F[:, chosen] * (self.weights[chosen] / totals["half"])) @ self.F[:, chosen].T
        np.testing.assert_allclose(grams["half"], expected, atol=2e-14)

    def test_constant_field_model(self):
        field = np.full((3, 7), 2.1)
        G = field @ field.T / field.shape[1]
        model = fit_from_gram(self.X[:3], G)
        self.assertEqual(model["details"]["rank"], 0)
        self.assertEqual(model["details"]["achieved_energy_fraction"], 1.)


if __name__ == "__main__":
    unittest.main()
