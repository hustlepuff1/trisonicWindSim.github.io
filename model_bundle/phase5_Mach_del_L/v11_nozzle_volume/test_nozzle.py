"""Nozzle-domain checks and proof that the restricted-domain POD is refitted."""
from __future__ import annotations

import unittest

import numpy as np

from train_nozzle import build_grams, region_masks
from volume_rom import fit_from_gram


class NozzleDomainTests(unittest.TestCase):
    def test_regions_partition_requested_domain(self):
        x = np.array([-5.84, -2., -.51, -.5, -.2, 0.])
        regions = region_masks(x)
        self.assertTrue(regions["nozzle_volume"].all())
        upstream = regions["upstream_x_lt_minus_0p5"]
        terminal = regions["terminal_x_minus_0p5_to_0"]
        self.assertTrue((upstream | terminal).all())
        self.assertFalse((upstream & terminal).any())
        with self.assertRaises(ValueError):
            region_masks(np.array([-1., .01]))

    def test_nozzle_basis_does_not_inherit_downstream_dominant_mode(self):
        # The discarded diffuser has huge independent variability. Its mode
        # dominates a rank-one whole-domain fit, but cannot affect a fresh
        # nozzle-only fit. This catches accidentally cropping the old basis.
        X = np.column_stack((np.arange(6.), np.linspace(1.1e5, 2e5, 6)))
        a = np.array([-2., -1., 0., 0., 1., 2.])
        b = np.array([1., -1., 1., -1., 1., -1.])
        nozzle = 2 + .1 * a[:, None] * np.array([[1., 2., 3., 4.]])
        downstream = 1000 + 100 * b[:, None] * np.array([[1., 2., 3., 4.]])
        full = np.column_stack((nozzle, downstream))
        full_gram = full @ full.T / full.shape[1]
        old = fit_from_gram(X, full_gram, max_rank=1)
        regions = region_masks(np.array([-3., -1., -.4, -.1]))
        grams, _ = build_grams(nozzle, np.ones(4), regions, block_size=2)
        fresh = fit_from_gram(X, grams["nozzle_volume"], max_rank=1)
        old_cropped_modes = (old["mode_snapshot_weights"] @ full)[:, :4]
        fresh_modes = fresh["mode_snapshot_weights"] @ nozzle
        reconstructed = fresh["mean_snapshot_weights"] @ nozzle + fresh["training_coefficients"] @ fresh_modes
        np.testing.assert_allclose(reconstructed, nozzle, atol=1e-12)
        self.assertGreater(float(np.linalg.norm(fresh_modes - old_cropped_modes)), .1)


if __name__ == "__main__":
    unittest.main()
