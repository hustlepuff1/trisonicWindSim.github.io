#!/usr/bin/env python3
"""Independent checks of clipping, prefix preservation, and downstream exclusion."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from prepare_domain import HERE, SOURCE, clip_tetra_x0, tetra_measure, write_json


def main():
    out = HERE / "results/cache"
    domain = json.loads((out / "domain_manifest.json").read_text())
    if not domain["complete"]:
        raise ValueError("Domain preparation incomplete")
    original = np.load(SOURCE / "registered_mach.npy", mmap_mode="r").reshape(29, 701, 71, 71)
    current = np.load(out / "registered_mach.npy", mmap_mode="r").reshape(29, 287, 71, 71)
    checks = []
    for i, row in enumerate(domain["cases"]):
        case = row["case_id"]
        geometry = np.load(out / "geometry" / f"{case}.npy", mmap_mode="r")
        v = np.load(out / "terminal" / f"{case}_clipped_source_volumes.npy")
        f = original[i, 286:288].astype(np.float64)
        expected = (v * f).sum(axis=0) / v.sum(axis=0)
        altered = f.copy()
        altered[v == 0] = 1e20
        altered_average = (v * altered).sum(axis=0) / v.sum(axis=0)
        prefix_same = np.array_equal(current[i, :286], original[i, :286])
        terminal_same = np.array_equal(current[i, -1], expected.astype(np.float32))
        no_downstream = np.array_equal(expected, altered_average)
        exact_cap = bool(np.all(geometry[-1, :, :, 0] == 0))
        upstream = bool(np.all(geometry[:-1, :, :, 0] < 0))
        if not all((prefix_same, terminal_same, no_downstream, exact_cap, upstream)):
            raise ValueError(f"Domain preservation/exclusion check failed: {case}")
        checks.append({"case_id": case, "native_prefix_exact": prefix_same,
                       "conservative_terminal_exact_after_float32_cast": terminal_same,
                       "entirely_downstream_terminal_values_have_zero_effect": no_downstream,
                       "geometry_strictly_upstream_before_exact_x0_cap": upstream and exact_cap,
                       "terminal_float32_integrated_mach_abs_difference": float(abs(((current[i, -1] - expected) * v.sum(axis=0)).sum()))})
    rng = np.random.default_rng(314159)
    tetra = rng.normal(size=(10000, 4, 3))
    volume, moment, full = clip_tetra_x0(tetra)
    reflected = tetra.copy()
    reflected[:, :, 0] *= -1
    complementary_volume, complementary_moment, _ = clip_tetra_x0(reflected)
    complementary_moment[:, 0] *= -1
    _, full_moment = tetra_measure(tetra)
    if not np.allclose(volume + complementary_volume, full, rtol=1e-10, atol=1e-12):
        raise ValueError("Clipped tetrahedra do not partition total volume")
    if not np.allclose(moment + complementary_moment, full_moment, rtol=1e-10, atol=1e-12):
        raise ValueError("Clipped tetrahedra do not partition first moments")
    q = np.array([[-.3, 0, 0], [.7, 0, 0], [.7, 1, 0], [-.3, 1, 0],
                  [-.3, 0, 1], [.7, 0, 1], [.7, 1, 1], [-.3, 1, 1]])
    faces = np.array([[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [1,2,6,5], [0,3,7,4]])
    cube_tetra = np.array([[q.mean(axis=0), q[a], q[b], q[c]] for face in faces for a,b,c in [face[[0,1,2]], face[[0,2,3]]]])
    cube_volume, cube_moment, _ = clip_tetra_x0(cube_tetra)
    if not np.isclose(cube_volume.sum(), .3) or not np.allclose(cube_moment.sum(axis=0) / cube_volume.sum(), [-.15, .5, .5]):
        raise ValueError("Analytic cut-cube volume/centroid mismatch")
    weights = np.load(out / "registered_reference_volumes.npy")
    centers = np.load(out / "registered_reference_centers.npy")
    if not np.all(weights > 0) or not np.all(centers[:, 0] <= 0):
        raise ValueError("M17 quadrature contains invalid weights or downstream centroids")
    report = {"passed": True, "case_count": len(checks), "cases": checks,
              "reference_weights_all_positive": True, "reference_centers_all_upstream": True,
              "reference_quarter_volume_m3": float(weights.sum()),
              "random_tetrahedron_count": len(tetra),
              "complement_volume_abs_error_max": float(np.abs(volume + complementary_volume - full).max()),
              "complement_first_moment_abs_error_max": float(np.abs(moment + complementary_moment - full_moment).max()),
              "analytic_cut_cube_volume": float(cube_volume.sum()),
              "analytic_cut_cube_centroid": (cube_moment.sum(axis=0) / cube_volume.sum()).tolist()}
    write_json(out / "domain_verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
