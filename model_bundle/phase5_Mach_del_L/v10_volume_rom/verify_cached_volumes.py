#!/usr/bin/env python3
"""Cross-check new 3D fields against the independently retained 2D extracts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from extract_volume import HERE, THESIS_ROOT, cases, sha256, write_json


def main():
    cache = HERE / "results/cache"
    manifest = json.loads((cache / "extraction_manifest.json").read_text())
    if not manifest["complete"] or not manifest["topology_correspondence_verified"]:
        raise ValueError("Registration not complete")
    mach = np.load(cache / "registered_mach.npy", mmap_mode="r")
    volume = np.load(cache / "registered_reference_volumes.npy")
    if not np.all(volume > 0) or not np.isfinite(volume).all():
        raise ValueError("Reference volumes must all be positive and finite")
    checks = []
    for i, row in enumerate(cases()):
        native = cache / "native" / row["case_id"].lower()
        cell_map = np.load(native / "structured_to_native_cells.npy")
        inverse = np.empty(len(cell_map), dtype=np.int32)
        inverse[cell_map] = np.arange(len(cell_map), dtype=np.int32)
        if row["training"]:
            plane = HERE.parent / "v07_flowfield_rom/results/slices" / f"{row['case_id']}_longitudinal.npz"
        else:
            directory = {"M175": "m1p75", "M325": "m3p25"}[row["case_id"]]
            plane = THESIS_ROOT / "validation" / directory / "source_planes" / f"{row['case_id'].lower()}_longitudinal.npz"
        data = np.load(plane)
        logical = inverse[data["cell_ids"]]
        if np.any((logical // 71) % 71 != 0):
            raise ValueError("Retained y=0 plane did not register to material j=0")
        prior = data["Ma"]
        current = mach[i, logical]
        if not np.array_equal(current, prior.astype(np.float32)):
            raise ValueError("New 3D Mach differs from previous source-plane extraction")
        if not np.isfinite(mach[i]).all() or np.any(mach[i] < 0):
            raise ValueError("Nonfinite or negative source Mach")
        checks.append({"case_id": row["case_id"], "compared_source_cells": len(logical),
                       "material_symmetry_plane_verified": True,
                       "exact_match_after_float32_cast": True,
                       "max_float32_quantization_abs_error_on_plane": float(np.max(np.abs(current - prior))),
                       "retained_2d_source_sha256": sha256(plane)})
    output = {"passed": True, "case_count": len(checks), "checks": checks,
              "all_reference_volumes_positive": True,
              "reference_quarter_volume_m3": float(volume.sum()),
              "reference_volume_min_m3": float(volume.min()),
              "retained_source_plane_cell_count": sum(r["compared_source_cells"] for r in checks),
              "outputs_sha256": {name: sha256(cache / name) for name in (
                  "registered_mach.npy", "registered_reference_centers.npy",
                  "registered_reference_volumes.npy", "registered_reference_nodes.npy", "cases.json")}}
    write_json(cache / "verification.json", output)
    print(json.dumps({k: v for k, v in output.items() if k not in {"checks", "outputs_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
