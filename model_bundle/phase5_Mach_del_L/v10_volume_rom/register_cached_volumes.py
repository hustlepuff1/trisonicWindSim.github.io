#!/usr/bin/env python3
"""Register complete native caches without changing any native field value."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from extract_volume import HERE, cases, write_json
from register_volume import recover, NODE_SHAPE, CELL_SHAPE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=HERE / "results/cache")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--reference-case", default="M17")
    args = parser.parse_args()
    out = args.cache
    rows = cases(args.cache / "cases.json")
    ncells = int(np.prod(CELL_SHAPE))
    path = out / "registered_mach.npy"
    mach = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=(len(rows), ncells))
    path = out / "registered_centers.npy"
    centers = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=(len(rows), ncells, 3))
    path = out / "registered_volumes.npy"
    volumes = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float64, shape=(len(rows), ncells))
    path = out / "registered_valid_geometry.npy"
    valid = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=bool, shape=(len(rows), ncells))
    manifest = {"complete": False, "topology_correspondence_verified": False,
                "registration": "bijective mesh-edge-distance material lattice; no field interpolation or deletion",
                "native_connectivity_identical": False, "native_cell_ids_are_corresponding": False,
                "node_shape": list(NODE_SHAPE), "cell_shape": list(CELL_SHAPE),
                "cell_count": ncells, "training_case_count": 27, "heldout_case_count": 2,
                "mach_dtype": "float32", "coordinate_source_dtype": "float64",
                "mach_cache": "registered_mach.npy", "case_metadata": "cases.json",
                "reference_case": args.reference_case,
                "reference_centers": "registered_reference_centers.npy",
                "reference_volumes": "registered_reference_volumes.npy",
                "cases": [],
                "domain": "entire original 1/4 CFD volume, inlet through downstream diffuser",
                "symmetry": "scalar Mach reflected at y=0 and z=0 for visualization only; no training duplication"}
    for i, row in enumerate(rows):
        native = out / "native" / row["case_id"].lower()
        marker = native / "registration.json"
        while True:
            try:
                xyz, faces, left, right, native_mach = [np.load(native / f"{name}.npy", mmap_mode="r") for name in ("nodes", "face_nodes", "left_cells", "right_cells", "mach")]
                break
            except (FileNotFoundError, ValueError, OSError):
                if not args.wait:
                    raise
                time.sleep(2)
        print(f"Register {i + 1}/{len(rows)} {row['case_id']} ...", flush=True)
        if marker.exists():
            audit = json.loads(marker.read_text())
            node_map = np.load(native / "structured_to_native_nodes.npy", mmap_mode="r")
            cell_map = np.load(native / "structured_to_native_cells.npy", mmap_mode="r")
        else:
            node_map, cell_map, audit = recover(xyz, faces, left, right, ncells)
            np.save(native / "structured_to_native_nodes.npy", node_map)
            np.save(native / "structured_to_native_cells.npy", cell_map)
            audit.update(case_id=row["case_id"], row_index=i, raw_sha256=row["source"]["sha256"],
                         node_map_sha256=hashlib.sha256(np.ascontiguousarray(node_map)).hexdigest(),
                         cell_map_sha256=hashlib.sha256(np.ascontiguousarray(cell_map)).hexdigest())
            write_json(marker, audit)
        mach[i] = native_mach[cell_map]
        # Reordering is exact in the exported float32 training representation.
        if not np.array_equal(mach[i], native_mach[cell_map]):
            raise ValueError("Mach values changed during registration")
        while not (native / "geometry.json").exists():
            if not args.wait:
                raise FileNotFoundError(native / "geometry.json")
            time.sleep(2)
        geometry = json.loads((native / "geometry.json").read_text())
        centers[i] = np.load(native / "centers.npy", mmap_mode="r")[cell_map]
        volumes[i] = np.load(native / "volumes.npy", mmap_mode="r")[cell_map]
        valid[i] = np.load(native / "valid_geometry.npy", mmap_mode="r")[cell_map]
        if row["case_id"] == args.reference_case:
            np.save(out / "registered_reference_centers.npy", np.load(native / "centers.npy", mmap_mode="r")[cell_map])
            np.save(out / "registered_reference_volumes.npy", volumes[i])
            np.save(out / "registered_reference_nodes.npy", xyz[node_map])
            np.save(out / "registered_reference_valid_geometry.npy", valid[i])
            manifest["reference_geometry"] = geometry
        audit = {**audit, "training": row["training"], "L_mm": row["L_mm"], "P0_Pa": row["P0_Pa"],
                 "registered_field_exact_value_permutation": True, "geometry": geometry}
        manifest["cases"].append(audit)
        mach.flush()
        centers.flush()
        volumes.flush()
        valid.flush()
        write_json(out / "extraction_manifest.json", manifest)
        print(json.dumps({"case_id": row["case_id"], "elapsed_seconds": audit["elapsed_seconds"], "every_cell_retained": audit["every_cell_retained"]}), flush=True)
    extraction = json.loads((out / "extraction_audit.json").read_text())
    source_audit = {r["case_id"]: r for r in extraction["cases"]}
    if not all(source_audit[r["case_id"]]["raw_sha256_verified"] for r in rows):
        raise ValueError("Missing full source SHA256 verification")
    manifest["complete"] = True
    manifest["topology_correspondence_verified"] = True
    manifest["all_sources_full_sha256_verified"] = True
    manifest["source_topology_audit"] = "extraction_audit.json"
    manifest["reference_volume_weights_positive_and_valid"] = bool(np.all(np.load(out / "registered_reference_volumes.npy") > 0) and np.all(np.load(out / "registered_reference_valid_geometry.npy")))
    write_json(out / "extraction_manifest.json", manifest)


if __name__ == "__main__":
    main()
