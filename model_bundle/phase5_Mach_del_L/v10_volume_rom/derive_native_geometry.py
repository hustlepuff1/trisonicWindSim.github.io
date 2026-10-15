#!/usr/bin/env python3
"""Consume finished native mesh caches and record geometric precision defects."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from extract_volume import HERE, cases, cell_geometry, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=HERE / "results/cache")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    rows = cases(args.cache / "cases.json")
    nc = rows[0]["source"]["header"]["elements"]
    out = args.cache
    geometry_audit = []
    for row in rows:
        native = out / "native" / row["case_id"].lower()
        marker = native / "geometry.json"
        if marker.exists():
            geometry_audit.append(json.loads(marker.read_text()))
            continue
        while True:
            try:
                xyz, faces, left, right = [np.load(native / f"{name}.npy", mmap_mode="r") for name in ("nodes", "face_nodes", "left_cells", "right_cells")]
                break
            except (FileNotFoundError, ValueError, OSError):
                if not args.wait:
                    raise
                time.sleep(2)
        start = time.time()
        print(f"Geometry {row['case_id']} ...", flush=True)
        centers, volumes, stats, valid = cell_geometry(xyz, faces, left, right, nc)
        np.save(native / "centers.npy", centers)
        np.save(native / "volumes.npy", volumes)
        np.save(native / "valid_geometry.npy", valid)
        stats.update(case_id=row["case_id"], elapsed_seconds=time.time() - start,
                     volume_interpretation="Valid_geometry cells only: triangulated physical polyhedron volume; other values are unsigned-tetra diagnostics, not accepted quadrature.",
                     defects_interpretation="Rounded source coordinates collapse some boundary-layer faces/cells. No coordinate repair or volume floor applied.")
        write_json(marker, stats)
        geometry_audit.append(stats)
        write_json(out / "geometry_audit.json", {"complete": len(geometry_audit) == len(rows), "cases": geometry_audit})
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
