#!/usr/bin/env python3
"""Extract all native 3D Mach cells and audit exact mesh correspondence.

The 27 canonical conditions and two held-out conditions remain separate.
Every connectivity block is checked byte for byte by its canonical binary
SHA256. Cell centres and volumes are derived from each actual polyhedral mesh,
never from the previously retained 2D slices. Large arrays remain local caches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parents[1]
THESIS_ROOT = FRAMEWORK.parent
sys.path.insert(0, str(FRAMEWORK / "phase4_CFD"))
from audit_reduced_volumes import read_header, _read_block, _skip_comment_lines


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cases(registry_path=None, raw_dir=None):
    """Load an explicit portable registry or construct it from frozen research inputs."""
    candidates = [Path(registry_path)] if registry_path is not None else [
        HERE / "results/cache/cases.json", HERE / "results/source_cases.json"]
    for registry in candidates:
        if not registry.exists():
            if registry_path is not None:
                raise FileNotFoundError(registry)
            continue
        payload = json.loads(registry.read_text())
        rows = payload["cases"] if isinstance(payload, dict) else payload
        if len(rows) != 29 or sum(bool(r["training"]) for r in rows) != 27:
            raise ValueError("Registry must preserve 27 training cases and two held-out cases")
        if [r["case_id"] for r in rows[-2:]] != ["M175", "M325"] or any(r["training"] for r in rows[-2:]):
            raise ValueError("Held-out row membership changed")
        if len({r["case_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate source case")
        if raw_dir is not None:
            for row in rows:
                row["source"]["path"] = str(Path(raw_dir).resolve() / Path(row["source"]["path"]).name)
        return rows
    v05 = HERE.parent / "v05_surrogate_training/results"
    frozen_manifest = json.loads((v05 / "manifest.json").read_text())
    for name in ("training_dataset.csv", "dataset_provenance.json"):
        expected = frozen_manifest["outputs_sha256"].get(name)
        if expected is not None and sha256(v05 / name) != expected:
            raise ValueError(f"Frozen input metadata changed: {name}")
    sources = json.loads((v05 / "dataset_provenance.json").read_text())["sources"]
    with (v05 / "training_dataset.csv").open(encoding="utf-8-sig") as stream:
        inputs = {r["case_id"]: r for r in csv.DictReader(stream)}
    rows = [{"case_id": r["case_id"], "L_mm": float(inputs[r["case_id"]]["L_mm"]),
             "P0_Pa": float(inputs[r["case_id"]]["P0_Pa"]), "training": True,
             "source": r["source"]} for r in sources]
    held = json.loads((THESIS_ROOT / "validation/raw_source_index.json").read_text())["sources"]
    inp = json.loads((THESIS_ROOT / "validation/actual_inputs_confirmed.json").read_text())["cases"]
    by_id = {r["case_id"]: r for r in inp.values()}
    rows.extend({"case_id": r["case_id"], "L_mm": by_id[r["case_id"]]["L_mm"],
                 "P0_Pa": by_id[r["case_id"]]["P0_Pa"], "training": False,
                 "source": r["source"]} for r in held)
    if raw_dir is not None:
        for row in rows:
            row["source"]["path"] = str(Path(raw_dir).resolve() / Path(row["source"]["path"]).name)
    return rows


def read_native(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        h = read_header(stream)
        if h.variables != ("x", "y", "z", "u", "v", "w", "rho", "p", "Ma"):
            raise ValueError("Unexpected variables")
        if h.zone_type != "FEPolyhedron" or h.total_face_nodes != 4 * h.faces:
            raise ValueError("This audited reader requires the observed all-quadrilateral face mesh")
        xyz = np.empty((h.nodes, 3), dtype=np.float64)
        for axis, name in enumerate(("x", "y", "z")):
            xyz[:, axis] = _read_block(stream, h.nodes, name, path)
        for name in ("u", "v", "w", "rho", "p", "Ma"):
            values = _read_block(stream, h.elements, name, path)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: nonfinite {name}")
        mach = values.astype(np.float32)
        _skip_comment_lines(stream)
        counts = np.fromfile(stream, dtype=np.int32, count=h.faces, sep=" ")
        if counts.size != h.faces or np.any(counts != 4):
            raise ValueError("All source faces must be quadrilateral")
        _skip_comment_lines(stream)
        nodes = np.fromfile(stream, dtype=np.int32, count=h.total_face_nodes, sep=" ")
        _skip_comment_lines(stream)
        left = np.fromfile(stream, dtype=np.int32, count=h.faces, sep=" ")
        _skip_comment_lines(stream)
        right = np.fromfile(stream, dtype=np.int32, count=h.faces, sep=" ")
        if nodes.size != h.total_face_nodes or left.size != h.faces or right.size != h.faces:
            raise ValueError("Incomplete connectivity")
    if not np.isfinite(xyz).all() or np.any(mach < 0):
        raise ValueError("Invalid coordinates or Mach values")
    if nodes.min() < 1 or nodes.max() > h.nodes or min(left.min(), right.min()) < 0 or max(left.max(), right.max()) > h.elements:
        raise ValueError("Connectivity index out of range")
    digest = hashlib.sha256()
    parts = {}
    for name, value in (("face_counts", counts), ("face_nodes", nodes), ("left_cells", left), ("right_cells", right)):
        canonical = np.ascontiguousarray(value, dtype="<i4")
        parts[name] = hashlib.sha256(canonical).hexdigest()
        digest.update(canonical)
    nodes -= 1
    left -= 1
    right -= 1
    return h, xyz, mach, nodes.reshape(-1, 4), left, right, digest.hexdigest(), parts


def cell_geometry(xyz, faces, left, right, ncells, chunk=400_000):
    """Boundary-integral volumes about each cell's face-centre average.

    All source faces are triangulated along diagonal (0,2). A face's outward
    direction is resolved against the cell interior point. The arithmetic mean
    of the six quadrilateral face centres equals the eight-vertex mean for a
    standard hexahedron. Adjacent-cell orientation consistency is audited.
    """
    centers = np.zeros((ncells, 3), dtype=np.float64)
    incidences = np.zeros(ncells, dtype=np.int32)
    for start in range(0, len(faces), chunk):
        end = min(start + chunk, len(faces))
        center = xyz[faces[start:end]].mean(axis=1)
        for owner in (left[start:end], right[start:end]):
            keep = owner >= 0
            np.add.at(centers, owner[keep], center[keep])
            np.add.at(incidences, owner[keep], 1)
    if np.any(incidences < 4):
        raise ValueError("Cells without closed polyhedral face topology")
    centers /= incidences[:, None]
    volumes = np.zeros(ncells, dtype=np.float64)
    opposite_bad = 0
    warped_sign = 0
    zero_face = 0
    max_closure = 0.
    closure = np.zeros_like(centers)
    area_sum = np.zeros(ncells)
    invalid_orientation = np.zeros(ncells, dtype=bool)
    for start in range(0, len(faces), chunk):
        end = min(start + chunk, len(faces))
        points = xyz[faces[start:end]]
        normal_a = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]) / 2
        normal_b = np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]) / 2
        normal = normal_a + normal_b
        zero_face += int(np.count_nonzero(np.linalg.norm(normal, axis=1) == 0))
        signs = []
        for owner in (left[start:end], right[start:end]):
            keep = owner >= 0
            delta = points[keep, 0] - centers[owner[keep]]
            va = np.einsum("ij,ij->i", delta, normal_a[keep]) / 3
            vb = np.einsum("ij,ij->i", delta, normal_b[keep]) / 3
            volume = va + vb
            sign = np.sign(volume)
            expanded = np.zeros(len(owner), dtype=np.int8)
            expanded[keep] = sign.astype(np.int8)
            signs.append(expanded)
            warped_sign += int(np.count_nonzero(va * vb < -1e-30))
            np.add.at(volumes, owner[keep], np.abs(volume))
            np.add.at(closure, owner[keep], normal[keep] * sign[:, None])
            np.add.at(area_sum, owner[keep], np.linalg.norm(normal[keep], axis=1))
        interior = (left[start:end] >= 0) & (right[start:end] >= 0)
        bad = interior & (signs[0] * signs[1] > 0)
        opposite_bad += int(np.count_nonzero(bad))
        invalid_orientation[left[start:end][bad]] = True
        invalid_orientation[right[start:end][bad]] = True
    relative_closure = np.linalg.norm(closure, axis=1) / np.maximum(area_sum, 1e-300)
    positive = volumes > 0
    valid = positive & (relative_closure <= 1e-6) & ~invalid_orientation
    stats = {"cell_count": ncells, "positive_volume_cells": int(positive.sum()),
             "zero_volume_cells": int((~positive).sum()),
             "volume_min_m3": float(volumes.min()), "volume_max_m3": float(volumes.max()),
             "total_quarter_volume_m3": float(volumes.sum()),
             "cell_face_count_min": int(incidences.min()), "cell_face_count_max": int(incidences.max()),
             "zero_area_source_faces": zero_face,
             "triangle_tetra_opposite_sign_count": warped_sign,
             "interior_same_orientation_faces": opposite_bad,
             "surface_closure_relative_max": float(relative_closure.max()),
             "surface_closure_relative_above_1e_minus_6_count": int(np.count_nonzero(relative_closure > 1e-6)),
             "geometrically_valid_cells": int(valid.sum()),
             "invalid_cell_volume_fraction_diagnostic": float(volumes[~valid].sum() / volumes.sum()),
             "coordinate_min_m": xyz.min(axis=0).tolist(), "coordinate_max_m": xyz.max(axis=0).tolist()}
    return centers, volumes, stats, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=HERE / "results/cache")
    parser.add_argument("--cases-json", type=Path, help="portable source/input registry; defaults to cache/cases.json or results/source_cases.json")
    parser.add_argument("--raw-dir", type=Path, help="directory containing the 29 raw CFD files; preserve each recorded basename and SHA256")
    parser.add_argument("--geometry", action="store_true", help="derive geometry after connectivity extraction")
    parser.add_argument("--allow-different-topology", action="store_true", help="cache native meshes for separate logical registration; never claim cell-ID correspondence")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = cases(args.cases_json, args.raw_dir)
    write_json(out / "cases.json", rows)
    state_path = out / "extraction_audit.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"cases": [], "complete": False}
    complete = {row["case_id"]: row for row in state["cases"]}
    h0 = rows[0]["source"]["header"]
    nn, nc = h0["nodes"], h0["elements"]
    def mmap(name, dtype, shape):
        p = out / name
        return np.lib.format.open_memmap(p, mode="r+" if p.exists() else "w+", dtype=dtype, shape=shape)
    mach_all = mmap("mach.npy", np.float32, (len(rows), nc))
    node_all = mmap("all_nodes.npy", np.float32, (len(rows), nn, 3))
    ref_hash = state.get("reference_topology_sha256")
    selected = rows[:args.limit] if args.limit else rows
    for i, row in enumerate(selected):
        native = out / "native" / row["case_id"].lower()
        if row["case_id"] in complete and (native / "right_cells.npy").exists():
            continue
        start = time.time()
        path = Path(row["source"]["path"])
        print(f"Reading {i + 1}/{len(rows)} {row['case_id']} ...", flush=True)
        h, xyz, mach, faces, left, right, top_hash, hashes = read_native(path)
        actual_raw = sha256(path)
        if actual_raw != row["source"]["sha256"]:
            raise ValueError(f"Raw source SHA256 mismatch: {path}")
        if ref_hash is None:
            ref_hash = top_hash
            state["reference_topology_sha256"] = ref_hash
            np.save(out / "reference_nodes.npy", xyz)
            for name, a in (("face_nodes", faces), ("left_cells", left), ("right_cells", right)):
                np.save(out / f"{name}.npy", a)
        mach_all[i] = mach
        node_all[i] = xyz
        native.mkdir(parents=True, exist_ok=True)
        for name, a in (("nodes", xyz), ("mach", mach), ("face_nodes", faces), ("left_cells", left), ("right_cells", right)):
            np.save(native / f"{name}.npy", a)
        # Keep double precision source geometry for accurate sub-micrometre cells.
        np.save(out / f"nodes_{row['case_id'].lower()}.npy", xyz)
        audit = {"case_id": row["case_id"], "row_index": i, "raw_sha256": actual_raw,
                 "raw_sha256_verified": True, "topology_sha256": top_hash,
                 "topology_block_sha256": hashes,
                 "exact_reference_topology": top_hash == ref_hash,
                 "mach_min": float(mach.min()), "mach_max": float(mach.max()),
                 "coordinate_min_m": xyz.min(axis=0).tolist(), "coordinate_max_m": xyz.max(axis=0).tolist(),
                 "elapsed_seconds": time.time() - start}
        state["cases"] = [a for a in state["cases"] if a["case_id"] != row["case_id"]]
        state["cases"].append(audit)
        complete[row["case_id"]] = audit
        mach_all.flush()
        node_all.flush()
        write_json(state_path, state)
        print(json.dumps(audit, ensure_ascii=False), flush=True)
        if top_hash != ref_hash and not args.allow_different_topology:
            raise ValueError("Connectivity/order mismatch: native cell correspondence is not valid; construct common 3D registration")
    if args.geometry:
        centers_all = mmap("cell_centers.npy", np.float32, (len(rows), nc, 3))
        volumes_all = mmap("volumes.npy", np.float64, (len(rows), nc))
        for i, row in enumerate(selected):
            if row["case_id"] not in complete or "geometry" in complete[row["case_id"]]:
                continue
            start = time.time()
            print(f"Geometry {i + 1}/{len(rows)} {row['case_id']} ...", flush=True)
            native = out / "native" / row["case_id"].lower()
            xyz = np.load(native / "nodes.npy", mmap_mode="r")
            faces = np.load(native / "face_nodes.npy", mmap_mode="r")
            left = np.load(native / "left_cells.npy", mmap_mode="r")
            right = np.load(native / "right_cells.npy", mmap_mode="r")
            centers, volumes, stats, valid = cell_geometry(xyz, faces, left, right, nc)
            centers_all[i], volumes_all[i] = centers, volumes
            if i == 0:
                np.save(out / "reference_centers.npy", centers)
                np.save(out / "reference_volumes.npy", volumes)
            complete[row["case_id"]]["geometry"] = stats
            complete[row["case_id"]]["geometry_seconds"] = time.time() - start
            centers_all.flush()
            volumes_all.flush()
            write_json(state_path, state)
            print(json.dumps({"case_id": row["case_id"], **stats, "seconds": time.time() - start}), flush=True)
    state["complete"] = len(complete) == len(rows) and all("geometry" in a for a in state["cases"])
    state["mach_extraction_complete"] = len(complete) == len(rows)
    state["all_exact_reference_topology"] = all(a["exact_reference_topology"] for a in state["cases"])
    state["training_case_count"] = 27
    state["heldout_case_count"] = 2
    state["domain"] = "entire exported 1/4 CFD volume, including downstream diffuser"
    state["cell_volume_method"] = "triangulated polyhedral surface integral oriented against each cell interior point; no volume floor applied"
    write_json(state_path, state)


if __name__ == "__main__":
    main()
