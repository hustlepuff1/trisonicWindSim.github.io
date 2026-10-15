#!/usr/bin/env python3
"""Prepare a new inlet-to-x=0 3D Mach training domain from audited v10 data.

The first 286 native axial cell layers are unchanged. The remaining upstream
portions of native layers 286 and 287 are clipped at the physical x=0 plane
and conservatively combined into one corresponding terminal control volume.
No entirely downstream cell contributes a value or a POD quadrature weight.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "v10_volume_rom/results/cache"
NODE_SHAPE = (288, 72, 72)
CELL_SHAPE = (287, 71, 71)
PREFIX_LAYERS = 286
TRANSVERSE_CELLS = 71 * 71


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tetra_measure(points):
    volume = np.abs(np.einsum("ij,ij->i", points[:, 1] - points[:, 0],
                    np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]))) / 6
    return volume, volume[:, None] * points.mean(axis=1)


def clip_tetra_x0(points):
    """Exact piecewise-linear tetrahedron/half-space volume and first moment."""
    inside = points[:, :, 0] <= 0
    count = inside.sum(axis=1)
    order = np.argsort(~inside, axis=1, kind="stable")
    p = np.take_along_axis(points, order[:, :, None], axis=1)
    volume = np.zeros(len(p))
    moment = np.zeros((len(p), 3))
    full_volume, full_moment = tetra_measure(p)
    complete = count == 4
    volume[complete], moment[complete] = full_volume[complete], full_moment[complete]
    one = count == 1
    if np.any(one):
        q = p[one]
        a, b = q[:, 0], q[:, 1:]
        fraction = -a[:, None, 0] / (b[:, :, 0] - a[:, None, 0])
        cut = a[:, None] + fraction[:, :, None] * (b - a[:, None])
        volume[one], moment[one] = tetra_measure(np.concatenate((a[:, None], cut), axis=1))
    three = count == 3
    if np.any(three):
        q = p[three]
        a, b = q[:, 3], q[:, :3]
        fraction = -a[:, None, 0] / (b[:, :, 0] - a[:, None, 0])
        cut = a[:, None] + fraction[:, :, None] * (b - a[:, None])
        outside_volume, outside_moment = tetra_measure(np.concatenate((a[:, None], cut), axis=1))
        volume[three] = full_volume[three] - outside_volume
        moment[three] = full_moment[three] - outside_moment
    two = count == 2
    if np.any(two):
        a, b, c, d = (p[two, i] for i in range(4))
        def intersection(u, v):
            f = -u[:, 0] / (v[:, 0] - u[:, 0])
            return u + f[:, None] * (v - u)
        ac, ad, bc, bd = intersection(a, c), intersection(a, d), intersection(b, c), intersection(b, d)
        for tet in ((a, b, ac, ad), (b, ac, ad, bd), (b, ac, bc, bd)):
            v, m = tetra_measure(np.stack(tet, axis=1))
            volume[two] += v
            moment[two] += m
    tolerance = np.maximum(full_volume * 1e-12, 1e-300)
    if np.any(volume < -tolerance) or np.any(volume > full_volume + tolerance):
        raise ValueError("Clipped tetrahedron volume outside original volume")
    if np.any(volume < 0):
        raise ValueError("Negative clipped volume; do not apply an artificial floor")
    return volume, moment, full_volume


def clipped_terminal_cells(native, cell_map):
    targets = np.asarray(cell_map[PREFIX_LAYERS * TRANSVERSE_CELLS:(PREFIX_LAYERS + 2) * TRANSVERSE_CELLS])
    lookup = np.full(len(cell_map), -1, dtype=np.int32)
    lookup[targets] = np.arange(len(targets), dtype=np.int32)
    owners, face_ids = [], []
    for side in ("left_cells", "right_cells"):
        original = np.load(native / f"{side}.npy", mmap_mode="r")
        mapped = lookup[np.maximum(original, 0)]
        mapped[original < 0] = -1
        retain = np.flatnonzero(mapped >= 0)
        owners.append(mapped[retain])
        face_ids.append(retain)
    owners, face_ids = np.concatenate(owners), np.concatenate(face_ids)
    if np.any(np.bincount(owners, minlength=len(targets)) != 6):
        raise ValueError("Terminal native cell does not have six source faces")
    face_ids = face_ids[np.argsort(owners, kind="stable")].reshape(len(targets), 6)
    faces = np.load(native / "face_nodes.npy", mmap_mode="r")
    nodes = np.load(native / "nodes.npy", mmap_mode="r")
    q = nodes[faces[face_ids]]
    center = np.load(native / "centers.npy", mmap_mode="r")[targets]
    tetra = np.empty((len(targets), 6, 2, 4, 3))
    tetra[:, :, :, 0] = center[:, None, None]
    tetra[:, :, 0, 1:] = q[:, :, [0, 1, 2]]
    tetra[:, :, 1, 1:] = q[:, :, [0, 2, 3]]
    clipped, moment, original = clip_tetra_x0(tetra.reshape(-1, 4, 3))
    volume = clipped.reshape(len(targets), 12).sum(axis=1)
    moment = moment.reshape(len(targets), 12, 3).sum(axis=1)
    original = original.reshape(len(targets), 12).sum(axis=1)
    audited = np.load(native / "volumes.npy", mmap_mode="r")[targets]
    valid = np.load(native / "valid_geometry.npy", mmap_mode="r")[targets]
    if not valid.all() or np.any(audited <= 0):
        raise ValueError("Terminal source cell geometry is invalid; no volume floor is allowed")
    relative = np.abs(original - audited) / audited
    if relative.max() > 1e-10:
        raise ValueError("Native face triangulation differs from audited source volume")
    return volume.reshape(2, 71, 71), moment.reshape(2, 71, 71, 3), {
        "native_terminal_cell_count": len(targets),
        "upstream_contributing_terminal_cells": int(np.count_nonzero(volume > 0)),
        "excluded_downstream_terminal_cells": int(np.count_nonzero(volume == 0)),
        "original_volume_relative_crosscheck_max": float(relative.max()),
        "degenerate_terminal_cells": 0, "terminal_weight_fallback_count": 0}


def crop_geometry(native):
    node_map = np.load(native / "structured_to_native_nodes.npy", mmap_mode="r")
    points = np.load(native / "nodes.npy", mmap_mode="r")[node_map].reshape(702, 72, 72, 3)
    x = points[:, :, :, 0]
    if np.any(np.diff(x, axis=0) <= 0):
        raise ValueError("Axial node tracks must be strictly monotone")
    crossing = (x > 0).argmax(axis=0)
    if np.any((crossing < 287) | (crossing > 288)):
        raise ValueError("New x=0 crossing outside the audited two-cell terminal slab")
    jj, kk = np.indices((72, 72))
    a, b = points[crossing - 1, jj, kk], points[crossing, jj, kk]
    fraction = -a[:, :, 0] / (b[:, :, 0] - a[:, :, 0])
    cap = a + fraction[:, :, None] * (b - a)
    cap[:, :, 0] = 0.
    geometry = np.concatenate((points[:287], cap[None]), axis=0)
    if np.any(geometry[:-1, :, :, 0] >= 0) or np.any(geometry[-1, :, :, 0] != 0):
        raise ValueError("New geometry must lie strictly upstream and terminate exactly at x=0")
    return geometry, {"first_positive_node_layer_min": int(crossing.min()),
                      "first_positive_node_layer_max": int(crossing.max()),
                      "geometry_x_min_m": float(geometry[:, :, :, 0].min()),
                      "geometry_x_max_m": float(geometry[:, :, :, 0].max()),
                      "terminal_plane_exact_x0": True,
                      "downstream_only_cell_layers_excluded": [288, 700]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=HERE / "results/cache")
    args = parser.parse_args()
    source, out = args.source_cache, args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "geometry").mkdir(exist_ok=True)
    (out / "terminal").mkdir(exist_ok=True)
    original = json.loads((source / "extraction_manifest.json").read_text())
    if not original["complete"] or not original["topology_correspondence_verified"]:
        raise ValueError("The complete audited v10 native data is required")
    # Invalidate both readiness gates before replacing any existing memmap.
    # A concurrent training watcher must never consume a partially rebuilt cache.
    for name in ("domain_manifest.json", "extraction_manifest.json"):
        write_json(out / name, {"complete": False, "preparation_in_progress": True})
    rows = json.loads((source / "cases.json").read_text())
    write_json(out / "cases.json", rows)
    ncells = int(np.prod(CELL_SHAPE))
    field = np.lib.format.open_memmap(out / "registered_mach.npy", mode="w+", dtype=np.float32, shape=(len(rows), ncells))
    source_field = np.load(source / "registered_mach.npy", mmap_mode="r").reshape(len(rows), 701, 71, 71)
    prefix_cells = PREFIX_LAYERS * TRANSVERSE_CELLS
    manifest = {"complete": False, "topology_correspondence_verified": True,
                "domain": "inlet through physical x=0 test-section plane; diffuser excluded from training and display",
                "domain_x_max_m": 0., "node_shape": list(NODE_SHAPE), "cell_shape": list(CELL_SHAPE),
                "cell_count": ncells, "training_case_count": 27, "heldout_case_count": 2,
                "reference_case": "M17", "mach_cache": "registered_mach.npy",
                "reference_centers": "registered_reference_centers.npy", "reference_volumes": "registered_reference_volumes.npy",
                "geometry_pattern": "geometry/{case_id}.npy", "preserved_native_axial_cell_layers": 286,
                "terminal_method": "Volume-conservative aggregation of actual native cell portions clipped to x<=0, using original face triangulations and their cell-centre tetrahedra",
                "terminal_geometry": "Original node grid through node layer286 plus exact x=0 intersections on every native axial node track; terminal rendering cells simplify intermediate native edges",
                "source_mach_representation": "piecewise constant in each original native CFD cell; a straddling cell contributes only its upstream clipped physical volume",
                "diffuser_cell_value_contributions": 0, "reference_volume_weights_positive_and_valid": False,
                "source_extraction_manifest_sha256": digest(source / "extraction_manifest.json"),
                "source_registered_mach_sha256": digest(source / "registered_mach.npy"),
                "cases": []}
    for i, row in enumerate(rows):
        started = time.monotonic()
        case = row["case_id"]
        native = source / "native" / case.lower()
        print(f"Preparing {i + 1}/{len(rows)} {case}", flush=True)
        cell_map = np.load(native / "structured_to_native_cells.npy", mmap_mode="r")
        geometry, geometry_audit = crop_geometry(native)
        volume, moment, volume_audit = clipped_terminal_cells(native, cell_map)
        total = volume.sum(axis=0)
        if np.any(total <= 0):
            raise ValueError("A terminal aggregate has zero physical upstream volume")
        terminal = (source_field[i, 286:288].astype(np.float64) * volume).sum(axis=0) / total
        field[i, :prefix_cells] = source_field[i, :286].reshape(-1)
        field[i, prefix_cells:] = terminal.reshape(-1)
        if not np.array_equal(field[i, :prefix_cells], source_field[i, :286].reshape(-1)):
            raise ValueError("Unchanged upstream native Mach values were modified")
        if not np.isfinite(field[i]).all() or np.any(field[i] < 0):
            raise ValueError("Invalid cropped Mach field")
        np.save(out / "geometry" / f"{case}.npy", geometry)
        np.save(out / "terminal" / f"{case}_clipped_source_volumes.npy", volume)
        if case == "M17":
            centers = np.load(source / "registered_reference_centers.npy", mmap_mode="r")
            weights = np.load(source / "registered_reference_volumes.npy", mmap_mode="r")
            reference_weights = np.concatenate((weights[:prefix_cells], total.reshape(-1)))
            reference_centers = np.concatenate((centers[:prefix_cells], (moment.sum(axis=0) / total[:, :, None]).reshape(-1, 3)))
            if np.any(reference_weights <= 0) or np.any(reference_centers[:, 0] > 1e-12):
                raise ValueError("Cropped reference weights/centres are invalid")
            np.save(out / "registered_reference_volumes.npy", reference_weights)
            np.save(out / "registered_reference_centers.npy", reference_centers)
            np.save(out / "registered_reference_nodes.npy", geometry.reshape(-1, 3))
            manifest["reference_volume_weights_positive_and_valid"] = True
            manifest["reference_quarter_volume_m3"] = float(reference_weights.sum())
            manifest["reference_terminal_quarter_volume_m3"] = float(total.sum())
            manifest["reference_weight_min_m3"] = float(reference_weights.min())
        audit = {"case_id": case, "training": row["training"], "row_index": i,
                 "L_mm": row["L_mm"], "P0_Pa": row["P0_Pa"],
                 **geometry_audit, **volume_audit,
                 "unchanged_native_cell_count": prefix_cells,
                 "represented_native_upstream_cell_count": prefix_cells + volume_audit["upstream_contributing_terminal_cells"],
                 "terminal_quarter_volume_m3": float(total.sum()),
                 "terminal_min_volume_m3": float(total.min()),
                 "mach_min": float(field[i].min()), "mach_max": float(field[i].max()),
                 "geometry_sha256": digest(out / "geometry" / f"{case}.npy"),
                 "elapsed_seconds": time.monotonic() - started}
        manifest["cases"].append(audit)
        field.flush()
        write_json(out / "domain_manifest.json", manifest)
        print(json.dumps(audit), flush=True)
    manifest["complete"] = True
    manifest["all_sources_full_sha256_verified"] = original["all_sources_full_sha256_verified"]
    manifest["outputs_sha256"] = {name: digest(out / name) for name in (
        "registered_mach.npy", "registered_reference_volumes.npy", "registered_reference_centers.npy",
        "registered_reference_nodes.npy", "cases.json")}
    write_json(out / "domain_manifest.json", manifest)
    write_json(out / "extraction_manifest.json", manifest)


if __name__ == "__main__":
    main()
