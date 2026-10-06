#!/usr/bin/env python3
"""Register y=0 CFD polygons with physical point probes in (x,eta=z/h)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "v05_surrogate_training/results"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PolygonColumns:
    """Even-odd vertical polygon intersections; no cell-index matching.

    Source ASCII precision can create micrometre gaps/overlaps near walls.
    The opt-in allowance records these defects and still requires each actual
    field probe to have exactly one source owner; it never fills a missing
    field or averages overlapping owners.
    """
    def __init__(self, vertices, offsets, x_bounds=None, allow_small_unprobed_defects=False):
        counts = np.diff(offsets)
        if len(counts) < 1 or np.any(counts < 3) or counts.max() > 8:
            raise ValueError("Unsupported polygon topology")
        self.vertices = np.stack([vertices[offsets[:-1] + np.minimum(j, counts - 1)]
                                  for j in range(int(counts.max()))], axis=1)
        self.source_ids = np.arange(len(counts))
        if x_bounds is not None:
            retain = (self.vertices[:, :, 0].max(axis=1) > x_bounds[0]) & (self.vertices[:, :, 0].min(axis=1) <= x_bounds[1])
            self.vertices = self.vertices[retain]
            self.source_ids = self.source_ids[retain]
        edges = np.roll(self.vertices, -1, axis=1) - self.vertices
        next_edges = np.roll(edges, -1, axis=1)
        cross = edges[:, :, 0] * next_edges[:, :, 1] - edges[:, :, 1] * next_edges[:, :, 0]
        self.nonconvex_face_count = int(np.count_nonzero((cross.min(axis=1) < -1e-14) & (cross.max(axis=1) > 1e-14)))
        self.allow_small_unprobed_defects = allow_small_unprobed_defects
        self.max_gap_or_overlap_m = 0.
        self.defective_column_queries = 0
        self.minimum_x = self.vertices[:, :, 0].min(axis=1)
        self.maximum_x = self.vertices[:, :, 0].max(axis=1)

    def intervals(self, x):
        ids = np.flatnonzero((self.minimum_x <= x) & (self.maximum_x > x))
        polygon = self.vertices[ids]
        a, b = polygon, np.roll(polygon, -1, axis=1)
        crossing = ((a[:, :, 0] <= x) & (b[:, :, 0] > x)) | ((b[:, :, 0] <= x) & (a[:, :, 0] > x))
        fraction = np.zeros_like(a[:, :, 0])
        np.divide(x - a[:, :, 0], b[:, :, 0] - a[:, :, 0], out=fraction, where=crossing)
        z = a[:, :, 1] + fraction * (b[:, :, 1] - a[:, :, 1])
        # A concave polygon may meet a vertical line in several intervals.
        # Pair sorted crossings instead of bridging them with min/max.
        if np.any(crossing.sum(axis=1) % 2):
            raise ValueError("Unpaired polygon crossings")
        cuts = np.sort(np.where(crossing, z, np.inf), axis=1)
        if cuts.shape[1] % 2:
            cuts = np.pad(cuts, ((0, 0), (0, 1)), constant_values=np.inf)
        lo, hi = cuts[:, ::2], cuts[:, 1::2]
        length = np.zeros_like(lo)
        np.subtract(hi, lo, out=length, where=np.isfinite(hi))
        nonempty = np.isfinite(hi) & (length > 1e-12)
        ids = np.broadcast_to(ids[:, None], lo.shape)[nonempty]
        low, high = lo[nonempty], hi[nonempty]
        order = np.argsort(low)
        ids, low, high = ids[order], low[order], high[order]
        if not len(ids) or abs(low[0]) > 1e-8 or np.any(high <= low):
            raise ValueError(f"No connected fluid column starts at z=0 at x={x}")
        gaps = low[1:] - high[:-1]
        gap = float(np.max(abs(gaps))) if len(gaps) else 0.
        self.max_gap_or_overlap_m = max(self.max_gap_or_overlap_m, gap)
        if gap > 1e-8:
            self.defective_column_queries += 1
            if not self.allow_small_unprobed_defects or gap > 1e-5:
                raise ValueError(f"Gap or overlap in y=0 fluid column x={x}: max={gap}")
        return self.source_ids[ids], low, high

    def height(self, x):
        return float(self.intervals(x)[2].max())

    def sample(self, x_centers, eta_centers, field):
        result = np.empty((len(eta_centers), len(x_centers)))
        heights = np.empty(len(x_centers))
        for j, x in enumerate(x_centers):
            ids, low, high = self.intervals(x)
            height = float(high.max())
            z = eta_centers * height
            membership = (z[:, None] >= low[None, :] - 1e-12) & (z[:, None] < high[None, :] - 1e-12)
            if not np.all(membership.sum(axis=1) == 1):
                raise ValueError("Every point probe must have exactly one source polygon owner")
            bins = membership.argmax(axis=1)
            result[:, j] = np.asarray(field)[ids[bins]]
            heights[j] = height
        return result, heights


def build_snapshots(output=HERE / "results", config_path=HERE / "config.json"):
    output = Path(output)
    config = json.loads(Path(config_path).read_text())
    manifest = json.loads((V05 / "manifest.json").read_text())
    if digest(V05 / "training_dataset.csv") != manifest["outputs_sha256"]["training_dataset.csv"]:
        raise ValueError("Training input table changed")
    with (V05 / "training_dataset.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    provenance = json.loads((V05 / "dataset_provenance.json").read_text())
    raw_sources = {r["case_id"]: r["source"] for r in provenance["sources"]}
    x_edges = np.linspace(config["x_min_m"], config["x_max_m"], config["nx"] + 1)
    eta_edges = 1 - (1 - np.linspace(0, 1, config["neta"] + 1)) ** 2
    x_centers, eta_centers = (x_edges[:-1] + x_edges[1:]) / 2, (eta_edges[:-1] + eta_edges[1:]) / 2
    weights = (np.diff(eta_edges)[:, None] * np.diff(x_edges)[None, :]).ravel()
    weights /= weights.sum()
    snapshots, heights, audits = [], [], []
    for row in rows:
        case = row["case_id"]
        path = output / "slices" / f"{case}_longitudinal.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        # Extractor metadata names are kept in its manifest; verify actual cache
        # checksum against whichever explicit checksum field it exposes.
        expected = meta.get("output_sha256", meta.get("npz_sha256"))
        if expected is None or digest(path) != expected:
            raise ValueError(f"Longitudinal cache checksum missing or changed: {case}")
        source = meta.get("source", meta.get("raw_source", {}))
        if source.get("sha256") != raw_sources[case]["sha256"]:
            raise ValueError(f"Longitudinal extraction source differs from v05: {case}")
        with np.load(path) as data:
            columns = PolygonColumns(data["polygon_vertices_xz"], data["polygon_offsets"],
                                     x_bounds=(config["x_min_m"], config["x_max_m"]),
                                     allow_small_unprobed_defects=True)
            field, h_mid = columns.sample(x_centers, eta_centers, data["Ma"])
            h_edges = np.array([columns.height(x) for x in x_edges])
            constant, _ = columns.sample(x_centers, eta_centers, np.ones(len(data["Ma"])))
            np.testing.assert_array_equal(constant, np.ones_like(constant))
            if not np.isfinite(field).all() or field.min() < 0 or np.any(h_edges <= 0):
                raise ValueError(f"Invalid registered field: {case}")
            snapshots.append(field.ravel())
            heights.append(h_edges)
            omitted = data["omitted_zero_area_bounds_xz"]
            omitted_in_roi = ((omitted[:, 1] >= config["x_min_m"]) &
                              (omitted[:, 0] <= config["x_max_m"]))
            audits.append({"case_id": case, "source_polygon_count": len(data["Ma"]),
                           "slice_sha256": expected, "slice_metadata_sha256": digest(path.with_suffix(".json")),
                           "source_raw_sha256": source["sha256"], "registered_sample_count": field.size,
                           "source_full_geometry_audit_passed": meta["geometry"].get("full_domain_geometry_audit_passed", True),
                           "source_area_minus_upper_wall_domain_area_m2": meta["geometry"]["area_minus_upper_wall_domain_area_m2"],
                           "source_face_winding_area_residual_diagnostic_m2": meta["geometry"]["area_minus_exterior_area_m2"],
                           "source_zero_projected_area_faces_omitted": meta["geometry"].get("omitted_zero_area_face_count", 0),
                           "source_zero_projected_area_faces_omitted_in_ROI": int(omitted_in_roi.sum()),
                           "source_nonconvex_faces_in_ROI": columns.nonconvex_face_count,
                           "all_requested_probe_locations_have_unique_owner": True,
                           "maximum_column_gap_or_overlap_m": columns.max_gap_or_overlap_m,
                           "all_columns_gapfree_at_1e_minus_8_m_tolerance": columns.max_gap_or_overlap_m <= 1e-8,
                           "column_queries_with_small_unprobed_defects": columns.defective_column_queries,
                           "constant_field_preserved": True,
                           "mach_min": float(field.min()), "mach_max": float(field.max()),
                           "reference_weighted_mean_mach": float(field.ravel() @ weights),
                           "throat_height_at_sampled_x_m": float(h_mid.min()),
                           "throat_sampled_x_m": float(x_centers[h_mid.argmin()])})
        print(f"Registered {case}: {field.size} Mach point probes", flush=True)
    np.savez_compressed(output / "registered_snapshots.npz", case_ids=np.array([r["case_id"] for r in rows]),
                        X=np.array([[float(r["L_mm"]), float(r["P0_Pa"])] for r in rows]),
                        Mach=np.stack(snapshots), wall_heights_m=np.stack(heights),
                        x_edges_m=x_edges, eta_edges=eta_edges, weights=weights)
    write_json(output / "registration_audit.json", {"config": config, "case_count": len(rows), "cases": audits,
        "method": "Even-odd containing-polygon lookup of piecewise-constant cell-centred values at body-fitted reference-grid midpoint probes; every actual probe requires one unique source owner",
        "coordinates": "x physical metres; eta=z/h(x) based on connected fluid-column geometry",
        "weights": "normalized dx*deta reference-domain quadrature; not physical area or mass weights",
        "conservative_remapping": False, "thin_wall_layers_resolved": False,
        "export_precision_policy": "Column gaps/overlaps up to1e-5m may be recorded only if all requested probes remain uniquely owned; no missing values filled or overlapping fields averaged",
        "training_dataset_sha256": digest(V05 / "training_dataset.csv")})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    build_snapshots(args.output_dir)
