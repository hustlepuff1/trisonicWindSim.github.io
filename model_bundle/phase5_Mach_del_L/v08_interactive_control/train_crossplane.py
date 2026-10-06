#!/usr/bin/env python3
"""Train a steady x=0 quarter-section Mach ROM from 27 verified CFD planes.

Source y is negative; the registration coordinate is eta_y=(-y)/width and
eta_z=z/h(eta_y). Fields flatten in [eta_z, eta_y] row-major order. Point
registration is not a conservative remap. Core feedback is a different,
fixed physical central 100x100 mm full-equivalent patch from v05's adaptive
Mach-uniformity region. No transient physics is learned here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/windtunnel-crossplane-matplotlib")
import numpy as np

HERE = Path(__file__).resolve().parent
# Portable framework: original absolute paths are provenance, not dependencies.
sys.path.insert(0, str(HERE.parents[1]))
from portable_paths import local_path, raw_source_status
V05 = HERE.parent / "v05_surrogate_training/results"
V07 = HERE.parent / "v07_flowfield_rom"
sys.path.insert(0, str(V07))
sys.path.insert(0, str(HERE.parent / "v03_cfd_polynomial"))
from core_area import clip_rectangle, polygon_area
from prepare_snapshots import PolygonColumns, digest, write_csv, write_json
from field_rom import model_details, predict_coefficients, predict_rom, projection, train_rom
from train_flow_rom import make_folds
from surrogate_models import supported_mask

CORE_SIDE_M = .05
LATEST = ["M15_new2", "M20_new2", "M25_new", "M30_new", "M35_new2"]


def _edges(values, name):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0) or values[0] != 0 or values[-1] != 1):
        raise ValueError(f"{name} must increase strictly from 0 to 1")
    return values


def core_weights(width_m, wall_heights_m, y_eta_edges, z_eta_edges):
    """Normalized physical intersection areas in flattened [eta_z,eta_y].

    Wall heights live at y grid edges; wall interpolation is piecewise linear.
    Registered values are treated as piecewise constant on these quads. The
    source-polygon core integral is stored separately to expose registration
    error. Require complete coverage of the fixed 0.05 x 0.05 m quarter patch.
    """
    y = _edges(y_eta_edges, "y_eta_edges")
    z = _edges(z_eta_edges, "z_eta_edges")
    h = np.asarray(wall_heights_m, dtype=float)
    if (not np.isscalar(width_m) or not np.isfinite(width_m) or width_m < CORE_SIDE_M
            or h.shape != y.shape or not np.isfinite(h).all() or np.any(h <= 0)):
        raise ValueError("Invalid physical width or wall heights")
    physical_y = width_m * y
    core_heights = np.r_[h[physical_y < CORE_SIDE_M], np.interp(CORE_SIDE_M, physical_y, h)]
    if np.min(core_heights) < CORE_SIDE_M:
        raise ValueError("Fixed physical core extends outside the fluid geometry")
    areas = np.zeros((len(z) - 1, len(y) - 1))
    for j in np.flatnonzero(physical_y[:-1] < CORE_SIDE_M):
        for k in np.flatnonzero(z[:-1] * min(h[j], h[j + 1]) < CORE_SIDE_M):
            quad = np.array([[physical_y[j], z[k] * h[j]],
                             [physical_y[j + 1], z[k] * h[j + 1]],
                             [physical_y[j + 1], z[k + 1] * h[j + 1]],
                             [physical_y[j], z[k + 1] * h[j]]])
            areas[k, j] = polygon_area(clip_rectangle(quad, (0, CORE_SIDE_M, 0, CORE_SIDE_M)))
    if not np.isclose(areas.sum(), CORE_SIDE_M ** 2, rtol=1e-10, atol=1e-14):
        raise ValueError("Registered geometry does not cover the fixed core exactly")
    return areas.ravel() / areas.sum()


def source_core_mean(vertices, offsets, values):
    """Exact piecewise-constant source-polygon mean on the central quarter."""
    values = np.asarray(values, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    vertices = np.asarray(vertices, dtype=float)
    if (len(offsets) != len(values) + 1 or offsets[0] != 0 or offsets[-1] != len(vertices)
            or np.any(np.diff(offsets) < 3) or not np.isfinite(values).all()
            or not np.isfinite(vertices).all()):
        raise ValueError("Invalid source polygons or values")
    areas = np.zeros(len(values))
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        poly = vertices[start:stop]
        if np.any(poly.min(axis=0) >= CORE_SIDE_M) or np.any(poly.max(axis=0) <= 0):
            continue
        areas[i] = polygon_area(clip_rectangle(poly, (0, CORE_SIDE_M, 0, CORE_SIDE_M)))
    if not np.isclose(areas.sum(), CORE_SIDE_M ** 2, rtol=1e-8, atol=1e-13):
        raise ValueError("Source core polygons fail complete-area coverage")
    return float(values @ areas / areas.sum())


def register(output):
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = V05 / "dataset_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    with (V05 / "training_dataset.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    ids = [r["case_id"] for r in rows]
    expected = ["M_start"] + [f"M{i}" for i in range(15, 36)] + LATEST
    if ids != expected or ids != provenance["case_ids"]:
        raise ValueError("Unexpected selected training cases")
    manifest = json.loads((V05 / "manifest.json").read_text())
    for name in ("training_dataset.csv", "dataset_provenance.json"):
        if digest(V05 / name) != manifest["outputs_sha256"][name]:
            raise ValueError(f"v05 source artifact changed: {name}")
    edges = 1 - (1 - np.linspace(0, 1, 65)) ** 2
    centers = (edges[:-1] + edges[1:]) / 2
    weights = np.outer(np.diff(edges), np.diff(edges)).ravel()
    fields, widths, heights, exact_core, sampled_core, core_rows, audits = [], [], [], [], [], [], []
    by_id = {s["case_id"]: s for s in provenance["sources"]}
    for case in ids:
        source = by_id[case]
        plane = local_path(source["plane"])
        metadata = plane.with_suffix(".json")
        if digest(plane) != source["plane_sha256"] or digest(metadata) != source["plane_metadata_sha256"]:
            raise ValueError(f"Unverified plane cache: {case}")
        raw_status = raw_source_status(source["source"])
        with np.load(plane, allow_pickle=False) as data:
            vertices = data["polygon_vertices"].copy()
            vertices[:, 0] *= -1
            offsets, values = data["polygon_offsets"], data["Ma"]
            width, height = vertices.max(axis=0)
            if np.max(abs(vertices.min(axis=0))) > 1e-9:
                raise ValueError("Crossplane does not start on both symmetry axes")
            columns = PolygonColumns(vertices, offsets)
            field, h_centers = columns.sample(centers * width, centers, values)
            # These actual x=0 sections are rectangles. Prove that simplification
            # per source before assigning edge heights, including both endpoints.
            h_edges = np.array([columns.height(x) for x in np.minimum(edges * width, np.nextafter(width, 0))])
            if (not np.allclose(h_centers, height, atol=1e-10, rtol=0)
                    or not np.allclose(h_edges, height, atol=1e-10, rtol=0)
                    or not np.isclose(data["area_m2"].sum(), width * height, atol=1e-11, rtol=1e-8)):
                raise ValueError("Nonrectangular crossplane needs general physical quadrature")
            cw = core_weights(width, h_edges, edges, edges)
            exact = source_core_mean(vertices, offsets, values)
            fields.append(field.ravel()); widths.append(width); heights.append(h_edges)
            exact_core.append(exact); sampled_core.append(float(field.ravel() @ cw)); core_rows.append(cw)
            audits.append({"case_id": case, "source_cell_count": len(values), "width_m": float(width),
                "height_m": float(height), "sample_count": int(field.size), "unique_probe_ownership": True,
                "column_max_gap_or_overlap_m": columns.max_gap_or_overlap_m,
                "source_nonconvex_polygon_count": columns.nonconvex_face_count,
                "source_plane": str(plane), "source_plane_sha256": source["plane_sha256"],
                "raw_volume_sha256_previously_verified_by_v05": source["source"]["sha256"],
                "raw_volume_size_mtime_rechecked_in_packaged_run": False,
                "raw_verification": raw_status,
                "exact_source_core_mean": exact, "registered_core_mean": sampled_core[-1]})
        print(f"Registered {case}: {field.shape}, core source={exact:.8f}, registered={sampled_core[-1]:.8f}", flush=True)
    np.savez_compressed(output / "registered_snapshots.npz", case_ids=np.array(ids),
        X=np.array([[float(r["L_mm"]), float(r["P0_Pa"])] for r in rows]), Mach=np.array(fields),
        y_eta_edges=edges, z_eta_edges=edges, width_m=np.array(widths), wall_heights_m=np.array(heights),
        weights=weights, core_weights_percase=np.array(core_rows), core_mean_percase=np.array(exact_core),
        registered_field_core_mean=np.array(sampled_core))
    write_json(output / "registration.json", {"case_count": len(ids), "case_ids": ids,
        "source_provenance_sha256": digest(provenance_path), "source_dataset_sha256": digest(V05 / "training_dataset.csv"),
        "source_hash_policy": "Verify all packaged cached planes and metadata against preserved v05 SHA256; raw-volume hashes remain historical provenance and are not rechecked",
        "coordinate_system": "x=0, source y<=0,z>=0; eta_y=-y/width; eta_z=z/h(eta_y)",
        "flatten_order": "C row-major [z_eta, y_eta]", "registration": "strict unique source polygon owner at physical grid midpoint probes; no blank filling",
        "edges": "64 cells per axis; eta=1-(1-s)^2 for uniform s, clustered near walls",
        "quadrature": "fixed normalized reference cell area d_eta_y*d_eta_z, equal to normalized physical area for verified rectangular source sections",
        "core": "Exact physical 0<=-y<=0.05m, 0<=z<=0.05m quarter, equivalent to central full 100x100mm under symmetry",
        "core_response_differs_from_v05_adaptive_core": True,
        "registration_is_conservative_remap": False,
        "pressure_caveat": "P0 uses user-confirmed fixed target reference pressures; solver input deck was not independently verified",
        "maximum_core_mean_registration_error_mach": float(np.max(abs(np.array(exact_core) - sampled_core))),
        "cases": audits})


def errors(observed, predicted, weights, core, edges):
    delta = predicted - observed
    centers = (edges[:-1] + edges[1:]) / 2
    near_wall = np.maximum(centers[:, None], centers[None, :]).ravel() >= .9
    return {"weighted_rmse_mach": float(np.sqrt(delta ** 2 @ weights)),
        "weighted_mae_mach": float(abs(delta) @ weights),
        "relative_weighted_L2_percent": float(100 * np.sqrt(delta ** 2 @ weights / (observed ** 2 @ weights))),
        "max_abs_error_mach": float(abs(delta).max()),
        "wall_region_rmse_mach": float(np.sqrt(delta[near_wall] ** 2 @ weights[near_wall] / weights[near_wall].sum())),
        "registered_core_error_mach": float(delta @ core),
        "negative_prediction_sample_count": int(np.count_nonzero(predicted < 0))}


def comparison_plot(output, data, held_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 3, figsize=(11, 15), constrained_layout=True)
    ids = data["case_ids"].tolist()
    for row, case in enumerate(LATEST):
        i = ids.index(case)
        observed, predicted = data["Mach"][i].reshape(64, 64), held_out[i].reshape(64, 64)
        difference = predicted - observed
        limit = max(.002, float(abs(difference).max()))
        for col, (field, title) in enumerate(((observed, "CFD registered"), (predicted, "Held-condition-out ROM"), (difference, "ROM minus CFD"))):
            ax = axes[row, col]
            artist = ax.pcolormesh(data["y_eta_edges"], data["z_eta_edges"], field, shading="flat",
                cmap="coolwarm" if col == 2 else "turbo", vmin=-limit if col == 2 else 0,
                vmax=limit if col == 2 else float(max(observed.max(), predicted.max())))
            ax.set(title=f"{case}: {title}", xlabel="eta_y = -y / width", ylabel="eta_z = z / wall height", aspect="equal")
            fig.colorbar(artist, ax=ax, label="Mach")
    fig.suptitle("Steady x=0 quarter-plane validation; all spatial points of each shown condition were held out")
    fig.savefig(output / "cfd_vs_heldout_rom.png", dpi=150)
    plt.close(fig)


def write_readme(output, details, summaries, validation):
    rows = [r for r in summaries if r["subset"] == "all"]
    lines = ["# Steady x=0 Mach field surrogate", "",
        "27 CFD conditions: M_start, M15 through M35, M15_new2, M20_new2, M25_new, M30_new, M35_new2. Superseded M15/M20/M35 new cases are excluded.", "",
        "Each source quarter section has y in [-0.25, 0] m and z in [0, h] m, with h=0.248886–0.251216 m. Registration uses a 64×64 grid in eta_y=-y/width and eta_z=z/h, clustered toward both walls. Every physical probe has one source cell owner; no missing values are filled. Source cell order is never treated as mesh correspondence. The NPZ field order is row-major [z_eta, y_eta].", "",
        "`registered_snapshots.npz` contains case_ids, X=[L_mm,P0_Pa], Mach[27,4096], y_eta_edges, z_eta_edges, width_m, wall_heights_m, weights, core_weights_percase, core_mean_percase and registered_field_core_mean. Fixed reference-area POD quadrature equals normalized physical cell area because each audited section is rectangular. Registration samples the source piecewise-constant field; it is not a conservative remap.", "",
        "`core_mean_percase` integrates clipped source polygons on the fixed physical quarter 0<=-y<=0.05 m, 0<=z<=0.05 m. Under symmetry this is the full central 100×100 mm patch. `registered_field_core_mean` integrates the registered field on clipped physical quadrilaterals. This definition differs from v05's adaptive Mach-uniformity core. `train_crossplane.core_weights(width_m, wall_heights_m, y_eta_edges, z_eta_edges)` supplies the geometry-dependent normalized weights for runtime field feedback.", "",
        f"Maximum source-versus-registered core mean difference: {validation['max_source_core_registration_error_mach']:.8g} Mach.", "",
        f"POD rank {details['rank']}; retained centered training energy {100 * details['achieved_energy_fraction']:.7f}% (target 99.99%, cap 12). Independent Matérn-5/2 GP coefficient models use L and log(P0). POD mean, basis, rank, scaling and GP parameters are fitted inside every holdout fold. All full-fit and fold GP optimizers converged.", "",
        "| Validation scheme | Field RMSE (Mach) | Mean relative L2 | Fixed source-core RMSE (Mach) | Worst point error (Mach) |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['cv_scheme']} | {float(row['field_rmse_mach']):.7f} | {float(row['mean_relative_L2_percent']):.5f}% | {float(row['source_core_rmse_mach']):.7f} | {float(row['worst_point_error_mach']):.7f} |")
    lines += ["", "The 48 folds produce 54 heldout fields. Many heldout conditions lie outside the remaining training-input convex hull: 23/27 condition holdouts and 24/27 pressure-group holdouts. The CSV separates interpolation-support and outside-support metrics. These tests evaluate new steady conditions within this campaign; all 27 conditions train the final model, so there is no independent final test campaign. Known heldout geometry is used for registration. Local near-wall error can exceed the area-average error; see the explicit error maps and metrics.", "",
        "Full-fit smoothing errors remain: the M15_new2 source central mean is 1.51052755, while its full-fit ROM central mean is about 1.49705938. Model-based closed-loop tracking does not remove that CFD-versus-ROM bias and cannot establish physical tracking accuracy.", "",
        "`field_model.pkl` is the trusted local v07 field_rom model dictionary. `field_model.npz` exports the POD mean, modes, weights, inputs, predicted training coefficients, and per-case core projections; it does not by itself encode GP inference. `field_model.json` identifies training inputs and hashes. `cfd_vs_heldout_rom.png` compares the latest five complete heldout conditions, not training reconstructions.", "",
        "Cached source plane and metadata SHA256 hashes are checked against the verified v05 provenance; raw volume size and mtime must still match that earlier full-volume SHA256 audit. Pressure values follow the user-confirmed fixed target-pressure rule; solver input decks were not independently verified.", "",
        "This model learns steady spatial fields. Evaluating it along assumed actuator/control trajectories is quasi-steady visualization, not transient CFD, an identified dynamic plant, or independent closed-loop validation. No new CFD run or sphere test is represented here.", "",
        "Reproduce from the workspace root:", "", "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 phase5_Mach_del_L/v08_interactive_control/train_crossplane.py",
        "OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s phase5_Mach_del_L/v08_interactive_control -p test_crossplane.py -v",
        "```", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output):
    write_json(output / "manifest.json", {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"},
        "source_code_sha256": {str(p): digest(p) for p in [Path(__file__), HERE / "test_crossplane.py", V07 / "field_rom.py", V07 / "prepare_snapshots.py", V07 / "train_flow_rom.py", HERE.parent / "v05_surrogate_training/surrogate_models.py", HERE.parent / "v03_cfd_polynomial/core_area.py"]}})


def run(output):
    register(output)
    data = dict(np.load(output / "registered_snapshots.npz", allow_pickle=False))
    X, F, weights, cores = data["X"], data["Mach"], data["weights"], data["core_weights_percase"]
    ids = data["case_ids"].tolist()
    protocol = {"locked_before_fitting_utc": datetime.now(timezone.utc).isoformat(),
        "energy_target": .9999, "rank_cap": 12, "kernel_family": "Matern52",
        "condition_count": len(ids), "spatial_samples_per_condition": F.shape[1],
        "one_case_one_snapshot": True, "POD_GP_preprocessing_refitted_per_fold": True,
        "spatial_points_not_independent_test_conditions": True,
        "geometry_registration": "Known held-out physical geometry for reference-map evaluation, geometry not learned in CV",
        "holdout_scope": "whole condition and whole pressure group; all27 used final training, no independent final test campaign",
        "steady_only": True, "transient_or_closed_loop_CFD_validation": False}
    write_json(output / "training_protocol.json", protocol)
    model = train_rom(X, F, weights, energy=.9999, max_rank=12)
    fitted = predict_rom(model, X)
    details = model_details(model)
    (output / "field_model.pkl").write_bytes(pickle.dumps(model, protocol=5))
    np.savez_compressed(output / "field_model.npz", mean=model["mean"], modes=model["modes"],
        weights=weights, training_inputs=X, training_coefficients=predict_coefficients(model, X),
        core_mean_offset_percase=cores @ model["mean"], core_mode_projection_percase=cores @ model["modes"].T)
    write_json(output / "field_model.json", {"details": details, "case_ids": ids, "training_inputs": X.tolist(),
        "feature_columns": ["L_mm", "P0_Pa"], "protocol": protocol,
        "registered_snapshots_sha256": digest(output / "registered_snapshots.npz"),
        "field_model_pickle_sha256": digest(output / "field_model.pkl"),
        "npz_scope": "POD arrays and full-fit training coefficients only; GP inference uses trusted local pickle and field_rom.predict_coefficients",
        "geometry_required_for_physical_display_and_core_mean": True})
    print(f"FULL_MODEL_READY {output / 'field_model.pkl'} rank={details['rank']}", flush=True)
    fit_rows = [{"case_id": ids[i], **errors(F[i], fitted[i], weights, cores[i], data["y_eta_edges"]),
                 "exact_source_core_error_mach": float(fitted[i] @ cores[i] - data["core_mean_percase"][i])} for i in range(len(X))]
    write_csv(output / "fit_metrics.csv", fit_rows)
    cv_rows, fold_details, predictions = [], [], {}
    for scheme, number, train, test in make_folds(X):
        fold = train_rom(X[train], F[train], weights, energy=.9999, max_rank=12)
        np.testing.assert_allclose(fold["mean"], F[train].mean(axis=0), atol=1e-12)
        prediction, projected = predict_rom(fold, X[test]), projection(fold, F[test])
        predictions.setdefault(scheme, np.empty_like(F))[test] = prediction
        inside = supported_mask(X[train], X[test])
        for j, i in enumerate(test):
            cv_rows.append({"cv_scheme": scheme, "case_id": ids[i], "fold_id": number,
                "training_count": len(train), "rank": fold["details"]["rank"], "in_training_hull": bool(inside[j]),
                "P0_within_training_range": bool(X[train, 1].min() <= X[i, 1] <= X[train, 1].max()),
                **errors(F[i], prediction[j], weights, cores[i], data["y_eta_edges"]),
                "exact_source_core_error_mach": float(prediction[j] @ cores[i] - data["core_mean_percase"][i]),
                "projection_only_rmse_mach": float(np.sqrt((projected[j] - F[i]) ** 2 @ weights))})
        fold_details.append({"cv_scheme": scheme, "fold_id": number,
            "train_case_ids": [ids[i] for i in train], "test_case_ids": [ids[i] for i in test], "details": model_details(fold)})
        print(f"CV {scheme} {number + 1}: {[ids[i] for i in test]}", flush=True)
    summaries = []
    for scheme in predictions:
        rows = [r for r in cv_rows if r["cv_scheme"] == scheme]
        assert len(rows) == 27
        for subset, selected in (("all", rows), ("inside_training_hull", [r for r in rows if r["in_training_hull"]]),
                                 ("outside_training_hull", [r for r in rows if not r["in_training_hull"]])):
            if selected:
                summaries.append({"cv_scheme": scheme, "subset": subset, "case_count": len(selected),
                    "field_rmse_mach": float(np.sqrt(np.mean([r["weighted_rmse_mach"] ** 2 for r in selected]))),
                    "mean_relative_L2_percent": float(np.mean([r["relative_weighted_L2_percent"] for r in selected])),
                    "source_core_rmse_mach": float(np.sqrt(np.mean([r["exact_source_core_error_mach"] ** 2 for r in selected]))),
                    "worst_point_error_mach": max(r["max_abs_error_mach"] for r in selected)})
    write_csv(output / "cross_validation_metrics.csv", cv_rows)
    write_csv(output / "cross_validation_summary.csv", summaries)
    write_json(output / "fold_diagnostics.json", fold_details)
    np.savez_compressed(output / "field_predictions.npz", fitted=fitted, **predictions)
    np.testing.assert_array_equal(predict_rom(pickle.loads((output / "field_model.pkl").read_bytes()), X), fitted)
    comparison_plot(output, data, predictions["leave_one_condition_out"])
    write_json(output / "validation.json", {"status": "completed", "case_count": 27, "fold_count": len(fold_details),
        "held_out_field_count": len(cv_rows), "all_source_plane_hashes_checked": True,
        "unique_probe_ownership_all_cases": True, "full_model_optimizer_success": details["all_coefficient_optimizers_successful"],
        "folds_with_nonconverged_optimizer": sum(not f["details"]["all_coefficient_optimizers_successful"] for f in fold_details),
        "rank": details["rank"], "energy_target_met": details["energy_target_met"],
        "achieved_energy_fraction": details["achieved_energy_fraction"],
        "max_source_core_registration_error_mach": float(abs(data["core_mean_percase"] - data["registered_field_core_mean"]).max()),
        "transient_or_closed_loop_CFD_validation": False, "independent_final_test_campaign": False,
        "negative_full_fit_sample_count": int(np.count_nonzero(fitted < 0)),
        "negative_heldout_sample_count": sum(r["negative_prediction_sample_count"] for r in cv_rows)})
    write_readme(output, details, summaries, json.loads((output / "validation.json").read_text()))
    write_manifest(output)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/crossplane")
    args = parser.parse_args()
    run(args.output_dir.resolve())
