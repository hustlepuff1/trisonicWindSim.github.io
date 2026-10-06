#!/usr/bin/env python3
"""Fit a geometry-registered steady Mach-field POD/GP surrogate and replay.

No temporal flow dynamics are learned. All displayed fields are static
predictions at the actuator states of a separately assumed controller model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from prepare_snapshots import HERE, V05, build_snapshots, digest, write_csv, write_json
from field_rom import model_details, predict_coefficients, predict_rom, projection, train_rom

sys.path.insert(0, str(HERE.parent / "v05_surrogate_training"))
from surrogate_models import supported_mask

V06 = HERE.parent / "v06_mach_controller/results"


def field_errors(observed, predicted, weights, eta, nx):
    error = predicted - observed
    near_wall = np.repeat(eta >= .9, nx)
    near_center = np.repeat(eta <= .2, nx)
    return {
        "weighted_rmse_mach": float(np.sqrt(error ** 2 @ weights)),
        "weighted_mae_mach": float(abs(error) @ weights),
        "relative_weighted_L2_percent": 100 * float(np.sqrt(error ** 2 @ weights) / np.sqrt(observed ** 2 @ weights)),
        "max_abs_error_mach": float(abs(error).max()),
        "wall_region_rmse_mach": float(np.sqrt(error[near_wall] ** 2 @ (weights[near_wall] / weights[near_wall].sum()))),
        "center_region_rmse_mach": float(np.sqrt(error[near_center] ** 2 @ (weights[near_center] / weights[near_center].sum()))),
        "negative_prediction_sample_count": int(np.count_nonzero(predicted < 0)),
    }


def make_folds(X):
    indices = np.arange(len(X))
    for scheme in ("leave_one_condition_out", "leave_one_pressure_out"):
        groups = [indices[i:i + 1] for i in indices] if scheme == "leave_one_condition_out" else [
            indices[X[:, 1] == pressure] for pressure in np.unique(X[:, 1])]
        for number, test in enumerate(groups):
            train = indices[~np.isin(indices, test)]
            if scheme == "leave_one_pressure_out":
                assert not set(X[train, 1]) & set(X[test, 1])
            yield scheme, number, train, test


def create_replay(model, dataset, output, config):
    import csv
    source = V06 / "timeseries.csv"
    source_manifest = json.loads((V06 / "manifest.json").read_text())
    if digest(source) != source_manifest["outputs_sha256"][source.name]:
        raise ValueError("Controller replay source changed")
    with source.open(encoding="utf-8-sig") as stream:
        rows = [r for r in csv.DictReader(stream) if r["scenario_id"] == "baseline"
                and r["plant_kind"] == "same_pressure_CFD_interpolant" and r["control_mode"] == "pi"]
    X = dataset["X"]
    cases, replay_checks = [], []
    for target in (1.5, 2., 2.5, 3., 3.5):
        trajectory = [r for r in rows if float(r["target_mach"]) == target][::config["replay_frame_step"]]
        pressure = float(trajectory[0]["P0_Pa"])
        points = np.array([[float(r["actual_L_mm"]), pressure] for r in trajectory])
        if not supported_mask(X, points).all():
            raise ValueError("Replay would extrapolate beyond full-data field input hull")
        same_pressure = np.flatnonzero(X[:, 1] == pressure)
        same_pressure = same_pressure[np.argsort(X[same_pressure, 0])]
        # Geometry-only interpolation inside the observed same-P0 L bracket.
        wall = PchipInterpolator(X[same_pressure, 0], dataset["wall_heights_m"][same_pressure], axis=0,
                                 extrapolate=False)(points[:, 0])
        if not np.isfinite(wall).all() or np.any(wall <= 0):
            raise ValueError("Replay wall geometry is outside observed support")
        coefficients = predict_coefficients(model, points)
        fields = predict_rom(model, points)
        delta = fields - fields[0]
        if not np.isfinite(fields).all():
            raise ValueError("Nonfinite replay Mach")
        cases.append({
            "target_mach": target, "P0_Pa": pressure,
            "times_s": [float(r["time_s"]) for r in trajectory],
            "actual_L_mm": points[:, 0].tolist(),
            "command_L_mm": [float(r["command_L_mm"]) for r in trajectory],
            "scalar_true_mach": [float(r["true_mach"]) for r in trajectory],
            "scalar_measured_mach": [float(r["measured_mach"]) for r in trajectory],
            "coefficients": coefficients.tolist(), "wall_heights_m": np.round(wall, 9).tolist(),
            "color_min": 0., "color_max": max(target * 1.1, float(fields.max())),
            "delta_limit": max(.002, float(abs(delta).max())),
            "scenario": "baseline", "plant_kind": "same_pressure_CFD_interpolant",
        })
        replay_checks.append({"target_mach": target, "frames": len(trajectory), "all_inputs_in_observed_hull": True,
                              "field_min_mach": float(fields.min()), "field_max_mach": float(fields.max()),
                              "negative_reconstructed_sample_count": int(np.count_nonzero(fields < 0)),
                              "maximum_field_change_mach": float(abs(delta).max()),
                              "geometry_policy": "same-P0 observed wall heights, PCHIP in actual L, no extrapolation",
                              "controller_uses_this_field_for_feedback": False})
    payload = {"meta": {"nx": config["nx"], "ny": config["neta"],
                        "x_edges_m": dataset["x_edges_m"].tolist(), "eta_edges": dataset["eta_edges"].tolist(),
                        "coordinate_system": "x,eta=z/h(x); y=0 upper half plane",
                        "field": "Mach", "training_case_count": len(X), "notes": config["limitations"],
                        "rank": model["details"]["rank"]},
               "rom": {"mean": model["mean"].tolist(), "modes": model["modes"].tolist()}, "cases": cases}
    write_json(output / "viewer_payload.json", payload)
    write_json(output / "replay_validation.json", {"cases": replay_checks, "v06_timeseries_sha256": digest(source),
        "time_axis": "v06 assumed simulation seconds, not transient CFD time",
        "field_physics": "quasi-steady ROM equilibrium field at current actuator L and fixed P0",
        "scope": "Visual replay overlay; longitudinal Mach is NOT the feedback signal of v06's adaptive cross-section Mach controller"})
    from build_viewer import build_viewer
    build_viewer(output / "viewer_payload.json", output / "controller_flow_viewer.html")


def run(output, *, use_registered=False):
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((HERE / "config.json").read_text())
    if not use_registered:
        build_snapshots(output)
    data = dict(np.load(output / "registered_snapshots.npz"))
    X, F, weights = data["X"], data["Mach"], data["weights"]
    ids = data["case_ids"].tolist()
    assert len(ids) == len(X) == len(F) == 27
    eta = (data["eta_edges"][:-1] + data["eta_edges"][1:]) / 2
    write_json(output / "training_protocol.json", {
        "locked_before_fitting_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "case_ids": ids, "one_CFD_condition_one_snapshot": True,
        "spatial_points_are_not_independent_train_test_cases": True,
        "POD_mean_basis_rank_and_GP_refitted_inside_every_fold": True,
        "geometry_registration": "known geometry maps each condition to common (x,eta) reference; geometry is not a field prediction target",
        "rank_and_kernel_policy": "fixed energy target and rank cap; fixed Matérn52 kernel family, marginal likelihood within training fold",
        "holdout_scope": "whole-condition and whole-P0-group field reconstruction; not independent final campaign or transient validation",
    })
    print(f"Fitting full Mach ROM: {F.shape}", flush=True)
    model = train_rom(X, F, weights, config["POD_energy_target"], config["POD_max_rank"])
    full_details = model_details(model)
    fitted = predict_rom(model, X)
    fit_rows = [{"case_id": ids[i], **field_errors(F[i], fitted[i], weights, eta, config["nx"])} for i in range(len(X))]
    cv_rows, fold_details = [], []
    predictions = {scheme: np.empty_like(F) for scheme in config["validation_schemes"]}
    for scheme, number, train, test in make_folds(X):
        fold_model = train_rom(X[train], F[train], weights, config["POD_energy_target"], config["POD_max_rank"])
        np.testing.assert_allclose(fold_model["mean"], F[train].mean(axis=0), atol=1e-12)
        prediction = predict_rom(fold_model, X[test])
        projected = projection(fold_model, F[test])
        predictions[scheme][test] = prediction
        inside = supported_mask(X[train], X[test])
        for j, index in enumerate(test):
            cv_rows.append({"cv_scheme": scheme, "case_id": ids[index], "fold_id": number,
                            "training_count": len(train), "rank": fold_model["details"]["rank"],
                            "in_training_hull": bool(inside[j]),
                            "P0_within_training_range": bool(X[train, 1].min() <= X[index, 1] <= X[train, 1].max()),
                            **field_errors(F[index], prediction[j], weights, eta, config["nx"]),
                            "projection_only_rmse_mach": float(np.sqrt((projected[j] - F[index]) ** 2 @ weights))})
        fold_details.append({"cv_scheme": scheme, "fold_id": number, "train_case_ids": [ids[i] for i in train],
                             "test_case_ids": [ids[i] for i in test], "details": model_details(fold_model)})
        print(f"{scheme} {number + 1}: rank{fold_model['details']['rank']}, held out {[ids[i] for i in test]}", flush=True)
    for scheme in predictions:
        assert len([r for r in cv_rows if r["cv_scheme"] == scheme]) == 27
    comparison = []
    for scheme in predictions:
        rows = [r for r in cv_rows if r["cv_scheme"] == scheme]
        for subset, selected in (("all", rows), ("inside_training_hull", [r for r in rows if r["in_training_hull"]]),
                                 ("outside_training_hull", [r for r in rows if not r["in_training_hull"]]),
                                 ("inside_pressure_range", [r for r in rows if r["P0_within_training_range"]])):
            if not selected:
                continue
            comparison.append({"cv_scheme": scheme, "subset": subset, "case_count": len(selected),
                               "field_rmse_mach": float(np.sqrt(np.mean([r["weighted_rmse_mach"] ** 2 for r in selected]))),
                               "mean_relative_L2_percent": float(np.mean([r["relative_weighted_L2_percent"] for r in selected])),
                               "worst_point_abs_error_mach": max(r["max_abs_error_mach"] for r in selected)})
    (output / "field_model.pkl").write_bytes(pickle.dumps(model, protocol=5))
    write_json(output / "field_model.json", {"details": full_details, "config": config,
        "case_ids": ids, "training_inputs": X.tolist(), "registered_snapshots_sha256": digest(output / "registered_snapshots.npz"),
        "time_dynamics_learned": False, "geometry_normalized": True})
    write_csv(output / "fit_metrics.csv", fit_rows)
    write_csv(output / "cross_validation_metrics.csv", cv_rows)
    write_csv(output / "cross_validation_summary.csv", comparison)
    write_json(output / "fold_diagnostics.json", fold_details)
    np.savez_compressed(output / "field_predictions.npz", fitted=fitted, **predictions)
    timings = []
    for i in range(100):
        start = time.perf_counter()
        predict_rom(model, X[i % len(X):i % len(X) + 1])
        timings.append(time.perf_counter() - start)
    write_json(output / "inference_timing.json", {"samples": 100, "spatial_samples": F.shape[1],
        "median_ms": 1000 * float(np.median(timings)), "p95_ms": 1000 * float(np.percentile(timings, 95)),
        "scope": "warm local Python GP coefficient prediction plus full Mach reconstruction, excludes extraction/training/graphics",
        "CFD_speedup_factor": None, "CFD_run_time_measured": False})
    create_replay(model, data, output, config)
    reloaded = pickle.loads((output / "field_model.pkl").read_bytes())
    np.testing.assert_allclose(predict_rom(reloaded, X), fitted, rtol=0, atol=0)
    write_json(output / "validation.json", {
        "status": "passed", "conditions": 27, "spatial_samples_per_condition": F.shape[1],
        "CV_prediction_fields": len(cv_rows), "training_only_fold_means_checked": True,
        "POD_full_fit_rank": full_details["rank"], "energy_target_met": full_details["energy_target_met"],
        "full_model_all_GP_optimizers_converged": full_details["all_coefficient_optimizers_successful"],
        "folds_with_nonconverged_coefficient_optimizer": sum(not f["details"]["all_coefficient_optimizers_successful"] for f in fold_details),
        "pickle_prediction_roundtrip_exact": True, "new_CFD_solver_runs": 0,
        "transient_flow_or_closed_loop_field_validation": False,
    })
    from report_flow_rom import generate_report
    generate_report(output)
    write_manifest(output)
    print(f"Steady Mach ROM and visual replay complete: {output}", flush=True)


def write_manifest(output):
    write_json(output / "manifest.json", {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs_sha256": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"},
        "source_code_sha256": {p.name: digest(p) for p in sorted(HERE.glob("*.py"))},
        "config_sha256": digest(HERE / "config.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--use-registered", action="store_true")
    args = parser.parse_args()
    run(args.output_dir.resolve(), use_registered=args.use_registered)
