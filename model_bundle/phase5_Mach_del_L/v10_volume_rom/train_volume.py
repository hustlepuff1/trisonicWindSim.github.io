#!/usr/bin/env python3
"""Train and audit the canonical 27-case full-volume Mach POD/GP extension.

The original 2D ROM and prospective frozen predictions are never modified.
Two existing independent CFD cases are retrospective held-out checks for
this new model; they never select the basis, rank, or GP hyperparameters.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from volume_rom import (fit_from_gram, predict_coefficients, predict_native,
                        project_coefficients, quadratic_error, snapshot_weights)
from surrogate_models import supported_mask

HERE = Path(__file__).resolve().parent


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def region_masks(x):
    return {"full_volume": np.ones(len(x), dtype=bool),
            "upstream_x_le_0": x <= 0,
            "downstream_x_gt_0": x > 0,
            "x_band_minus_0p5_to_0p5": (x >= -.5) & (x <= .5)}


def gram_cache_binding(cache):
    """Bind reusable integrals to actual arrays, row metadata and builder code."""
    names = ("registered_mach.npy", "registered_reference_volumes.npy",
             "registered_reference_centers.npy", "cases.json")
    return {"schema_version": 2,
            "input_files_sha256": {name: digest(cache / name) for name in names},
            "gram_builder_script_sha256": digest(Path(__file__).resolve()),
            "weight_policy": "fixed_M17_reference_physical_volume_weights"}


def build_grams(fields, weights, regions, block_size=65536):
    """Accumulate small raw Grams without loading a double 29×3.53M matrix."""
    if fields.ndim != 2 or weights.shape != (fields.shape[1],):
        raise ValueError("Mach matrix and cell weights disagree.")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Every corresponding CFD cell requires a positive finite discrete weight.")
    totals = {key: float(weights[mask].sum()) for key, mask in regions.items()}
    if any(total <= 0 for total in totals.values()):
        raise ValueError("A named evaluation region is empty.")
    grams = {key: np.zeros((len(fields), len(fields)), dtype=float) for key in regions}
    for start in range(0, fields.shape[1], block_size):
        stop = min(start + block_size, fields.shape[1])
        block = np.asarray(fields[:, start:stop], dtype=float)
        if not np.isfinite(block).all():
            raise ValueError(f"Nonfinite Mach value in cells {start}:{stop}.")
        local_weights = weights[start:stop]
        for key, mask in regions.items():
            select = mask[start:stop]
            if not select.any():
                continue
            chosen = block[:, select]
            grams[key] += (chosen * (local_weights[select] / totals[key])) @ chosen.T
    return grams, totals


def evaluate(model, train, test, X, grams, ids, scheme):
    """Evaluate complete fields using quadratic forms, separately by region."""
    coefficients = predict_coefficients(model, X[test])
    predictions = snapshot_weights(model, coefficients)
    projections = snapshot_weights(model, project_coefficients(model, grams["full_volume"][np.ix_(train, test)]))
    supported = supported_mask(X[train], X[test])
    rows = []
    for local, index in enumerate(test):
        weights = np.zeros(len(X))
        weights[train] = predictions[local]
        projected_weights = np.zeros(len(X))
        projected_weights[train] = projections[local]
        row = {"scheme": scheme, "case_id": ids[index],
               "training_count": len(train), "rank": model["details"]["rank"],
               "in_training_input_hull": bool(supported[local]),
               "L_mm": float(X[index, 0]), "P0_Pa": float(X[index, 1])}
        for region, gram in grams.items():
            metrics = quadratic_error(gram, int(index), weights)
            row.update({f"{region}_{name}": value for name, value in metrics.items()})
            row[f"{region}_projection_only_rmse_mach"] = quadratic_error(gram, int(index), projected_weights)["weighted_rmse_mach"]
        rows.append(row)
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["scheme"], []).append(row)
    output = []
    for scheme, selected in groups.items():
        for subset, subset_rows in (("all", selected),
                                    ("inside_training_input_hull", [row for row in selected if row["in_training_input_hull"]]),
                                    ("outside_training_input_hull", [row for row in selected if not row["in_training_input_hull"]])):
            if not subset_rows:
                continue
            result = {"scheme": scheme, "subset": subset, "case_count": len(subset_rows)}
            for key in subset_rows[0]:
                if key.endswith("weighted_rmse_mach") or key.endswith("projection_only_rmse_mach"):
                    result[key] = float(np.sqrt(np.mean([row[key] ** 2 for row in subset_rows])))
                elif key.endswith("relative_weighted_L2_percent"):
                    result[f"mean_{key}"] = float(np.mean([row[key] for row in subset_rows]))
            output.append(result)
    return output


def write_report(path, model, rows, summaries, native_rows):
    details = model["details"]
    lines = ["전체 체적 마하수 POD–GPR 학습·검증 결과", "",
             f"학습: 정상 CFD 27조건, 각 조건의 전체 1/4 체적 {model['native_cell_count']:,}셀.",
             "대칭 복사는 표시 단계에만 적용하며 4배 중복 학습하지 않음.",
             f"POD 차수: {details['rank']}; 훈련장 변동 에너지 보존율: {100 * details['achieved_energy_fraction']:.6f}%.",
             "고정 M17 기준 형상의 양의 물리 셀 체적을 정규화한 내적을 사용함.",
             "이 에너지 보존율은 새로운 조건의 예측 정확도가 아님.", "",
             "기존 독립 CFD를 사용한 새 3D 모델의 사후 평가 (학습·차수 선택에서 제외):"]
    for row in rows:
        if row["scheme"] == "independent_retrospective_heldout":
            lines.append(f"  {row['case_id']}: 전체 기준 체적 RMSE {row['full_volume_weighted_rmse_mach']:.6g} Mach, "
                         f"상대 L2 {row['full_volume_relative_weighted_L2_percent']:.6g}%; "
                         f"x≤0 RMSE {row['upstream_x_le_0_weighted_rmse_mach']:.6g}, "
                         f"x>0 RMSE {row['downstream_x_gt_0_weighted_rmse_mach']:.6g}.")
    lines += ["", "조건 전체를 제외하는 LOCO 검증 (각 fold에서 평균·기저·차수·정규화·GP 재학습):"]
    for row in summaries:
        if row["scheme"] == "leave_one_condition_out":
            lines.append(f"  {row['subset']} ({row['case_count']}조건): "
                         f"전체 기준 체적 RMSE {row['full_volume_weighted_rmse_mach']:.6g} Mach, "
                         f"평균 상대 L2 {row['mean_full_volume_relative_weighted_L2_percent']:.6g}%.")
    lines += ["", "전체 셀에서의 독립 조건 예측값 검사 (음수 절삭 없음):"]
    for row in native_rows:
        lines.append(f"  {row['case_id']}: 최대 절대 오차 {row['max_abs_error_mach']:.6g} Mach; "
                     f"예측 범위 [{row['minimum_predicted_mach']:.6g}, {row['maximum_predicted_mach']:.6g}], "
                     f"음의 예측 {row['negative_prediction_cell_count']:,}셀.")
    lines += ["", "해석 범위:",
              "정상 스냅샷 기반 준정상 체적장 모델이며 비정상 CFD나 실험 동기화 검증이 아님.",
              "오차는 정합된 셀의 장 예측 오차이며 브라우저의 형상 보간 오차를 포함하지 않음.",
              "학습 조건은 기존 L/P0 연동 운전 경로에 집중되어 있고 임의 독립 L/P0 조합을 검증한 것이 아님.",
              "기존 2D 동결 예측·모델은 변경하지 않음."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize(model, fields, train, output, block_size=65536):
    """Save native fields with bounded RAM and audit float32 storage error."""
    rank, cells = model["details"]["rank"], fields.shape[1]
    mean = np.lib.format.open_memmap(output / "mean.npy", mode="w+", dtype="<f4", shape=(cells,))
    modes = np.lib.format.open_memmap(output / "modes.npy", mode="w+", dtype="<f4", shape=(rank, cells))
    max_mean_storage = max_mode_storage = 0.
    for start in range(0, cells, block_size):
        stop = min(start + block_size, cells)
        block = np.asarray(fields[train, start:stop], dtype=float)
        mean64 = model["mean_snapshot_weights"] @ block
        modes64 = model["mode_snapshot_weights"] @ block
        mean[start:stop] = mean64
        modes[:, start:stop] = modes64
        max_mean_storage = max(max_mean_storage, float(np.max(np.abs(mean64 - mean[start:stop]))))
        if rank:
            max_mode_storage = max(max_mode_storage, float(np.max(np.abs(modes64 - modes[:, start:stop]))))
    mean.flush()
    modes.flush()
    return mean, modes, {"mean_max_abs_float32_rounding_mach": max_mean_storage,
                          "mode_max_abs_float32_rounding": max_mode_storage,
                          "storage_dtype": "little_endian_float32"}


def native_heldout_metrics(model, fields, X, heldout, ids):
    """Inspect all native predictions, retaining unclipped Mach values."""
    rows = []
    for index in heldout:
        prediction = predict_native(model, X[index:index + 1])[0]
        truth = np.asarray(fields[index], dtype=float)
        error = prediction - truth
        row = {"case_id": ids[index], "weight_policy": "equal_corresponding_cell_weights",
               "max_abs_error_mach": float(np.max(np.abs(error))),
               "minimum_predicted_mach": float(np.min(prediction)),
               "maximum_predicted_mach": float(np.max(prediction)),
               "negative_prediction_cell_count": int(np.count_nonzero(prediction < 0))}
        rmse = float(np.sqrt(np.mean(error ** 2)))
        row["full_volume_cell_weighted_rmse_mach"] = rmse
        row["full_volume_cell_weighted_relative_L2_percent"] = float(100 * rmse / np.sqrt(np.mean(truth ** 2)))
        rows.append(row)
    return rows


def run(cache, output, *, energy=.9999, max_rank=26, reuse_grams=False):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    extraction_path = cache / "extraction_manifest.json"
    extraction = json.loads(extraction_path.read_text())
    if extraction.get("complete") is not True or extraction.get("topology_correspondence_verified") is not True:
        raise ValueError("Full-cell training requires verified topology and cell correspondence for all cases.")
    if extraction.get("reference_volume_weights_positive_and_valid") is not True:
        raise ValueError("Reference weights must have verified positive closed physical-cell geometry.")
    if extraction.get("all_sources_full_sha256_verified") is not True:
        raise ValueError("Every original CFD source must match its archived full SHA-256 digest.")
    cases = json.loads((cache / "cases.json").read_text())
    ids = [row["case_id"] for row in cases]
    X = np.array([[row["L_mm"], row["P0_Pa"]] for row in cases], dtype=float)
    train = np.array([i for i, row in enumerate(cases) if row["training"]], dtype=int)
    heldout = np.array([i for i, row in enumerate(cases) if not row["training"]], dtype=int)
    if len(cases) != 29 or len(train) != 27 or {ids[i] for i in heldout} != {"M175", "M325"}:
        raise ValueError("Expected canonical 27 training conditions plus excluded M175/M325.")
    # Never use native per-case ordering directly: node/cell numbering differs.
    F = np.load(cache / "registered_mach.npy", mmap_mode="r")
    centers = np.load(cache / "registered_reference_centers.npy", mmap_mode="r")
    if F.shape != (29, 3533741):
        raise ValueError(f"Unexpected full-volume dimensions: {F.shape}.")
    weights = np.load(cache / "registered_reference_volumes.npy", mmap_mode="r")
    regions = region_masks(centers[:, 0])
    protocol = {"locked_before_fitting_utc": datetime.now(timezone.utc).isoformat(),
                "training_case_ids": [ids[i] for i in train], "heldout_case_ids": [ids[i] for i in heldout],
                "POD_energy_target": energy, "POD_max_rank": max_rank,
                "rank_selection": "training_energy_only; no heldout or CV error tuning",
                "mean_basis_rank_input_scaling_and_GP_refit_in_each_fold": True,
                "one_CFD_condition_one_snapshot": True,
                "native_cells_per_snapshot": F.shape[1], "field": "Mach",
                "weight_policy": "fixed M17 reference physical cell volumes, normalized per evaluation region; every native cell has positive weight",
                "weight_choice_reason": "M17 has higher coordinate precision and verified positive closed cell geometry for all cells; no fabricated volume floor or omitted Mach cell",
                "weight_scope": "physical-volume quadrature in the fixed M17 reference geometry, not each deformed case's physical geometry",
                "registration": "verified bijective structured logical-grid correspondence recovered from boundary-anchored mesh topology; native cell IDs differ across cases",
                "independent_case_label": "retrospective heldout for this new 3D model; not the prior frozen prospective 2D predictions",
                "CV_scheme": "leave_one_whole_CFD_condition_out",
                "symmetry": "train original quarter volume once; scalar Mach reflected only for display",
                "display_and_model_decoupled": "full native cells train the model; browser may select a smaller rendering sample",
                "limitations": ["steady snapshots only", "no experimental synchronization or transient flow identification",
                                "CFD-discrepancy and geometry-interpolation errors excluded from registered-field metrics",
                                "operating-condition coverage concentrated along existing coordinated L/P0 schedule"],
                "extraction_manifest_sha256": digest(extraction_path)}
    write_json(output / "training_protocol.json", protocol)
    gram_path = output / "gram_matrices.npz"
    gram_metadata_path = output / "gram_metadata.json"
    cache_binding = gram_cache_binding(cache)
    if reuse_grams and gram_path.exists() and gram_metadata_path.exists():
        gram_meta = json.loads(gram_metadata_path.read_text())
        if gram_meta["extraction_manifest_sha256"] != protocol["extraction_manifest_sha256"]:
            raise ValueError("Cached Gram data do not match the extraction manifest.")
        if gram_meta.get("cache_binding") != cache_binding:
            raise ValueError("Cached Gram data do not match the registered arrays, case order or builder implementation; rebuild without --reuse-grams.")
        with np.load(gram_path) as archive:
            grams = {key: archive[key] for key in archive.files}
        totals = gram_meta["region_reference_volumes_m3"]
    else:
        print(f"Accumulating full native-volume Gram matrices: {F.shape}", flush=True)
        grams, totals = build_grams(F, weights, regions)
        np.savez(gram_path, **grams)
        write_json(gram_metadata_path, {"extraction_manifest_sha256": protocol["extraction_manifest_sha256"],
                                       "cache_binding": cache_binding,
                                       "region_reference_volumes_m3": totals,
                                       "region_cell_counts": {key: int(mask.sum()) for key, mask in regions.items()}})
    G = grams["full_volume"]
    print("Fitting full 27-case model", flush=True)
    model = fit_from_gram(X[train], G[np.ix_(train, train)], energy=energy, max_rank=max_rank)
    model["training_case_ids"] = [ids[i] for i in train]
    model["training_indices"] = train
    model["native_cell_count"] = F.shape[1]
    model["details"].update(
        native_cell_count=F.shape[1], reference_geometry_case="M17",
        reference_quarter_volume_m3=float(weights.sum()),
        quadrature_weight_policy="fixed positive physical cell volumes in registered M17 reference geometry",
        registration="bijective boundary-anchored logical-grid permutation of every native cell",
        field="Mach", spatial_scope="entire quarter CFD domain including downstream diffuser")
    rows = evaluate(model, train, train, X, grams, ids, "training_fit")
    rows += evaluate(model, train, heldout, X, grams, ids, "independent_retrospective_heldout")
    write_json(output / "model.json", {"details": model["details"], "training_case_ids": model["training_case_ids"],
                                      "training_inputs": X[train].tolist(), "native_cell_count": F.shape[1]})
    (output / "coefficient_model.pkl").write_bytes(pickle.dumps(model, protocol=5))
    print(f"Materializing full model: rank {model['details']['rank']}", flush=True)
    model["mean"], model["modes"], storage = materialize(model, F, train, output)
    write_json(output / "storage_precision.json", storage)
    native_rows = native_heldout_metrics(model, F, X, heldout, ids)
    write_csv(output / "heldout_native_cell_metrics.csv", native_rows)
    fold_details = []
    for fold, omitted in enumerate(train):
        selected = train[train != omitted]
        fold_model = fit_from_gram(X[selected], G[np.ix_(selected, selected)], energy=energy, max_rank=max_rank)
        fold_rows = evaluate(fold_model, selected, np.array([omitted]), X, grams, ids, "leave_one_condition_out")
        rows += fold_rows
        fold_details.append({"omitted_case": ids[omitted], "training_case_ids": [ids[i] for i in selected],
                             "details": fold_model["details"]})
        print(f"LOCO {fold + 1:02d}/27 {ids[omitted]}: rank={fold_model['details']['rank']}, "
              f"RMSE={fold_rows[0]['full_volume_weighted_rmse_mach']:.6g}", flush=True)
        write_csv(output / "field_metrics.csv", rows)
        write_json(output / "fold_diagnostics.json", fold_details)
    summaries = summarize(rows)
    write_csv(output / "validation_summary.csv", summaries)
    write_report(output / "report.txt", model, rows, summaries, native_rows)
    # Compare direct full-native inference to the exact snapshot Gram error.
    native_check = []
    for index in heldout:
        prediction = predict_native(model, X[index:index + 1])[0]
        error = prediction - F[index]
        direct = float(np.sqrt(np.dot(error ** 2, weights / weights.sum())))
        gram_value = next(row["full_volume_weighted_rmse_mach"] for row in rows
                          if row["scheme"] == "independent_retrospective_heldout" and row["case_id"] == ids[index])
        if not np.isclose(direct, gram_value, atol=2e-6, rtol=1e-5):
            raise ValueError("Float32 native model and exact Gram diagnostic disagree.")
        native_check.append({"case_id": ids[index], "native_rmse_mach": direct,
                             "float64_Gram_rmse_mach": gram_value})
    timings = []
    for index in train[::6]:
        tick = time.perf_counter()
        prediction = predict_native(model, X[index:index + 1])
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite native inference result.")
        timings.append(1000 * (time.perf_counter() - tick))
    report = {"status": "completed", "training_condition_count": 27, "heldout_condition_count": 2,
              "native_cell_count": F.shape[1], "rank": model["details"]["rank"],
              "achieved_energy_fraction": model["details"]["achieved_energy_fraction"],
              "full_model_all_GP_optimizers_converged": model["details"]["all_coefficient_optimizers_successful"],
              "folds_with_nonconverged_GP_optimizers": sum(not fold["details"]["all_coefficient_optimizers_successful"] for fold in fold_details),
              "full_native_vs_Gram_checks": native_check,
              "inference_timing": {"scope": "local Python coefficient GP plus all native cells, includes finite-value check; warm file cache",
                                   "sample_count": len(timings), "median_ms": float(np.median(timings)),
                                   "p95_ms": float(np.percentile(timings, 95))},
              "elapsed_seconds": time.perf_counter() - started,
              "validation_summary": summaries, "heldout_native_cell_metrics": native_rows,
              "weight_policy": "fixed_M17_reference_physical_volume_weights",
              "new_CFD_runs": 0, "transient_or_experimental_validation": False}
    write_json(output / "validation.json", report)
    outputs = ["coefficient_model.pkl", "mean.npy", "modes.npy", "model.json", "training_protocol.json",
               "gram_matrices.npz", "gram_metadata.json", "field_metrics.csv", "validation_summary.csv",
               "fold_diagnostics.json", "heldout_native_cell_metrics.csv", "storage_precision.json", "validation.json", "report.txt"]
    write_json(output / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                          "model_scope": "full_native_quarter_volume_Mach_only",
                                          "extraction_manifest_sha256": protocol["extraction_manifest_sha256"],
                                          "implementation_sha256": {name: digest(HERE / name) for name in
                                                                     ("volume_rom.py", "train_volume.py", "test_volume_rom.py")},
                                          "outputs_sha256": {name: digest(output / name) for name in outputs}})
    print(f"Full-volume training and validation complete: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE / "results/cache")
    parser.add_argument("--output", type=Path, default=HERE / "results/model")
    parser.add_argument("--energy", type=float, default=.9999)
    parser.add_argument("--max-rank", type=int, default=26)
    parser.add_argument("--reuse-grams", action="store_true")
    args = parser.parse_args()
    run(args.cache, args.output, energy=args.energy, max_rank=args.max_rank, reuse_grams=args.reuse_grams)
