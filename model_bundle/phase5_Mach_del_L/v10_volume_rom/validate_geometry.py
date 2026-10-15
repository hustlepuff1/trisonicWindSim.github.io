#!/usr/bin/env python3
"""Compare predicted and held-out roof geometry at matched physical x."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from extract_volume import HERE, write_json


def roof(cache, case):
    native = cache / "native" / case.lower()
    node_map = np.load(native / "structured_to_native_nodes.npy", mmap_mode="r")
    # k=71 is the actual upper-wall patch; j spans the quarter width.
    mapping = node_map.reshape(702, 72, 72)[:, :, -1]
    return np.load(native / "nodes.npy", mmap_mode="r")[mapping]


def summary(error, mask):
    values = error[mask]
    return {"sample_count": int(len(values)), "max_abs_height_error_m": float(np.max(np.abs(values))),
            "median_abs_height_error_m": float(np.median(np.abs(values))),
            "mean_abs_height_error_m": float(np.mean(np.abs(values))),
            "mean_signed_prediction_minus_actual_m": float(np.mean(values))}


def main():
    cache = HERE / "results/cache"
    rows = json.loads((cache / "cases.json").read_text())
    training = sorted([r for r in rows if r["training"]], key=lambda r: r["L_mm"])
    input_l = np.array([r["L_mm"] for r in training])
    report = {"method": "Linear interpolation in L between bracketing training geometries, followed by per-span interpolation of predicted physical upper-wall z(x) to held-out physical x. No comparison of interior mesh placement.",
              "wall_patch": "upper roof k=71, all 72 quarter-span node tracks; symmetry track j=0 also reported separately",
              "statistics": "unweighted roof-node statistics over the common physical-x support; no extrapolation",
              "regions": {"inlet_and_nozzle": "x<=0 m", "diffuser": "x>0 m"},
              "cases": []}
    for row in [r for r in rows if not r["training"]]:
        index = int(np.searchsorted(input_l, row["L_mm"]))
        lower, upper = training[index - 1], training[index]
        fraction = (row["L_mm"] - lower["L_mm"]) / (upper["L_mm"] - lower["L_mm"])
        prediction = (1 - fraction) * roof(cache, lower["case_id"]) + fraction * roof(cache, upper["case_id"])
        actual = roof(cache, row["case_id"])
        error = np.empty(actual.shape[:2])
        support = np.empty(actual.shape[:2], dtype=bool)
        for j in range(72):
            xp, zp = prediction[:, j, 0], prediction[:, j, 2]
            xa, za = actual[:, j, 0], actual[:, j, 2]
            if np.any(np.diff(xp) <= 0) or np.any(np.diff(xa) <= 0):
                raise ValueError("Roof cannot be represented by monotone x tracks")
            support[:, j] = (xa >= xp[0] - 1e-12) & (xa <= xp[-1] + 1e-12)
            error[:, j] = np.interp(xa, xp, zp) - za
        regions = {"inlet_and_nozzle": actual[:, :, 0] <= 0, "diffuser": actual[:, :, 0] > 0}
        stats = {name: summary(error, support & mask) for name, mask in regions.items()}
        symmetry = {name: summary(error[:, 0], support[:, 0] & mask[:, 0]) for name, mask in regions.items()}
        largest = np.unravel_index(np.argmax(np.where(support, np.abs(error), -1)), error.shape)
        report["cases"].append({"case_id": row["case_id"], "L_mm": row["L_mm"],
            "lower_training_case": lower["case_id"], "upper_training_case": upper["case_id"],
            "fraction_toward_upper_L": float(fraction), "all_span_tracks": stats,
            "y0_symmetry_track": symmetry, "unsupported_endpoint_node_count": int((~support).sum()),
            "max_error_location_actual_xyz_m": actual[largest].tolist(),
            "outlet": {"physical_x_m": float(actual[-1, 0, 0]),
                       "actual_mean_roof_height_m": float(actual[-1, :, 2].mean()),
                       "predicted_mean_roof_height_m": float(prediction[-1, :, 2].mean()),
                       "mean_height_prediction_minus_actual_m": float(error[-1].mean()),
                       "max_abs_height_error_m": float(np.abs(error[-1]).max())}})
    write_json(HERE / "results/geometry_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
