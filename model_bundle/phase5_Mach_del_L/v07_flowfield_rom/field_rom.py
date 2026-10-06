"""Area-weighted POD with independent Matérn-5/2 GP coefficient regressions.

Every row is one CFD condition; every column is the same registered spatial
sample across conditions. Geometry registration and fixed quadrature weights
belong to the caller. Training learns the mean, POD basis/rank, coefficient
normalization and GP hyperparameters from the supplied training rows only.

This is a static field surrogate. Evaluating it along an actuator trajectory
does not add transient flow physics, conservation constraints, or independent
CFD validation. The energy fraction describes training reconstruction, not
out-of-sample accuracy. A rank cap can prevent meeting the energy target.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any

import numpy as np


V05 = Path(__file__).resolve().parent.parent / "v05_surrogate_training"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))
from surrogate_models import (  # noqa: E402
    model_details as _gp_details,
    predict_model as _predict_gp,
    train_model as _train_gp,
)


def _validate_inputs(X: Any, *, allow_empty: bool = False) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("X must have shape (n, 2): [L_mm, P0_Pa].")
    if len(values) == 0 and not allow_empty:
        raise ValueError("Training inputs cannot be empty.")
    if not np.all(np.isfinite(values)) or np.any(values[:, 1] <= 0):
        raise ValueError("X must be finite and P0_Pa must be positive.")
    return values


def _validate_fields(F: Any, *, cells: int | None = None) -> np.ndarray:
    values = np.asarray(F, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Fields must have shape (n_conditions, n_cells > 0).")
    if cells is not None and values.shape[1] != cells:
        raise ValueError("Field columns do not match the registered POD geometry.")
    if not np.all(np.isfinite(values)):
        raise ValueError("All field values must be finite.")
    return values


def train_rom(X: Any, F: Any, weights: Any, energy: float = 0.9999,
              max_rank: int = 12) -> dict[str, Any]:
    """Fit a scalar-field ROM and return a plain pickleable dictionary.

    Positive quadrature weights are normalized internally to sum to one.
    Zero-weight/masked spatial samples must be removed by the caller. The
    covariance-free POD calculation uses an economy-size SVD of the
    ``n_conditions x n_cells`` matrix, never a spatial covariance matrix.
    GP optimizer status is reported without masking failed convergence.
    """
    X = _validate_inputs(X).copy()
    fields = _validate_fields(F)
    if len(X) < 2 or len(fields) != len(X):
        raise ValueError("Require at least two conditions and matching X/F row counts.")
    if isinstance(energy, (bool, str)) or not np.isfinite(energy) or not 0 < energy <= 1:
        raise ValueError("energy must be a finite fraction in (0, 1].")
    if isinstance(max_rank, (bool, str)) or not np.isscalar(max_rank) or int(max_rank) != max_rank or max_rank < 1:
        raise ValueError("max_rank must be a positive integer.")
    max_rank = int(max_rank)
    area = np.asarray(weights, dtype=float)
    if area.ndim != 1 or len(area) != fields.shape[1]:
        raise ValueError("weights must be a vector matching the field columns.")
    if not np.all(np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("Quadrature weights must be finite and strictly positive.")
    total_area = float(np.sum(area))
    if not np.isfinite(total_area):
        raise ValueError("Total quadrature weight must be finite.")
    area = area.copy() / total_area
    sqrt_area = np.sqrt(area)
    mean = fields.mean(axis=0)
    centered = fields - mean
    # Exact duplicate snapshots can acquire a rounding residual through mean().
    if np.all(fields == fields[0]):
        mean = fields[0].copy()
        centered = np.zeros_like(fields)
    weighted = centered * sqrt_area
    _, singular, right = np.linalg.svd(weighted, full_matrices=False)
    variance_energy = float(np.dot(singular, singular))
    data_rms = float(np.sqrt(np.mean(np.sum(fields * fields * area, axis=1))))
    zero_threshold = (100 * np.finfo(float).eps * max(1.0, data_rms)) ** 2
    if variance_energy <= zero_threshold:
        numerical_rank = required_rank = rank = 0
        modes = np.empty((0, fields.shape[1]))
        coefficients = np.empty((len(fields), 0))
        achieved_energy = 1.0
        retained_energy = 0.0
    else:
        tolerance = np.finfo(float).eps * max(weighted.shape) * singular[0]
        numerical_rank = min(len(fields) - 1, int(np.count_nonzero(singular > tolerance)))
        cumulative_energy = np.cumsum(singular ** 2) / variance_energy
        required_rank = min(len(singular), int(np.searchsorted(cumulative_energy, energy)) + 1)
        rank = min(required_rank, max_rank, numerical_rank)
        modes = right[:rank].copy() / sqrt_area
        # Remove arbitrary SVD sign choices for reproducible coefficient targets.
        for mode in modes:
            if mode[np.argmax(np.abs(mode))] < 0:
                mode *= -1.0
        coefficients = centered @ (modes * area).T
        retained_energy = float(np.dot(singular[:rank], singular[:rank]))
        achieved_energy = retained_energy / variance_energy
    coefficient_models = [_train_gp("gpr_matern52", X, coefficients[:, j]) for j in range(rank)]
    coefficient_details = [_gp_details(model) for model in coefficient_models]
    details = {
        "model_kind": "scalar_field_area_weighted_centered_POD_with_Matern52_GP_coefficients",
        "n_training_conditions": len(X), "n_cells": fields.shape[1],
        "input_columns": ["L_mm", "P0_Pa"],
        "coefficient_GP_features": ["L_mm", "ln(P0_Pa)"],
        "rank": rank, "numerical_rank": numerical_rank, "rank_cap": max_rank,
        "requested_energy_fraction": float(energy),
        "rank_required_for_energy": required_rank,
        "achieved_energy_fraction": float(achieved_energy),
        "energy_target_met": bool(achieved_energy >= energy - 1e-12),
        "rank_cap_active": bool(required_rank > max_rank),
        "total_centered_weighted_energy": variance_energy,
        "retained_centered_weighted_energy": retained_energy,
        "discarded_centered_weighted_energy": max(0.0, variance_energy - retained_energy),
        "zero_variance_threshold": zero_threshold,
        "constant_snapshot_model": rank == 0,
        "singular_values": singular.tolist(),
        "quadrature_weight_sum_before_normalization": total_area,
        "quadrature_weight_sum": float(area.sum()),
        "weight_policy": "immutable caller-supplied registered geometry, normalized sum one",
        "training_only_preprocessing": True,
        "POD_sign_convention": "largest-absolute spatial mode entry positive",
        "coefficient_GP_model": "gpr_matern52",
        "coefficient_optimizer_success": [d["selected_optimizer_success"] for d in coefficient_details],
        "all_coefficient_optimizers_successful": all(d["selected_optimizer_success"] for d in coefficient_details),
        "coefficient_model_details": coefficient_details,
        "physical_constraints_enforced": False,
        "time_dynamics_learned": False,
    }
    return {
        "format_version": 1, "mean": mean.copy(), "modes": modes,
        "weights": area, "training_inputs": X,
        "coefficient_models": coefficient_models, "details": details,
    }


def predict_coefficients(model: dict[str, Any], X: Any) -> np.ndarray:
    """Predict one row of POD coefficients per raw [L_mm, P0_Pa] input."""
    inputs = _validate_inputs(X, allow_empty=True)
    if not model["coefficient_models"]:
        return np.empty((len(inputs), 0))
    return np.column_stack([_predict_gp(gp, inputs) for gp in model["coefficient_models"]])


def reconstruct(model: dict[str, Any], coefficients: Any) -> np.ndarray:
    """Reconstruct physical scalar values from an (n_query, rank) matrix."""
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(model["modes"]):
        raise ValueError("coefficients must have shape (n_query, retained_rank).")
    if not np.all(np.isfinite(values)):
        raise ValueError("Coefficients must be finite.")
    return model["mean"] + values @ model["modes"]


def predict_rom(model: dict[str, Any], X: Any) -> np.ndarray:
    """Return static reconstructed fields, with shape (n_query, n_cells)."""
    return reconstruct(model, predict_coefficients(model, X))


def projection(model: dict[str, Any], F: Any) -> np.ndarray:
    """Project supplied fields onto the frozen basis, without fitting to them.

    This isolates spatial truncation error; it does not measure coefficient-GP
    prediction accuracy and must not be reported as the full surrogate error.
    """
    fields = _validate_fields(F, cells=len(model["mean"]))
    coefficients = (fields - model["mean"]) @ (model["modes"] * model["weights"]).T
    return reconstruct(model, coefficients)


def model_details(model: dict[str, Any]) -> dict[str, Any]:
    """Return independent JSON-compatible metadata, including convergence flags."""
    return copy.deepcopy(model["details"])
