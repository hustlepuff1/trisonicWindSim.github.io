"""Full-volume reference-volume-weighted Mach POD/GP, method of snapshots.

Spatial samples are corresponding CFD cells, not independent training cases.
The fit accepts only the training-case Gram matrix. Validation cases may be
used to evaluate quadratic errors after fitting, never to set the mean,
basis, rank, feature scaling, or GP hyperparameters. The compact model stores
each spatial mode as a linear combination of training snapshots; native-cell
mean/modes are materialized separately for deployment and field export.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

V05 = Path(__file__).resolve().parent.parent / "v05_surrogate_training"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))
from surrogate_models import model_details, predict_model, train_model  # noqa: E402


def _inputs(X: Any) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 2 or not np.isfinite(X).all() or np.any(X[:, 1] <= 0):
        raise ValueError("Inputs must be finite [L_mm, P0_Pa] pairs with P0 > 0.")
    return X


def fit_from_gram(X: Any, gram: Any, *, energy: float = .9999,
                  max_rank: int = 26) -> dict[str, Any]:
    """Fit from F W F.T, where W is a fixed positive normalized cell metric.

The n-by-n centering operation is recomputed at every call. Thus extracting
a fold's raw Gram submatrix before calling this function is equivalent to
fitting weighted POD on that fold's complete training fields.
"""
    X = _inputs(X).copy()
    G = np.asarray(gram, dtype=np.float64)
    n = len(X)
    if n < 2 or G.shape != (n, n) or not np.isfinite(G).all():
        raise ValueError("Require at least two training cases and a finite matching Gram matrix.")
    if not np.allclose(G, G.T, rtol=0, atol=1e-10):
        raise ValueError("Gram matrix must be symmetric.")
    if not 0 < energy <= 1 or int(max_rank) != max_rank or max_rank < 1:
        raise ValueError("Require 0 < energy <= 1 and a positive integer rank cap.")
    C = np.eye(n) - np.full((n, n), 1 / n)
    centered = C @ G @ C
    centered = (centered + centered.T) * .5
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    numerical_tolerance = 100 * np.finfo(float).eps * max(1., float(np.max(np.abs(eigenvalues)))) * n
    if float(eigenvalues.min()) < -numerical_tolerance:
        raise ValueError("Training Gram matrix is not positive semidefinite within roundoff.")
    eigenvalues = np.maximum(eigenvalues, 0)
    numerical_rank = min(n - 1, int(np.count_nonzero(eigenvalues > numerical_tolerance)))
    variance = float(eigenvalues.sum())
    if numerical_rank == 0 or variance <= numerical_tolerance:
        rank = required_rank = 0
        singular = np.empty(0)
        coefficients = np.empty((n, 0))
        transform = np.empty((0, n))
        achieved = 1.
    else:
        required_rank = int(np.searchsorted(np.cumsum(eigenvalues) / variance, energy)) + 1
        rank = min(required_rank, int(max_rank), numerical_rank)
        singular = np.sqrt(eigenvalues[:rank])
        U = eigenvectors[:, :rank].copy()
        # A snapshot-coefficient sign rule avoids materializing huge modes.
        for j in range(rank):
            if U[np.argmax(np.abs(U[:, j])), j] < 0:
                U[:, j] *= -1
        coefficients = U * singular
        transform = (U.T / singular[:, None]) @ C
        achieved = float(eigenvalues[:rank].sum() / variance)
    gps = [train_model("gpr_matern52", X, coefficients[:, j]) for j in range(rank)]
    gp_details = [model_details(gp) for gp in gps]
    details = {
        "model_kind": "scalar_Mach_reference_volume_weighted_centered_POD_Matern52_GP",
        "algorithm": "method_of_snapshots_training_Gram_eigendecomposition",
        "training_count": n, "rank": rank, "numerical_rank": numerical_rank,
        "rank_cap": int(max_rank), "rank_required_for_energy": required_rank,
        "requested_energy_fraction": energy, "achieved_energy_fraction": achieved,
        "energy_target_met": bool(achieved >= energy - 1e-12),
        "rank_cap_active": bool(required_rank > max_rank),
        "total_centered_weighted_energy": variance,
        "singular_values": np.sqrt(eigenvalues).tolist(),
        "eigenvalue_numerical_tolerance": numerical_tolerance,
        "input_columns": ["L_mm", "P0_Pa"],
        "coefficient_GP_features": ["L_mm", "ln(P0_Pa)"],
        "coefficient_model_details": gp_details,
        "all_coefficient_optimizers_successful": all(d["selected_optimizer_success"] for d in gp_details),
        "POD_sign_convention": "largest_absolute_training_coefficient_positive",
        "training_only_preprocessing": True,
        "physical_constraints_enforced": False, "time_dynamics_learned": False,
    }
    return {"format_version": 1, "training_inputs": X, "training_gram": G.copy(),
            "mean_snapshot_weights": np.full(n, 1 / n),
            "mode_snapshot_weights": transform,
            "training_coefficients": coefficients,
            "coefficient_models": gps, "details": details}


def predict_coefficients(model: dict[str, Any], X: Any) -> np.ndarray:
    X = _inputs(X)
    if not model["coefficient_models"]:
        return np.empty((len(X), 0))
    return np.column_stack([predict_model(gp, X) for gp in model["coefficient_models"]])


def snapshot_weights(model: dict[str, Any], coefficients: Any) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    rank = model["details"]["rank"]
    if coefficients.ndim != 2 or coefficients.shape[1] != rank or not np.isfinite(coefficients).all():
        raise ValueError("Coefficient matrix must be finite with retained_rank columns.")
    return model["mean_snapshot_weights"] + coefficients @ model["mode_snapshot_weights"]


def project_coefficients(model: dict[str, Any], training_query_gram: Any) -> np.ndarray:
    """Project query snapshots onto an already frozen training basis.

    Input shape is (n_training, n_query), using the same fixed cell metric.
    This is an error diagnostic; it is never a field prediction.
    """
    cross = np.asarray(training_query_gram, dtype=np.float64)
    n = len(model["training_inputs"])
    if cross.ndim != 2 or cross.shape[0] != n:
        raise ValueError("Projection Gram cross-matrix has incorrect shape.")
    # Mode transforms have zero row sum, but subtract the field mean in
    # physical inner-product space explicitly for a transparent formula.
    mean_cross = np.asarray(model["training_gram"]) @ model["mean_snapshot_weights"]
    return (model["mode_snapshot_weights"] @ (cross - mean_cross[:, None])).T


def quadratic_error(gram: Any, observed_index: int, predicted_weights: Any) -> dict[str, float]:
    """Exact weighted error from a snapshot-linear-combination prediction."""
    G = np.asarray(gram, dtype=np.float64)
    residual = np.asarray(predicted_weights, dtype=np.float64).copy()
    if residual.shape != (len(G),):
        raise ValueError("Prediction weights must index every row of the evaluation Gram.")
    residual[observed_index] -= 1
    squared = float(residual @ G @ residual)
    tolerance = 1e-10 * max(1., float(np.max(np.abs(G))) * float(residual @ residual))
    if squared < -tolerance:
        raise ValueError("Quadratic error is negative beyond floating-point cancellation.")
    squared = max(0., squared)
    observed_energy = float(G[observed_index, observed_index])
    return {"weighted_rmse_mach": float(np.sqrt(squared)),
            "relative_weighted_L2_percent": float(100 * np.sqrt(squared / observed_energy)) if observed_energy > 0 else None}


def load_native_model(folder: str | Path) -> dict[str, Any]:
    """Load trusted local model metadata and memory-map complete native fields."""
    import pickle
    folder = Path(folder)
    model = pickle.loads((folder / "coefficient_model.pkl").read_bytes())
    model["mean"] = np.load(folder / "mean.npy", mmap_mode="r")
    model["modes"] = np.load(folder / "modes.npy", mmap_mode="r")
    return model


def predict_native(model: dict[str, Any], X: Any) -> np.ndarray:
    """Reconstruct all CFD cells; returned shape is (queries, cells)."""
    coefficients = predict_coefficients(model, X)
    return model["mean"] + coefficients @ model["modes"]
