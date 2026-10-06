"""Small-data CFD surrogate primitives using NumPy and SciPy only.

Public inputs are always ``[L_mm, P0_reference_Pa]``; pressure is transformed
to its natural logarithm internally. All fitted transforms use training rows
only. The L-only models are references that deliberately ignore pressure.

The exact Gaussian process follows Rasmussen & Williams (2006), Algorithm 2.1
and the Matérn kernels in Chapter 4:
https://gaussianprocess.org/gpml/chapters/RW2.pdf
https://gaussianprocess.org/gpml/chapters/RW4.pdf
Its returned standard deviation is conditional latent-function uncertainty:
it excludes future observation noise, hyperparameter uncertainty, CFD model
discrepancy, and uncertainty from sparsely sampled operating conditions.

Convex-hull membership is a geometric flag, not evidence that the interior is
experimentally validated. These data primarily sample an operating schedule.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


MODEL_NAMES = (
    "l_pchip", "l_quadratic", "affine_2d", "quadratic_2d",
    "gpr_matern32", "gpr_matern52",
)
MODEL_LABELS = {
    "l_pchip": "L-only PCHIP reference",
    "l_quadratic": "L-only quadratic reference",
    "affine_2d": "Affine in L and log(P0)",
    "quadratic_2d": "Quadratic in L and log(P0)",
    "gpr_matern32": "GPR Matern 3/2",
    "gpr_matern52": "GPR Matern 5/2",
}
RANDOM_SEED = 20260924
GP_JITTER = 1e-10
GP_CONVERGENCE_NLL_TOLERANCE = 1e-6
# Log parameters: signal variance, length in L, length in log(P0), noise variance.
GP_BOUNDS = np.log(np.array([[1e-3, 1e3], [0.05, 100.0],
                             [0.05, 100.0], [1e-8, 1e-1]]))


def _validate_X(X: Any, *, allow_empty: bool = False) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("X must have shape (n, 2): [L_mm, P0_reference_Pa].")
    if len(values) == 0 and not allow_empty:
        raise ValueError("Training X must contain at least one row.")
    if not np.all(np.isfinite(values)):
        raise ValueError("L and pressure values must be finite.")
    if np.any(values[:, 1] <= 0):
        raise ValueError("P0_reference_Pa must be strictly positive.")
    return values


def _features(X: np.ndarray) -> np.ndarray:
    return np.column_stack((X[:, 0], np.log(X[:, 1])))


def _fit_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return center, scale


def _design(name: str, Z: np.ndarray) -> np.ndarray:
    one = np.ones(len(Z))
    x, p = Z[:, 0], Z[:, 1]
    if name == "l_quadratic":
        return np.column_stack((one, x, x * x))
    if name == "affine_2d":
        return np.column_stack((one, x, p))
    if name == "quadratic_2d":
        return np.column_stack((one, x, p, x * x, x * p, p * p))
    raise ValueError(f"No polynomial basis for {name!r}.")


def _matern(
    A: np.ndarray, B: np.ndarray, theta: np.ndarray, nu: float,
    *, gradient: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    signal, ell0, ell1, _ = np.exp(theta)
    squared_components = ((A[:, None, :] - B[None, :, :]) /
                          np.array([ell0, ell1])) ** 2
    radius = np.sqrt(np.sum(squared_components, axis=2))
    if nu == 1.5:
        scaled_radius = np.sqrt(3.0) * radius
        decay = np.exp(-scaled_radius)
        kernel = signal * (1.0 + scaled_radius) * decay
        length_derivative_factor = 3.0 * signal * decay
    elif nu == 2.5:
        scaled_radius = np.sqrt(5.0) * radius
        decay = np.exp(-scaled_radius)
        kernel = signal * (1.0 + scaled_radius + scaled_radius ** 2 / 3.0) * decay
        length_derivative_factor = (5.0 / 3.0) * signal * (1.0 + scaled_radius) * decay
    else:
        raise ValueError("Only Matern nu=1.5 or nu=2.5 is supported.")
    if not gradient:
        return kernel
    # Derivatives with respect to log(signal variance) and log(length scales).
    derivatives = np.concatenate((kernel[:, :, None],
                                 length_derivative_factor[:, :, None] *
                                 squared_components), axis=2)
    return kernel, derivatives


def _gp_objective(theta: np.ndarray, Z: np.ndarray, y: np.ndarray,
                  nu: float) -> tuple[float, np.ndarray]:
    kernel, derivatives = _matern(Z, Z, theta, nu, gradient=True)
    noise = np.exp(theta[3])
    covariance = kernel + (noise + GP_JITTER) * np.eye(len(Z))
    try:
        factor = cholesky(covariance, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e100, np.zeros(4)
    alpha = cho_solve((factor, True), y, check_finite=False)
    objective = (0.5 * np.dot(y, alpha) + np.log(np.diag(factor)).sum() +
                 0.5 * len(y) * np.log(2.0 * np.pi))
    weights = (cho_solve((factor, True), np.eye(len(y)), check_finite=False) -
               np.outer(alpha, alpha))
    derivative = np.empty(4)
    derivative[:3] = 0.5 * np.einsum("ij,ijk->k", weights, derivatives)
    derivative[3] = 0.5 * noise * np.trace(weights)
    return float(objective), derivative


def _select_gp_fit(fits: list[Any]) -> tuple[Any, float]:
    """Prefer convergence when likelihood differences are numerical noise.

    This policy examines training marginal likelihood only. An unconverged
    solution that is substantially better is retained and explicitly flagged
    by the caller, so the run can be reviewed instead of silently discarded.
    """
    finite = [fit for fit in fits if np.isfinite(fit.fun) and fit.fun < 1e99]
    if not finite:
        raise RuntimeError("All Gaussian-process optimization starts failed.")
    best = min(finite, key=lambda fit: fit.fun)
    near_converged = [fit for fit in finite if fit.success and
                      fit.fun - best.fun <= GP_CONVERGENCE_NLL_TOLERANCE]
    selected = min(near_converged, key=lambda fit: fit.fun) if near_converged else best
    return selected, float(best.fun)


def train_model(name: str, X: Any, y: Any) -> dict[str, Any]:
    """Fit a pickleable model; no information outside X/y is accessed.

    ``l_pchip`` averages outputs at exactly identical L values, because PCHIP
    requires unique abscissae. This is recorded in ``model_details``. Polynomial
    models use ordinary least squares; deficient rank is reported explicitly.
    GP hyperparameters maximize marginal likelihood at three deterministic
    starts with bounded signal, length scales, and noise variance.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}.")
    X = _validate_X(X).copy()
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) != len(X) or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite one-dimensional array matching X.")
    if len(y) < 2:
        raise ValueError("At least two training rows are required.")
    features = _features(X)
    center, scale = _fit_scaler(features)
    Z = (features - center) / scale
    details: dict[str, Any] = {
        "name": name, "label": MODEL_LABELS[name], "n_training": len(X),
        "feature_order": ["L_mm", "ln(P0_reference_Pa)"],
        "scaler_center": center.tolist(), "scaler_scale": scale.tolist(),
        "training_only_preprocessing": True,
        "uses_pressure": not name.startswith("l_"),
        "selection_eligible": not name.startswith("l_"),
        "selection_scope": ("fixed-pressure model candidate" if not name.startswith("l_")
                            else "L-only reference; ineligible for fixed-pressure operation"),
        "training_min_raw": X.min(axis=0).tolist(),
        "training_max_raw": X.max(axis=0).tolist(),
        "uncertainty": "unavailable",
    }
    model: dict[str, Any] = {
        "format_version": 1, "name": name, "center": center,
        "scale": scale, "details": details,
    }
    if name == "l_pchip":
        unique_L, inverse, counts = np.unique(X[:, 0], return_inverse=True,
                                               return_counts=True)
        if len(unique_L) < 2:
            raise ValueError("PCHIP requires at least two distinct L values.")
        unique_y = np.bincount(inverse, weights=y) / counts
        model.update(knots=unique_L, knot_values=unique_y)
        details.update(n_unique_L=len(unique_L),
                       duplicate_L_policy="arithmetic mean of y at exact duplicate L",
                       extrapolation="end polynomial segment; geometric support flagged separately")
    elif name in ("l_quadratic", "affine_2d", "quadratic_2d"):
        design = _design(name, Z)
        coef, _, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
        model["coefficients"] = coef
        details.update(coefficients=coef.tolist(), design_rank=int(rank),
                       design_columns=int(design.shape[1]),
                       rank_deficient=bool(rank < design.shape[1]),
                       design_singular_values=singular_values.tolist())
    else:
        nu = 1.5 if name == "gpr_matern32" else 2.5
        y_center, y_scale = float(np.mean(y)), float(np.std(y))
        if y_scale < 1e-12:
            y_scale = 1.0
        normalized_y = (y - y_center) / y_scale
        rng = np.random.default_rng(RANDOM_SEED)
        starts = [np.log([1.0, 1.0, 1.0, 1e-5])]
        for _ in range(2):
            starts.append(np.log([1.0, *np.exp(rng.uniform(np.log(0.3), np.log(5.0), 2)), 1e-4]))
        fits = [minimize(_gp_objective, start, args=(Z, normalized_y, nu),
                         jac=True, bounds=GP_BOUNDS, method="L-BFGS-B",
                         options={"maxiter": 250, "maxls": 50, "ftol": 1e-10, "gtol": 1e-6})
                for start in starts]
        best, best_finite_nll = _select_gp_fit(fits)
        theta = best.x.copy()
        signal, ell0, ell1, noise = np.exp(theta)
        covariance = _matern(Z, Z, theta, nu) + (noise + GP_JITTER) * np.eye(len(Z))
        factor = cholesky(covariance, lower=True, check_finite=False)
        alpha = cho_solve((factor, True), normalized_y, check_finite=False)
        model.update(Z_training=Z, theta=theta, nu=nu, factor=factor, alpha=alpha,
                     y_center=y_center, y_scale=y_scale)
        parameter_names = ["signal_variance", "L_length_scale", "logP_length_scale", "noise_variance"]
        bounds_hit = [parameter_names[j] for j in range(4)
                      if np.any(np.isclose(theta[j], GP_BOUNDS[j], atol=1e-4, rtol=0))]
        details.update(
            implementation="exact GP, NumPy/SciPy, GPML Algorithm 2.1",
            matern_nu=nu, normalize_y=True, y_center=y_center, y_scale=y_scale,
            signal_variance_normalized=float(signal),
            length_scales_standardized=[float(ell0), float(ell1)],
            noise_variance_normalized=float(noise), jitter_normalized=GP_JITTER,
            hyperparameter_bounds_natural=np.exp(GP_BOUNDS).tolist(),
            parameter_order=parameter_names, bounds_hit=bounds_hit,
            log_marginal_likelihood=-float(best.fun), random_seed=RANDOM_SEED,
            optimizer_starts=len(starts), selected_optimizer_success=bool(best.success),
            selected_optimizer_message=str(best.message),
            selected_optimizer_policy="prefer successful optimization within 1e-6 NLL of the best finite start; otherwise retain the best finite start and flag convergence status",
            convergence_nll_tolerance=GP_CONVERGENCE_NLL_TOLERANCE,
            best_finite_optimizer_nll=best_finite_nll,
            selected_optimizer_nll_gap=float(best.fun - best_finite_nll),
            selected_optimizer_gradient_inf=float(np.max(np.abs(best.jac))),
            optimizer_runs=[{"success": bool(fit.success), "nll": float(fit.fun),
                             "iterations": int(fit.nit), "message": str(fit.message)}
                            for fit in fits],
            uncertainty="latent function std conditional on fitted kernel/hyperparameters; excludes noise, hyperparameter uncertainty and CFD discrepancy",
        )
    return model


def predict_model(model: dict[str, Any], X: Any, return_std: bool = False):
    """Return means, or (means, std); non-GP std values are NaN, never zero.

    Predictions outside geometric support are allowed so extrapolation errors
    can be scored, but callers must flag them with ``supported_mask``.
    """
    X = _validate_X(X, allow_empty=True)
    name = model["name"]
    if len(X) == 0:
        empty = np.empty(0, dtype=float)
        return (empty, empty.copy()) if return_std else empty
    Z = (_features(X) - model["center"]) / model["scale"]
    std = np.full(len(X), np.nan)
    if name == "l_pchip":
        mean = PchipInterpolator(model["knots"], model["knot_values"],
                                 extrapolate=True)(X[:, 0])
    elif name in ("l_quadratic", "affine_2d", "quadratic_2d"):
        mean = _design(name, Z) @ model["coefficients"]
    elif name in ("gpr_matern32", "gpr_matern52"):
        cross = _matern(Z, model["Z_training"], model["theta"], model["nu"])
        mean = model["y_center"] + model["y_scale"] * (cross @ model["alpha"])
        if return_std:
            solved = solve_triangular(model["factor"], cross.T, lower=True,
                                      check_finite=False)
            variance = np.exp(model["theta"][0]) - np.sum(solved * solved, axis=0)
            std = model["y_scale"] * np.sqrt(np.maximum(variance, 0.0))
    else:
        raise ValueError(f"Unknown fitted model {name!r}.")
    return (mean, std) if return_std else mean


def model_details(model: dict[str, Any]) -> dict[str, Any]:
    """Return independent JSON-serializable fitted metadata."""
    return copy.deepcopy(model["details"])


def supported_mask(training_X: Any, query_X: Any, tolerance: float = 1e-9) -> np.ndarray:
    """Check the training convex hull in scaled [L, ln(P0)] coordinates.

    Scaling uses training feature ranges and is independent of model fits.
    Boundary points are included with the specified absolute tolerance in
    scaled coordinates. Degenerate training sets are treated as a segment or
    a single point without jittering them into artificial 2D support.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Support tolerance must be finite and nonnegative.")
    training = _features(_validate_X(training_X))
    query = _features(_validate_X(query_X, allow_empty=True))
    if len(query) == 0:
        return np.empty(0, dtype=bool)
    low = training.min(axis=0)
    span = np.ptp(training, axis=0)
    span = np.where(span > 1e-12, span, 1.0)
    training = np.unique((training - low) / span, axis=0)
    query = (query - low) / span
    center = np.mean(training, axis=0)
    centered_training, centered_query = training - center, query - center
    _, singular_values, directions = np.linalg.svd(centered_training, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > 1e-12))
    if rank == 0:
        return np.linalg.norm(centered_query, axis=1) <= tolerance
    if rank == 1:
        direction = directions[0]
        train_projection = centered_training @ direction
        query_projection = centered_query @ direction
        perpendicular = centered_query - np.outer(query_projection, direction)
        return ((np.linalg.norm(perpendicular, axis=1) <= tolerance) &
                (query_projection >= train_projection.min() - tolerance) &
                (query_projection <= train_projection.max() + tolerance))
    hull = ConvexHull(training)
    signed_distances = query @ hull.equations[:, :-1].T + hull.equations[:, -1]
    return np.all(signed_distances <= tolerance, axis=1)
