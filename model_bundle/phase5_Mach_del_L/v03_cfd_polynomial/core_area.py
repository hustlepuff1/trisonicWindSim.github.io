"""Cell-resolved, connected Mach-uniformity cores on an actual CFD test plane.

All integrations use piecewise-constant cell fields and physical polygon areas.
No boundary-layer thickness, rectangular usable area, or experimental accuracy
is inferred from the Mach-only masks.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REFERENCE_BOUNDS = (-0.050, 0.0, 0.0, 0.050)
COORDINATE_TOLERANCE_M = 1.0e-9


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    polygon = polygon - polygon[0]
    return abs(float(np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    ))) / 2.0


def clip_rectangle(polygon: np.ndarray, bounds: tuple[float, ...]) -> np.ndarray:
    """Sutherland--Hodgman clipping, retaining exact rectangle intersections."""
    result = np.asarray(polygon, dtype=float)
    ymin, ymax, zmin, zmax = bounds
    for axis, boundary, lower in ((0, ymin, True), (0, ymax, False),
                                  (1, zmin, True), (1, zmax, False)):
        if not len(result):
            break
        output = []
        previous = result[-1]
        previous_inside = previous[axis] >= boundary if lower else previous[axis] <= boundary
        for point in result:
            inside = point[axis] >= boundary if lower else point[axis] <= boundary
            if inside != previous_inside:
                fraction = (boundary - previous[axis]) / (point[axis] - previous[axis])
                cross = previous + fraction * (point - previous)
                cross[axis] = boundary
                output.append(cross)
            if inside:
                output.append(point.copy())
            previous, previous_inside = point, inside
        result = np.asarray(output, dtype=float).reshape(-1, 2)
    return result


def rectangle_weights(polygons: list[np.ndarray], bounds: tuple[float, ...]) -> np.ndarray:
    return np.array([polygon_area(clip_rectangle(p, bounds)) for p in polygons])


def polygon_contains_origin(polygon: np.ndarray) -> bool:
    """Origin containment for the convex sections of the source volume cells."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    cross = edges[:, 0] * (-polygon[:, 1]) - edges[:, 1] * (-polygon[:, 0])
    allowance = COORDINATE_TOLERANCE_M * np.linalg.norm(edges, axis=1)
    return bool(np.all(cross >= -allowance) or np.all(cross <= allowance))


def build_adjacency(polygons: list[np.ndarray]) -> tuple[list[list[int]], dict]:
    """Connect positive-length shared edges, including split/hanging edges.

    Coordinate rounding (0.1 nm) is used only for topology, never for areas.
    Every edge is split at collinear vertices before ownership is matched.
    A common vertex alone therefore never connects two cells.
    """
    vertices = np.unique(np.round(np.concatenate(polygons), 10), axis=0)
    tree = cKDTree(vertices)
    owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell, polygon in enumerate(polygons):
        q = np.round(polygon, 10)
        for first, last in zip(q, np.roll(q, -1, axis=0), strict=True):
            vector = last - first
            length = float(np.linalg.norm(vector))
            if length <= COORDINATE_TOLERANCE_M:
                continue
            ids = np.asarray(tree.query_ball_point(
                (first + last) / 2, length / 2 + COORDINATE_TOLERANCE_M
            ), dtype=int)
            delta = vertices[ids] - first
            projection = delta @ vector / length**2
            distance = np.abs(delta[:, 0] * vector[1] - delta[:, 1] * vector[0]) / length
            use = ((projection >= -1e-8) & (projection <= 1 + 1e-8)
                   & (distance <= COORDINATE_TOLERANCE_M))
            ids = ids[use][np.argsort(projection[use])]
            for a, b in zip(ids[:-1], ids[1:], strict=True):
                if a == b:
                    continue
                key = (int(min(a, b)), int(max(a, b)))
                entry = (cell, 1 if a < b else -1)
                bucket = owners.setdefault(key, [])
                if entry not in bucket:
                    bucket.append(entry)
    adjacency: list[set[int]] = [set() for _ in polygons]
    exterior_twice_area = 0.0
    boundary_segments = []
    for key, cells in owners.items():
        if len(cells) > 2:
            raise ValueError("Nonmanifold or overlapping plane cells on a shared edge")
        if len(cells) == 2:
            (a, sign_a), (b, sign_b) = cells
            if a == b or sign_a == sign_b:
                raise ValueError("Duplicate/overlapping footprints or inconsistent orientation")
            adjacency[a].add(b)
            adjacency[b].add(a)
        else:
            cell, sign = cells[0]
            a, b = vertices[list(key)]
            exterior_twice_area += sign * (a[0] * b[1] - b[0] * a[1])
            boundary_segments.append((a.tolist(), b.tolist(), cell))
    areas = np.array([polygon_area(p) for p in polygons])
    exterior_area = abs(exterior_twice_area) / 2
    if not np.isclose(exterior_area, areas.sum(), atol=1e-9, rtol=1e-7):
        raise ValueError("Boundary-area and cell-area sums disagree")
    if len(component(adjacency, np.ones(len(polygons), dtype=bool), [0])) != len(polygons):
        raise ValueError("Extracted test plane is not edge-connected")
    return [sorted(items) for items in adjacency], {
        "edge_count": len(owners),
        "boundary_area_m2": exterior_area,
        "cell_area_sum_m2": float(areas.sum()),
        "topology_rounding_m": 1e-10,
        "topology_distance_tolerance_m": COORDINATE_TOLERANCE_M,
        "edge_owners": owners,
        "vertices": vertices,
    }


def component(adjacency, allowed: np.ndarray, seeds: list[int]) -> set[int]:
    eligible_seeds = [int(i) for i in seeds if allowed[i]]
    if not eligible_seeds:
        return set()
    seen = {eligible_seeds[0]}
    pending = deque(seen)
    while pending:
        current = pending.popleft()
        for other in adjacency[current]:
            if allowed[other] and other not in seen:
                seen.add(other)
                pending.append(other)
    if not set(eligible_seeds) <= seen:
        raise ValueError("Passing cells at the origin do not define a unique edge-connected core")
    return seen


def region_statistics(weights: np.ndarray, fields: dict, p0_reference: float) -> dict:
    selected = weights > 0
    w = weights[selected]
    area = float(w.sum())
    if area <= 0:
        raise ValueError("Empty integration region")
    data = {name: values[selected] for name, values in fields.items()}
    mean = lambda values: float(w @ values / area)
    mach_mean = mean(data["Ma"])
    std = float(np.sqrt(mean((data["Ma"] - mach_mean)**2)))
    angle = np.degrees(np.arctan2(np.hypot(data["v"], data["w"]), data["u"]))
    total_pressure = data["p"] * (1 + 0.2 * data["Ma"]**2)**3.5
    flux = data["rho"] * data["u"]
    flux_integral = float(w @ flux)
    if flux_integral <= 0:
        raise ValueError("Nonpositive net mass flux in the selected region")
    p0_mass_mean = float(w @ (flux * total_pressure) / flux_integral)
    return {
        "cell_count": int(selected.sum()),
        "area_quarter_m2": area,
        "area_full_equivalent_m2": 4 * area,
        "area_percent_of_nominal_full_section": 100 * 4 * area / 0.25,
        "mean_mach": mach_mean,
        "std_mach": std,
        "mach_cv_percent": 100 * std / mach_mean,
        "max_abs_deviation_from_mean_mach": float(np.max(np.abs(data["Ma"] - mach_mean))),
        "flow_angle_rms_deg": float(np.sqrt(mean(angle**2))),
        "reverse_flow_area_percent": 100 * float(w @ (data["u"] <= 0)) / area,
        "total_pressure_mass_weighted_Pa": p0_mass_mean,
        "total_pressure_deficit_from_reference_percent": 100 * (1 - p0_mass_mean / p0_reference),
        "mass_flow_full_equivalent_kg_s": 4 * flux_integral,
    }


def analyze_plane(path: Path, p0_reference: float,
                  thresholds=(0.005, 0.01, 0.02)) -> dict:
    with np.load(path, allow_pickle=False) as data:
        offsets = data["polygon_offsets"]
        vertices = data["polygon_vertices"]
        polygons = [vertices[a:b].copy() for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
        cached_areas = data["area_m2"].copy()
        fields = {name: data[name].copy() for name in ("u", "v", "w", "rho", "p", "Ma")}
    if any(not np.all(np.isfinite(v)) for v in fields.values()):
        raise ValueError("Nonfinite cell fields")
    if np.any(fields["rho"] <= 0) or np.any(fields["p"] <= 0) or np.any(fields["Ma"] < 0):
        raise ValueError("Nonphysical density, pressure, or Mach field")
    areas = np.array([polygon_area(p) for p in polygons])
    if np.any(areas <= 0) or not np.allclose(areas, cached_areas, rtol=1e-8, atol=1e-15):
        raise ValueError("Invalid or inconsistent cached polygon areas")
    adjacency, topology = build_adjacency(polygons)
    seed = [i for i, polygon in enumerate(polygons) if polygon_contains_origin(polygon)]
    if not seed:
        raise ValueError("No physical cell covers the symmetry-axis origin")
    reference_weights = rectangle_weights(polygons, REFERENCE_BOUNDS)
    if not np.isclose(reference_weights.sum(), 0.05**2, rtol=1e-7, atol=1e-10):
        raise ValueError("Central reference patch is not fully covered")
    reference_mach = float(reference_weights @ fields["Ma"] / reference_weights.sum())
    relative_deviation = np.abs(fields["Ma"] - reference_mach) / reference_mach
    rows, masks = [], {}
    last_mask = np.zeros(len(polygons), dtype=bool)
    for threshold in sorted(thresholds):
        passes = relative_deviation <= threshold
        retained = component(adjacency, passes, seed)
        if not retained:
            raise ValueError(f"No qualifying central core at tolerance {threshold}")
        mask = np.zeros(len(polygons), dtype=bool)
        mask[list(retained)] = True
        if np.any(last_mask & ~mask):
            raise ValueError("Core nesting failed as the tolerance increased")
        last_mask = mask
        stats = region_statistics(np.where(mask, areas, 0.0), fields, p0_reference)
        if abs(stats["mean_mach"] - reference_mach) / reference_mach > threshold + 1e-12:
            raise ValueError("Selected core mean violates its Mach criterion")
        stats.update({
            "tolerance_percent": 100 * threshold,
            "reference_mach": reference_mach,
            "area_percent_of_extracted_plane": 100 * stats["area_quarter_m2"] / areas.sum(),
            "max_deviation_from_reference_percent": float(100 * relative_deviation[mask].max()),
            "excluded_disconnected_passing_area_quarter_m2": float(areas[passes & ~mask].sum()),
        })
        rows.append(stats)
        masks[float(threshold)] = mask
    fixed = {}
    for extent in (0.200, 0.225):
        weights = rectangle_weights(polygons, (-extent, 0.0, 0.0, extent))
        if not np.isclose(weights.sum(), extent**2, rtol=1e-7, atol=1e-10):
            raise ValueError(f"Fixed {extent} m comparison rectangle is not fully covered")
        fixed[extent] = region_statistics(weights, fields, p0_reference)
    full = region_statistics(areas, fields, p0_reference)
    return {
        "polygons": polygons, "fields": fields, "areas": areas,
        "reference_mach": reference_mach, "core_rows": rows, "masks": masks,
        "fixed": fixed, "full": full, "topology": topology,
    }
