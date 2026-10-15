#!/usr/bin/env python3
"""Recover every native hexahedron's structured material-grid address.

CFD exports permute node/cell numbers between cases. Mesh-edge distances from
the inlet and two symmetry faces recover the full 702 x 72 x 72 node lattice.
No field interpolation or cell deletion occurs in this registration.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

NODE_SHAPE = (702, 72, 72)
CELL_SHAPE = (701, 71, 71)


def recover(xyz, faces, left, right, ncells, chunk=300_000):
    started = time.monotonic()
    n = len(xyz)
    if n != np.prod(NODE_SHAPE) or ncells != np.prod(CELL_SHAPE):
        raise ValueError("Unexpected structured mesh dimensions")
    boundary = (left < 0) | (right < 0)
    bf = faces[boundary]
    points = xyz[bf]
    xmin = float(xyz[:, 0].min())
    selections = [np.all(abs(points[:, :, 0] - xmin) < 1e-8, axis=1),
                  np.all(abs(points[:, :, 1]) < 1e-10, axis=1),
                  np.all(abs(points[:, :, 2]) < 1e-10, axis=1)]
    sources = [np.unique(bf[keep]) for keep in selections]
    expected = (72 * 72, 702 * 72, 702 * 72)
    if tuple(map(len, sources)) != expected:
        raise ValueError(f"Inlet/symmetry patch size mismatch: {list(map(len, sources))}")
    del points, bf
    row, col = faces.ravel(), faces[:, [1, 2, 3, 0]].ravel()
    graph = coo_matrix((np.ones(len(row), dtype=np.uint8), (row, col)), shape=(n, n)).tocsr()
    graph = graph + graph.T
    graph.data[:] = 1
    logical = np.empty((n, 3), dtype=np.int16)
    for axis, ids in enumerate(sources):
        distance = dijkstra(graph, directed=True, indices=ids, min_only=True, unweighted=True)
        if not np.isfinite(distance).all() or np.any(distance != np.floor(distance)):
            raise ValueError("Disconnected or nonintegral node lattice")
        if distance.max() != NODE_SHAPE[axis] - 1:
            raise ValueError("Unexpected lattice extent")
        logical[:, axis] = distance
    del graph, col, distance
    node_key = (logical[:, 0].astype(np.int64) * 72 + logical[:, 1]) * 72 + logical[:, 2]
    if len(np.unique(node_key)) != n or node_key.min() != 0 or node_key.max() != n - 1:
        raise ValueError("Native node -> material node map is not bijective")
    lo = np.full((ncells, 3), np.iinfo(np.int16).max, np.int16)
    hi = np.full((ncells, 3), -1, np.int16)
    counts = np.zeros(ncells, np.int8)
    for start in range(0, len(faces), chunk):
        stop = min(start + chunk, len(faces))
        q = logical[faces[start:stop]]
        qmin, qmax = q.min(axis=1), q.max(axis=1)
        extent = qmax - qmin
        edges = np.abs(q - q[:, [1, 2, 3, 0]]).sum(axis=2)
        if np.any(edges != 1) or np.any(np.sort(extent, axis=1) != [0, 1, 1]):
            raise ValueError("Native face is not a unit logical quadrilateral")
        for owner in (left[start:stop], right[start:stop]):
            keep = owner >= 0
            np.minimum.at(lo, owner[keep], qmin[keep])
            np.maximum.at(hi, owner[keep], qmax[keep])
            np.add.at(counts, owner[keep], 1)
    if np.any(counts != 6) or np.any(hi - lo != 1):
        raise ValueError("Native cell is not a unit logical hexahedron with six faces")
    cell_key = (lo[:, 0].astype(np.int64) * 71 + lo[:, 1]) * 71 + lo[:, 2]
    if len(np.unique(cell_key)) != ncells or cell_key.min() != 0 or cell_key.max() != ncells - 1:
        raise ValueError("Native cell -> material cell map is not bijective")
    # Check each interior face joins the expected two material neighbours.
    for start in range(0, len(faces), chunk):
        stop = min(start + chunk, len(faces))
        l, r = left[start:stop], right[start:stop]
        interior = (l >= 0) & (r >= 0)
        if np.any(np.abs(lo[l[interior]] - lo[r[interior]]).sum(axis=1) != 1):
            raise ValueError("Native adjacency does not match material-grid adjacency")
    audit = {"method": "bijective native mesh-edge distance lattice; no field interpolation",
             "node_shape": list(NODE_SHAPE), "cell_shape": list(CELL_SHAPE),
             "node_count": n, "cell_count": ncells, "every_node_retained": True,
             "every_cell_retained": True, "unit_quad_faces_verified": True,
             "six_face_unit_hexahedra_verified": True, "adjacency_verified": True,
             "boundary_seed_node_counts": list(map(len, sources)),
             "elapsed_seconds": time.monotonic() - started}
    return np.argsort(node_key).astype(np.int32), np.argsort(cell_key).astype(np.int32), audit
