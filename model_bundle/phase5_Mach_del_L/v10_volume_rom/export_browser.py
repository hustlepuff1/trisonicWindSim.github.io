#!/usr/bin/env python3
"""Export a light display sampling of the trained entire-volume Mach ROM.

The statistical model is fitted on all 3,533,741 cells. Only browser display is
decimated. Coordinates come from the native 3D nodes at each observed geometry;
display scalar nodes average adjacent cell values, a fixed linear operation on
the full-volume mean/modes. No two-dimensional field is extruded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from register_volume import NODE_SHAPE, CELL_SHAPE
from volume_rom import predict_coefficients

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def select_stations(values, count, extra=()):
    targets = np.r_[np.linspace(values[0], values[-1], count), extra]
    return np.unique(np.argmin(abs(values[:, None] - targets), axis=0))


def sample_geometry(cache, cases, output):
    ref = cache / 'native/m17'
    native_nodes = np.load(ref / 'nodes.npy', mmap_mode='r')
    node_order = np.load(ref / 'structured_to_native_nodes.npy', mmap_mode='r')
    xyz = native_nodes[node_order].reshape(*NODE_SHAPE, 3)
    x_axis = np.median(xyz[..., 0], axis=(1, 2))
    y_axis = np.median(xyz[..., 1], axis=(0, 2))
    eta_z = np.median(xyz[..., 2] / xyz[:, :, -1:, 2], axis=(0, 1))
    indices = [select_stations(x_axis, 241, [-.5, 0., .5]),
               select_stations(y_axis, 21, y_axis[-1] * np.array([.97, .99, .999])),
               select_stations(eta_z, 29, [.97, .99, .999])]
    mesh = np.meshgrid(*indices, indexing='ij')
    nodes = np.ravel_multi_index(tuple(mesh), NODE_SHAPE).ravel()
    shape = [len(a) for a in indices]
    rows = sorted([r for r in cases if r['training']], key=lambda r: r['L_mm'])
    lengths = [r['L_mm'] for r in rows]
    if np.any(np.diff(lengths) <= 0):
        raise ValueError('Distinct ascending observed lengths are required')
    coords = np.empty((len(rows), len(nodes), 3), dtype='<f4')
    bounds = []
    for i, row in enumerate(rows):
        folder = cache / 'native' / row['case_id'].lower()
        native = np.load(folder / 'nodes.npy', mmap_mode='r')
        order = np.load(folder / 'structured_to_native_nodes.npy', mmap_mode='r')
        coords[i] = native[order[nodes]]
        bounds.append({'case_id': row['case_id'], 'min_m': coords[i].min(axis=0).tolist(),
                       'max_m': coords[i].max(axis=0).tolist()})
    if not np.isfinite(coords).all():
        raise ValueError('Nonfinite browser coordinates')
    coords.tofile(output / 'coordinates.f32')
    # At physical boundary nodes, the repeated clipped neighbour is intentional:
    # averaging the eight entries reduces to 4/2/1 distinct adjacent cells.
    adjacent = []
    for di in [-1, 0]:
        for dj in [-1, 0]:
            for dk in [-1, 0]:
                cells = tuple(np.clip(m + d, 0, n - 1) for m, d, n in zip(mesh, [di, dj, dk], CELL_SHAPE))
                adjacent.append(np.ravel_multi_index(cells, CELL_SHAPE).ravel())
    adjacent = np.asarray(adjacent)
    np.save(output / 'display_adjacent_cells.npy', adjacent.astype(np.int32))
    return shape, coords, adjacent, lengths, bounds, [a.tolist() for a in indices]


def sample_fields(model_dir, adjacent, output):
    mean = np.load(model_dir / 'mean.npy', mmap_mode='r')
    modes = np.load(model_dir / 'modes.npy', mmap_mode='r')
    sampled_mean = np.mean(np.asarray(mean[adjacent], dtype=np.float64), axis=0).astype('<f4')
    sampled_modes = np.empty((len(modes), adjacent.shape[1]), dtype='<f4')
    for r, mode in enumerate(modes):
        sampled_modes[r] = np.mean(np.asarray(mode[adjacent], dtype=np.float64), axis=0)
    sampled_mean.tofile(output / 'mean.f32')
    sampled_modes.tofile(output / 'modes.f32')
    return sampled_mean, sampled_modes


def heldout_geometry(cache, cases):
    """Check the browser's L interpolation independently against held-out meshes."""
    train = sorted([r for r in cases if r['training']], key=lambda r: r['L_mm'])
    lengths = np.array([r['L_mm'] for r in train])
    rows = []
    def structured(row):
        p = cache / 'native' / row['case_id'].lower()
        return np.load(p / 'nodes.npy', mmap_mode='r')[np.load(p / 'structured_to_native_nodes.npy', mmap_mode='r')]
    for row in cases:
        if row['training']:
            continue
        hi = int(np.searchsorted(lengths, row['L_mm']))
        if hi < 1 or hi >= len(lengths):
            raise ValueError('Held-out geometry requires extrapolation')
        low = hi - 1
        t = (row['L_mm'] - lengths[low]) / (lengths[hi] - lengths[low])
        predicted = (1-t) * structured(train[low]) + t * structured(train[hi])
        error = predicted - structured(row)
        rows.append({'case_id': row['case_id'], 'node_count': len(error),
                     'bracketing_cases': [train[low]['case_id'], train[hi]['case_id']],
                     'coordinate_rmse_m': np.sqrt(np.mean(error**2, axis=0)).tolist(),
                     'coordinate_max_abs_error_m': np.max(abs(error), axis=0).tolist(),
                     'node_displacement_max_m': float(np.linalg.norm(error, axis=1).max()),
                     'method': 'linear observed-L interpolation on corresponding native 3D nodes'})
    return rows


def export(cache, model_dir, payload_path, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    registration = json.loads((cache / 'extraction_manifest.json').read_text())
    if not registration.get('topology_correspondence_verified'):
        raise ValueError('All training and validation meshes must be registered first')
    cases = json.loads((cache / 'cases.json').read_text())
    payload = json.loads(payload_path.read_text())
    model = pickle.loads((model_dir / 'coefficient_model.pkl').read_bytes())
    shape, coords, adjacent, lengths, bounds, indices = sample_geometry(cache, cases, output)
    mean, modes = sample_fields(model_dir, adjacent, output)
    schedule = payload['cases'][0]['samples']
    source_l = np.array(schedule['L_mm'])
    source_p = np.array(schedule['P0_Pa'])
    rank = model['details']['rank']
    factor = 1
    while True:
        t = np.linspace(0, len(source_l) - 1, (len(source_l) - 1) * factor + 1)
        path_l = np.interp(t, np.arange(len(source_l)), source_l)
        path_p = np.interp(path_l, source_l, source_p)
        query = np.column_stack([path_l, path_p])
        coefficients = predict_coefficients(model, query)
        mid_query = (query[1:] + query[:-1]) / 2
        direct = predict_coefficients(model, mid_query)
        difference = (coefficients[1:] + coefficients[:-1]) / 2 - direct
        weighted_lookup = float(np.linalg.norm(difference, axis=1).max())
        # The full volume modes are orthonormal in the fixed reference metric.
        if weighted_lookup < 1e-5 or factor >= 8:
            break
        factor *= 2
    if weighted_lookup >= 1e-4:
        raise ValueError('Operating-path lookup error exceeds tolerance')
    display_lookup = 0.
    value_min, value_max = np.inf, -np.inf
    for start in range(0, len(coefficients), 64):
        values = mean + coefficients[start:start + 64] @ modes
        value_min = min(value_min, float(values.min()))
        value_max = max(value_max, float(values.max()))
    for start in range(0, len(difference), 64):
        display_lookup = max(display_lookup, float(np.max(abs(difference[start:start + 64] @ modes))))
    raw_audit = json.loads((cache / 'extraction_audit.json').read_text())
    training_ids = {row['case_id'] for row in cases if row['training']}
    training_max = max(row['mach_max'] for row in raw_audit['cases'] if row['case_id'] in training_ids)
    color_max = float(np.ceil(max(training_max, value_max) * 10) / 10)
    arrays = {}
    for key, name, count in [('mean', 'mean.f32', len(mean)), ('modes', 'modes.f32', modes.size),
                              ('coordinates', 'coordinates.f32', coords.size)]:
        path = output / name
        arrays[key] = {'file': name, 'count': count, 'bytes': path.stat().st_size, 'sha256': digest(path)}
    manifest = {
        'schema': 'nozzle-volume-v1', 'grid': {'shape': shape, 'order': 'xyz-zfast'},
        'rank': rank, 'geometry_L_mm': lengths,
        'path': {'L_mm': path_l.tolist(), 'coefficients': coefficients.tolist()},
        'arrays': arrays, 'color_range': [0, color_max],
        'color_range_policy': '0 to rounded maximum of training full-volume Mach and predicted display operating path; separate from 2D plot limits',
        'training': {'case_count': 27, 'cell_count': int(np.prod(CELL_SHAPE)),
                     'quantity': 'Mach', 'domain': 'entire quarter CFD volume including downstream diffuser',
                     'reference_geometry': 'M17', 'POD_energy_fraction': model['details']['achieved_energy_fraction']},
        'geometry_method': 'Linear interpolation in observed L of native 3D material-grid nodes; no new FEA solution',
        'display_method': 'Decimated native material lattice, scalar nodes are arithmetic means of adjacent native cells',
        'cut_plane_definition': 'Material-grid cross section; x may vary slightly over a section, not an exact physical x plane',
        'symmetry': {'planes': ['y=0', 'z=0'], 'scalar_transform': 'Mach unchanged', 'quarter_training_only': True},
        'notes': ['전체 체적 3,533,741셀을 학습하고 화면 표시용 격자만 줄였습니다.',
                  '¼ CFD를 y=0, z=0에서 대칭 복원합니다.',
                  '정상 CFD 기반 마하수 예측이며 구동기·센서 시간 응답은 기존 가정 모델입니다.',
                  '중앙 Mach 제어 피드백은 기존 2D 단면 모델을 유지합니다.'],
        'validation_file': 'validation.json', 'source_model_sha256': digest(model_dir / 'coefficient_model.pkl'),
    }
    write_json(output / 'manifest.json', manifest)
    audit = {'display_shape': shape, 'display_node_count': len(mean), 'native_cell_count': int(np.prod(CELL_SHAPE)),
             'display_node_logical_indices': indices, 'geometry_bounds': bounds,
             'lookup_nodes': len(path_l), 'lookup_midpoint_count': len(difference),
             'lookup_max_reference_volume_rmse_mach': weighted_lookup,
             'lookup_max_display_node_error_mach': display_lookup,
             'display_mach_min_on_path': value_min, 'display_mach_max_on_path': value_max,
             'training_native_mach_max': training_max, 'display_color_range': [0, color_max],
             'negative_mach_clipped': False, 'color_limits_only_saturate_display': True,
             'asset_bytes': sum(a['bytes'] for a in arrays.values()),
             'full_native_mean_sha256': digest(model_dir / 'mean.npy'),
             'full_native_modes_sha256': digest(model_dir / 'modes.npy'),
             'heldout_geometry_interpolation': heldout_geometry(cache, cases),
             'elapsed_seconds': time.monotonic() - started}
    write_json(output / 'export_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=HERE / 'results/cache')
    parser.add_argument('--model', type=Path, default=HERE / 'results/model')
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.cache, args.model, args.payload, args.output)
