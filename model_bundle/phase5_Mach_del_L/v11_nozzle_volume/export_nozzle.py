#!/usr/bin/env python3
"""Export the freshly trained inlet-to-x=0 nozzle Mach ROM for the browser.

The model is fitted solely on the upstream domain. Its final layer merges the
upstream parts of native cells cut by x=0. Display coordinates preserve the
native 3D interior nodes and the exact physical terminal plane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'v10_volume_rom'))
from volume_rom import predict_coefficients


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def select_stations(values, count, extra=()):
    targets = np.r_[np.linspace(values[0], values[-1], count), extra]
    return np.unique(np.argmin(abs(values[:, None] - targets), axis=0))


def sample_geometry(cache, cases, output):
    xyz = np.load(cache / 'geometry/M17.npy', mmap_mode='r')
    node_shape = xyz.shape[:3]
    cell_shape = tuple(n - 1 for n in node_shape)
    x_axis = np.median(xyz[..., 0], axis=(1, 2))
    y_axis = np.median(xyz[..., 1], axis=(0, 2))
    eta_z = np.median(xyz[..., 2] / xyz[:, :, -1:, 2], axis=(0, 1))
    indices = [select_stations(x_axis, 201, [-.5, -.05, 0.]),
               select_stations(y_axis, 21, y_axis[-1] * np.array([.97, .99, .999])),
               select_stations(eta_z, 29, [.97, .99, .999])]
    mesh = np.meshgrid(*indices, indexing='ij')
    nodes = np.ravel_multi_index(tuple(mesh), node_shape).ravel()
    shape = [len(a) for a in indices]
    rows = sorted([r for r in cases if r['training']], key=lambda r: r['L_mm'])
    lengths = [r['L_mm'] for r in rows]
    if np.any(np.diff(lengths) <= 0):
        raise ValueError('Distinct ascending observed lengths are required')
    coords = np.empty((len(rows), len(nodes), 3), dtype='<f4')
    bounds = []
    for i, row in enumerate(rows):
        native = np.load(cache / 'geometry' / (row['case_id'] + '.npy'), mmap_mode='r')
        coords[i] = native.reshape(-1, 3)[nodes]
        bounds.append({'case_id': row['case_id'], 'min_m': coords[i].min(axis=0).tolist(),
                       'max_m': coords[i].max(axis=0).tolist()})
    if not np.isfinite(coords).all():
        raise ValueError('Nonfinite browser coordinates')
    shaped = coords.reshape(len(rows), *shape, 3)
    if np.any(coords[..., 0] > 0) or np.any(shaped[:, -1, ..., 0] != 0):
        raise ValueError('Every geometry must end at exact x=0, with no downstream node')
    coords.tofile(output / 'coordinates.f32')
    # At physical boundary nodes, the repeated clipped neighbour is intentional:
    # averaging the eight entries reduces to 4/2/1 distinct adjacent cells.
    adjacent = []
    for di in [-1, 0]:
        for dj in [-1, 0]:
            for dk in [-1, 0]:
                cells = tuple(np.clip(m + d, 0, n - 1) for m, d, n in zip(mesh, [di, dj, dk], cell_shape))
                adjacent.append(np.ravel_multi_index(cells, cell_shape).ravel())
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
        return np.load(cache / 'geometry' / (row['case_id'] + '.npy'), mmap_mode='r').reshape(-1, 3)
    for row in cases:
        if row['training']:
            continue
        hi = int(np.searchsorted(lengths, row['L_mm']))
        if hi < 1 or hi >= len(lengths):
            raise ValueError('Held-out geometry requires extrapolation')
        low = hi - 1
        t = (row['L_mm'] - lengths[low]) / (lengths[hi] - lengths[low])
        predicted = (1-t) * structured(train[low]) + t * structured(train[hi])
        actual = structured(row)
        error = predicted - actual
        node_shape = np.load(cache / 'geometry' / (row['case_id'] + '.npy'), mmap_mode='r').shape
        predicted_wall = predicted.reshape(node_shape)[:, :, -1, :]
        actual_wall = actual.reshape(node_shape)[:, :, -1, :]
        height_errors = []
        unsupported = 0
        for span in range(node_shape[1]):
            px, pz = predicted_wall[:, span, 0], predicted_wall[:, span, 2]
            ax, az = actual_wall[:, span, 0], actual_wall[:, span, 2]
            keep = (ax >= px[0]) & (ax <= px[-1])
            unsupported += int((~keep).sum())
            height_errors.extend(np.interp(ax[keep], px, pz) - az[keep])
        height_errors = np.asarray(height_errors)
        rows.append({'case_id': row['case_id'], 'node_count': len(error),
                     'bracketing_cases': [train[low]['case_id'], train[hi]['case_id']],
                     'coordinate_rmse_m': np.sqrt(np.mean(error**2, axis=0)).tolist(),
                     'coordinate_max_abs_error_m': np.max(abs(error), axis=0).tolist(),
                     'node_displacement_max_m': float(np.linalg.norm(error, axis=1).max()),
                     'upper_wall_matched_physical_x_max_abs_height_error_m': float(abs(height_errors).max()),
                     'upper_wall_matched_physical_x_rmse_m': float(np.sqrt(np.mean(height_errors**2))),
                     'upper_wall_compared_node_count': len(height_errors),
                     'unsupported_wall_endpoint_nodes': unsupported,
                     'method': 'linear observed-L interpolation on registered nozzle nodes, inlet to exact x=0 only'})
    return rows


def export(cache, model_dir, payload_path, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    registration = json.loads((cache / 'extraction_manifest.json').read_text())
    if not registration.get('complete') or not registration.get('topology_correspondence_verified'):
        raise ValueError('All training and validation meshes must be registered first')
    cases = json.loads((cache / 'cases.json').read_text())
    payload = json.loads(payload_path.read_text())
    model = pickle.loads((model_dir / 'coefficient_model.pkl').read_bytes())
    if not model['details'].get('fresh_model_fit') or model['details'].get('cropped_from_full_domain_model') is not False:
        raise ValueError('Only the freshly fitted nozzle-domain model may be exported')
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
    fields = np.load(cache / 'registered_mach.npy', mmap_mode='r')
    native_shape = np.load(cache / 'geometry/M17.npy', mmap_mode='r').shape[:3]
    cell_shape = [n - 1 for n in native_shape]
    terminal_count = int(np.prod(cell_shape[1:]))
    training_max = max(float(fields[i].max()) for i, row in enumerate(cases) if row['training'])
    color_max = max(payload['cases'][0]['color_max'], float(np.ceil(max(training_max, value_max) * 10) / 10))
    arrays = {}
    for key, name, count in [('mean', 'mean.f32', len(mean)), ('modes', 'modes.f32', modes.size),
                              ('coordinates', 'coordinates.f32', coords.size)]:
        path = output / name
        arrays[key] = {'file': name, 'count': count, 'bytes': path.stat().st_size, 'sha256': digest(path)}
    manifest = {
        'schema': 'nozzle-volume-v1', 'grid': {'shape': shape, 'order': 'xyz-zfast'},
        'domain': {'id': 'nozzle_x_le_0', 'x_max_m': 0, 'endpoint': 'physical_x_plane',
                   'description': 'inlet through x=0 test section; diffuser excluded from both training and display'},
        'rank': rank, 'geometry_L_mm': lengths,
        'path': {'L_mm': path_l.tolist(), 'coefficients': coefficients.tolist()},
        'arrays': arrays, 'color_range': [0, color_max],
        'color_range_policy': '0 to maximum of original 2D color limit and rounded nozzle-training/display-path Mach maximum',
        'training': {'case_count': 27, 'cell_count': fields.shape[1],
                     'quantity': 'Mach', 'domain': 'quarter nozzle volume from inlet through physical x=0',
                     'fresh_fit': True, 'previous_full_domain_model_used': False,
                     'cell_shape': cell_shape,
                     'unchanged_native_cell_count': fields.shape[1] - terminal_count,
                     'terminal_aggregated_cell_count': terminal_count,
                     'reference_geometry': 'M17', 'POD_energy_fraction': model['details']['achieved_energy_fraction']},
        'geometry_method': 'Linear interpolation in observed L of registered 3D nozzle nodes; terminal slice exactly x=0; no new FEA solution',
        'display_method': 'Decimated registered nozzle grid; scalar nodes average adjacent registered cells',
        'cut_plane_definition': 'Last slice is exact physical x=0; interior material-grid sections may have a small x range',
        'symmetry': {'planes': ['y=0', 'z=0'], 'scalar_transform': 'Mach unchanged', 'quarter_training_only': True},
        'notes': [f'입구부터 x=0까지 {fields.shape[1]:,}개 노즐 체적 셀로 새 POD-GPR을 학습했습니다.',
                  '¼ CFD를 y=0, z=0에서 대칭 복원합니다.',
                  '정상 CFD 기반 마하수 예측이며 구동기·센서 시간 응답은 기존 가정 모델입니다.',
                  '중앙 Mach 제어 피드백은 기존 2D 단면 모델을 유지합니다.'],
        'validation_file': 'validation.json', 'source_model_sha256': digest(model_dir / 'coefficient_model.pkl'),
    }
    write_json(output / 'manifest.json', manifest)
    audit = {'display_shape': shape, 'display_node_count': len(mean), 'native_cell_count': fields.shape[1],
             'unchanged_native_cell_count': fields.shape[1] - terminal_count,
             'terminal_aggregated_cell_count': terminal_count,
             'domain': manifest['domain'], 'all_geometry_nodes_x_le_0': True,
             'all_terminal_geometry_nodes_exact_x_zero': True,
             'domain_manifest_sha256': digest(cache / 'domain_manifest.json'),
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
    write_json(output / 'geometry_validation.json', {
        'domain': manifest['domain'], 'cases': audit['heldout_geometry_interpolation'],
        'scope': 'Nozzle-only corresponding-node displacement and upper-wall height at matched physical x; no wall extrapolation'})
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
