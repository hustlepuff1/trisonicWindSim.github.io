#!/usr/bin/env python3
"""Audit the reduced three-dimensional Tecplot CFD volume archives.

The retained ASCII files contain nodal coordinates, cell-centred velocity,
density, static pressure, Mach number, and complete FEPolyhedron connectivity.
This script records a checksummed manifest, verifies every declared data and
connectivity block, and performs perfect-gas consistency diagnostics without
loading the full solver solution that was used to export the files.

The resulting global ranges are file-integrity diagnostics only.  They are not
flow-quality metrics because the MOC-short domain differs from the MOC-long
and Structural domains and because the full domains include regions outside
the test section.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "phase4_CFD"
OUTPUT_DIR = PHASE4 / "postprocessing_3d"
MANIFEST_PATH = OUTPUT_DIR / "volume_dataset_manifest.csv"

TARGETS = (
    (1.5, "1_5"),
    (2.0, "2_0"),
    (2.5, "2_5"),
    (3.0, "3_0"),
    (3.5, "3_5"),
)

CASE_FILES = {
    "MOC-long": (
        PHASE4 / "results(MOC_long)",
        "results_MOC_long_{suffix}.dat",
    ),
    "MOC-short": (
        PHASE4 / "results(MOC_short)",
        "results_MOC_short_{suffix}.dat",
    ),
    "Structural": (
        PHASE4 / "results(Structural)",
        "results_mach{suffix}.dat",
    ),
}

EXPECTED_VARIABLES = ("x", "y", "z", "u", "v", "w", "rho", "p", "Ma")
NODAL_VARIABLES = ("x", "y", "z")
CELL_VARIABLES = ("u", "v", "w", "rho", "p", "Ma")
GAMMA = 1.4
GAS_CONSTANT_J_PER_KG_K = 287.05
HEADER_INT_RE = {
    name: re.compile(rf"\b{name}\s*=\s*(\d+)", re.IGNORECASE)
    for name in ("Nodes", "Faces", "Elements", "TotalNumFaceNodes")
}


@dataclass(frozen=True)
class VolumeHeader:
    text: str
    variables: tuple[str, ...]
    nodes: int
    faces: int
    elements: int
    total_face_nodes: int
    zone_type: str
    data_packing: str
    variable_location: str


def _header_int(text: str, name: str) -> int:
    match = HEADER_INT_RE[name].search(text)
    if not match:
        raise ValueError(f"Missing {name}=... in Tecplot header")
    return int(match.group(1))


def read_header(stream) -> VolumeHeader:
    lines: list[str] = []
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("Unexpected EOF before Tecplot DT declaration")
        lines.append(line)
        if line.lstrip().upper().startswith("DT="):
            break

    text = "".join(lines)
    variables = tuple(re.findall(r'^\s*"([^"]+)"\s*$', text, re.MULTILINE))
    first_variable = re.search(r'^\s*VARIABLES\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if first_variable:
        variables = (first_variable.group(1),) + variables

    zone_match = re.search(r"\bZONETYPE\s*=\s*(\w+)", text, re.IGNORECASE)
    packing_match = re.search(r"\bDATAPACKING\s*=\s*(\w+)", text, re.IGNORECASE)
    location_match = re.search(r"^\s*VARLOCATION\s*=\s*(.+)$", text, re.MULTILINE)
    if not zone_match or not packing_match or not location_match:
        raise ValueError("Incomplete Tecplot zone declaration")

    return VolumeHeader(
        text=text,
        variables=variables,
        nodes=_header_int(text, "Nodes"),
        faces=_header_int(text, "Faces"),
        elements=_header_int(text, "Elements"),
        total_face_nodes=_header_int(text, "TotalNumFaceNodes"),
        zone_type=zone_match.group(1),
        data_packing=packing_match.group(1),
        variable_location=location_match.group(1).strip(),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_block(stream, count: int, label: str, path: Path) -> np.ndarray:
    values = np.fromfile(stream, dtype=np.float64, count=count, sep=" ")
    if values.size != count:
        raise ValueError(
            f"{path}: {label} expected {count:,} values, found {values.size:,}"
        )
    return values


def _skip_comment_lines(stream) -> None:
    """Advance past Tecplot's optional ``# block label`` lines."""
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line:
            return
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stream.seek(position)
        return


def verify_payload(path: Path, verify_connectivity: bool) -> dict[str, float | int | str]:
    stats: dict[str, float | int | str] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        header = read_header(stream)
        if header.variables != EXPECTED_VARIABLES:
            raise ValueError(
                f"{path}: variables {header.variables} != {EXPECTED_VARIABLES}"
            )
        if header.zone_type.upper() != "FEPOLYHEDRON":
            raise ValueError(f"{path}: unexpected zone type {header.zone_type}")
        if header.data_packing.upper() != "BLOCK":
            raise ValueError(f"{path}: unexpected packing {header.data_packing}")

        for variable in NODAL_VARIABLES:
            values = _read_block(stream, header.nodes, variable, path)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite {variable} values")
            stats[f"{variable}_min"] = float(np.min(values))
            stats[f"{variable}_max"] = float(np.max(values))

        cell_fields: dict[str, np.ndarray] = {}
        for variable in CELL_VARIABLES:
            values = _read_block(stream, header.elements, variable, path)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite {variable} values")
            stats[f"{variable}_min"] = float(np.min(values))
            stats[f"{variable}_max"] = float(np.max(values))
            if variable in {"rho", "p", "Ma"}:
                percentiles = np.percentile(values, (1.0, 50.0, 99.0))
                stats[f"{variable}_p01"] = float(percentiles[0])
                stats[f"{variable}_p50"] = float(percentiles[1])
                stats[f"{variable}_p99"] = float(percentiles[2])
            cell_fields[variable] = values

        if np.any(cell_fields["rho"] <= 0.0):
            raise ValueError(f"{path}: density must be positive")
        if np.any(cell_fields["p"] <= 0.0):
            raise ValueError(f"{path}: static pressure must be positive")

        transverse = np.hypot(cell_fields["v"], cell_fields["w"])
        angle = np.degrees(
            np.arctan2(transverse, np.abs(cell_fields["u"]))
        )
        stats["flow_angle_p50_deg"] = float(np.percentile(angle, 50.0))
        stats["flow_angle_p99_deg"] = float(np.percentile(angle, 99.0))

        static_temperature = cell_fields["p"] / (
            cell_fields["rho"] * GAS_CONSTANT_J_PER_KG_K
        )
        total_pressure = cell_fields["p"] * (
            1.0 + 0.5 * (GAMMA - 1.0) * cell_fields["Ma"] ** 2
        ) ** (GAMMA / (GAMMA - 1.0))
        mass_flux_x = cell_fields["rho"] * cell_fields["u"]
        speed = np.sqrt(
            cell_fields["u"] ** 2
            + cell_fields["v"] ** 2
            + cell_fields["w"] ** 2
        )
        sound_speed = np.sqrt(GAMMA * cell_fields["p"] / cell_fields["rho"])
        mach_from_primitive = speed / sound_speed
        mach_difference = mach_from_primitive - cell_fields["Ma"]

        for name, values in (
            ("T_static_K", static_temperature),
            ("P0_Pa", total_pressure),
            ("rho_u_kg_m2_s", mass_flux_x),
        ):
            percentiles = np.percentile(values, (1.0, 50.0, 99.0))
            stats[f"{name}_min"] = float(np.min(values))
            stats[f"{name}_p01"] = float(percentiles[0])
            stats[f"{name}_p50"] = float(percentiles[1])
            stats[f"{name}_p99"] = float(percentiles[2])
            stats[f"{name}_max"] = float(np.max(values))

        absolute_mach_difference = np.abs(mach_difference)
        stats["Ma_primitive_consistency_RMS"] = float(
            np.sqrt(np.mean(mach_difference**2))
        )
        stats["Ma_primitive_consistency_p99_abs"] = float(
            np.percentile(absolute_mach_difference, 99.0)
        )
        stats["Ma_primitive_consistency_max_abs"] = float(
            np.max(absolute_mach_difference)
        )

        if verify_connectivity:
            _skip_comment_lines(stream)
            counts = np.fromfile(
                stream, dtype=np.int64, count=header.faces, sep=" "
            )
            if counts.size != header.faces:
                raise ValueError(f"{path}: incomplete face-node-count block")
            if int(np.sum(counts)) != header.total_face_nodes:
                raise ValueError(
                    f"{path}: face-node sum {int(np.sum(counts)):,} != "
                    f"declared {header.total_face_nodes:,}"
                )
            stats["face_nodes_per_face_min"] = int(np.min(counts))
            stats["face_nodes_per_face_max"] = int(np.max(counts))
            del counts

            _skip_comment_lines(stream)
            node_ids = np.fromfile(
                stream,
                dtype=np.int64,
                count=header.total_face_nodes,
                sep=" ",
            )
            if node_ids.size != header.total_face_nodes:
                raise ValueError(f"{path}: incomplete face-node connectivity")
            if int(np.min(node_ids)) < 1 or int(np.max(node_ids)) > header.nodes:
                raise ValueError(f"{path}: face node ID outside declared node range")
            del node_ids

            for side in ("left", "right"):
                _skip_comment_lines(stream)
                element_ids = np.fromfile(
                    stream, dtype=np.int64, count=header.faces, sep=" "
                )
                if element_ids.size != header.faces:
                    raise ValueError(f"{path}: incomplete {side}-element block")
                if int(np.min(element_ids)) < 0 or int(np.max(element_ids)) > header.elements:
                    raise ValueError(
                        f"{path}: {side}-element ID outside declared element range"
                    )
                del element_ids

            _skip_comment_lines(stream)
            trailing = np.fromfile(stream, dtype=np.float64, count=1, sep=" ")
            if trailing.size:
                raise ValueError(f"{path}: unexpected numeric payload after connectivity")

    return stats


def cases() -> list[tuple[str, float, Path]]:
    selected: list[tuple[str, float, Path]] = []
    for configuration, (directory, pattern) in CASE_FILES.items():
        for target, suffix in TARGETS:
            selected.append((configuration, target, directory / pattern.format(suffix=suffix)))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Record headers and hashes without reading the field/connectivity blocks.",
    )
    parser.add_argument(
        "--skip-connectivity",
        action="store_true",
        help="Verify all variable blocks but not the FEPolyhedron connectivity arrays.",
    )
    parser.add_argument(
        "--configuration",
        choices=tuple(CASE_FILES),
        help="Audit only one configuration (use with --merge-existing to refresh its manifest rows).",
    )
    parser.add_argument(
        "--target",
        type=float,
        choices=tuple(target for target, _ in TARGETS),
        help="Audit only one target Mach number.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge selected audit rows into the existing complete manifest.",
    )
    args = parser.parse_args()

    fields = (
        "configuration",
        "target_mach",
        "relative_path",
        "size_bytes",
        "sha256",
        "variables",
        "nodes",
        "faces",
        "elements",
        "total_face_nodes",
        "zone_type",
        "data_packing",
        "variable_location",
        "payload_verified",
        "connectivity_verified",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
        "u_min",
        "u_max",
        "v_min",
        "v_max",
        "w_min",
        "w_max",
        "rho_min",
        "rho_p01",
        "rho_p50",
        "rho_p99",
        "rho_max",
        "p_min",
        "p_p01",
        "p_p50",
        "p_p99",
        "p_max",
        "Ma_min",
        "Ma_p01",
        "Ma_p50",
        "Ma_p99",
        "Ma_max",
        "T_static_K_min",
        "T_static_K_p01",
        "T_static_K_p50",
        "T_static_K_p99",
        "T_static_K_max",
        "P0_Pa_min",
        "P0_Pa_p01",
        "P0_Pa_p50",
        "P0_Pa_p99",
        "P0_Pa_max",
        "rho_u_kg_m2_s_min",
        "rho_u_kg_m2_s_p01",
        "rho_u_kg_m2_s_p50",
        "rho_u_kg_m2_s_p99",
        "rho_u_kg_m2_s_max",
        "Ma_primitive_consistency_RMS",
        "Ma_primitive_consistency_p99_abs",
        "Ma_primitive_consistency_max_abs",
        "flow_angle_p50_deg",
        "flow_angle_p99_deg",
        "face_nodes_per_face_min",
        "face_nodes_per_face_max",
    )
    rows: list[dict[str, object]] = []
    all_cases = cases()
    selected = [
        item
        for item in all_cases
        if (args.configuration is None or item[0] == args.configuration)
        and (args.target is None or item[1] == args.target)
    ]
    if not selected:
        raise ValueError("No CFD volumes match the requested filters")
    for index, (configuration, target, path) in enumerate(selected, start=1):
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig") as stream:
            header = read_header(stream)
        if header.variables != EXPECTED_VARIABLES:
            raise ValueError(f"{path}: unexpected variables {header.variables}")

        print(f"[{index:02d}/{len(selected):02d}] {configuration} Mach {target:.1f}: {path.name}")
        row: dict[str, object] = {
            "configuration": configuration,
            "target_mach": f"{target:.1f}",
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "variables": ";".join(header.variables),
            "nodes": header.nodes,
            "faces": header.faces,
            "elements": header.elements,
            "total_face_nodes": header.total_face_nodes,
            "zone_type": header.zone_type,
            "data_packing": header.data_packing,
            "variable_location": header.variable_location,
            "payload_verified": "no" if args.quick else "yes",
            "connectivity_verified": (
                "no" if args.quick or args.skip_connectivity else "yes"
            ),
        }
        if not args.quick:
            row.update(verify_payload(path, not args.skip_connectivity))
        rows.append(row)

    if args.merge_existing:
        if not MANIFEST_PATH.is_file():
            raise FileNotFoundError(
                f"Cannot merge without an existing manifest: {MANIFEST_PATH}"
            )
        with MANIFEST_PATH.open(encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
        indexed = {
            (str(row["configuration"]), float(row["target_mach"])): row
            for row in existing
        }
        for row in rows:
            indexed[(str(row["configuration"]), float(row["target_mach"]))] = row
        rows = [indexed[(configuration, target)] for configuration, target, _ in all_cases]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    total_bytes = sum(int(row["size_bytes"]) for row in rows)
    if not math.isfinite(float(total_bytes)):
        raise AssertionError("Non-finite total size")
    print(f"manifest: {MANIFEST_PATH}")
    print(f"files: {len(rows)}, total: {total_bytes / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
