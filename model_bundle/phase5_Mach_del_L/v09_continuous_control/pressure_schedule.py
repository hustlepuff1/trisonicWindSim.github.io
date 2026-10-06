#!/usr/bin/env python3
"""Analytical reference P0 schedule for the nominal 80 kg/s design family.

This is an ideal isentropic, calorically perfect-gas reference equation at the
nominal full test-section area. It neither measures nor regulates CFD mass
flow. P0 is absolute stagnation pressure; outlet pressure is a separate BC.
The supported reference-Mach interval is explicitly restricted to [1.5,3.5].
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Portable inventory includes indexed external raw files; none are reread.
import sys
sys.path.insert(0, str(ROOT))
from portable_paths import external_raw_names
GAMMA = 1.4
R_AIR_J_KG_K = 287.0
T0_K = 300.0
TEST_SECTION_AREA_M2 = 0.25
REFERENCE_MASS_FLOW_KG_S = 80.0
SYMMETRY_FACTOR = 4
OUTLET_STATIC_PRESSURE_PA = 10132.5
FACILITY_REFERENCE_LIMIT_PA = 9.8 * 101325.0
MACH_MIN = 1.5
MACH_MAX = 3.5
# Existing Phase1/training pressure tables are rounded to 0.001 Pa. This
# allowance only restores endpoint values within that known rounding interval.
TABLE_ROUNDING_TOLERANCE_PA = 0.000501
NASA_EQUATION_URL = "https://www.grc.nasa.gov/WWW/K-12/airplane/mflchk.html"


def _numbers(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all() or array.size == 0:
        raise ValueError(f"{name} must contain finite real numbers, not bool/string/complex")
    return np.asarray(array, dtype=float)


def _return_shape(value):
    return float(value) if np.ndim(value) == 0 else value


def _mach(value):
    mach = _numbers(value, "reference Mach")
    if np.any(mach < MACH_MIN) or np.any(mach > MACH_MAX):
        raise ValueError(f"Reference Mach must lie within [{MACH_MIN}, {MACH_MAX}]")
    return mach


def p0_reference(mach):
    """Return absolute reference stagnation pressure [Pa], scalar or ndarray.

    P0 = mdot*sqrt(R*T0)/(A*M*sqrt(gamma)) *
         (1 + (gamma-1)*M**2/2)**((gamma+1)/(2*(gamma-1))).
    M is a design/path parameter, not an assertion of achieved core Mach.
    """
    mach = _mach(mach)
    exponent = (GAMMA + 1) / (2 * (GAMMA - 1))
    pressure = (REFERENCE_MASS_FLOW_KG_S * np.sqrt(R_AIR_J_KG_K * T0_K)
                / (TEST_SECTION_AREA_M2 * mach * np.sqrt(GAMMA))
                * (1 + (GAMMA - 1) * mach ** 2 / 2) ** exponent)
    return _return_shape(pressure)


def dp0_dmach(mach):
    """Analytical dP0/dM [Pa/Mach], strictly positive on this supersonic branch."""
    mach = _mach(mach)
    derivative = (p0_reference(mach) * (mach ** 2 - 1)
                  / (mach * (1 + (GAMMA - 1) * mach ** 2 / 2)))
    return _return_shape(derivative)


def mach_from_p0_reference(pressure_pa):
    """Invert the unique reference branch, rejecting extrapolation.

    The 0.001-Pa historical table rounding is allowed at the two endpoints;
    those pressures return exactly the corresponding endpoint Mach. Interior
    values use 60 bounded bisection iterations, with no fit or data dependence.
    """
    pressure = _numbers(pressure_pa, "absolute reference pressure [Pa]")
    low_pressure, high_pressure = p0_reference(MACH_MIN), p0_reference(MACH_MAX)
    if (np.any(pressure < low_pressure - TABLE_ROUNDING_TOLERANCE_PA)
            or np.any(pressure > high_pressure + TABLE_ROUNDING_TOLERANCE_PA)):
        raise ValueError("Pressure lies outside the supported reference-Mach branch")
    target = np.clip(pressure, low_pressure, high_pressure)
    low, high = np.full_like(target, MACH_MIN), np.full_like(target, MACH_MAX)
    for _ in range(60):
        middle = (low + high) / 2
        lower = p0_reference(middle) < target
        low = np.where(lower, middle, low)
        high = np.where(lower, high, middle)
    result = (low + high) / 2
    result = np.where(target == low_pressure, MACH_MIN, result)
    result = np.where(target == high_pressure, MACH_MAX, result)
    return _return_shape(result)


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_provenance(output=HERE / "results/pressure_schedule_provenance.json"):
    """Write a reproducible local-source audit, without loading raw CFD volumes."""
    phase1 = ROOT / "phase1_quasi1D/phase1_2_envelope_get_target_p0.py"
    table = ROOT / "phase1_quasi1D/results/phase1_2_envelope_results/target_p0_for_cfd.csv"
    v02 = ROOT / "phase5_Mach_del_L/v02_actual_geometry/generate_actual_geometry_map_v02.py"
    v05 = ROOT / "phase5_Mach_del_L/v05_surrogate_training/results/training_dataset.csv"
    sources = [phase1, table, v02, v05, ROOT / "phase4_CFD/README.md",
               ROOT / "phase5_Mach_del_L/v03_cfd_polynomial/core_area.py", Path(__file__)]
    with table.open(encoding="utf-8-sig") as stream:
        references = [r for r in csv.DictReader(stream) if MACH_MIN <= float(r["Mach"]) <= MACH_MAX]
    discrepancy = max(abs(p0_reference(float(r["Mach"])) - float(r["Target_P0_Pa"])) for r in references)
    if discrepancy > TABLE_ROUNDING_TOLERANCE_PA:
        raise ValueError("Reference pressure table differs from analytical source equation")
    with v05.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    latest = [r for r in rows if "_new" in r["case_id"]]
    raw_dir = ROOT / "phase4_CFD/results(Structural)/CFD_results"
    indexed = external_raw_names("phase4_CFD/results(Structural)/CFD_results")
    raw_names = sorted({p.name for p in raw_dir.glob("*.dat") if "results" in p.stem} |
                       {name for name in indexed if name.endswith(".dat") and "results" in Path(name).stem})
    import re
    existing_175 = [n for n in raw_names if re.search(r"(?i)M(?:1[._]?75|175)(?:_|[.])", n)]
    value = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_equation": "mdot*sqrt(R*T0)/(A*M*sqrt(gamma))*(1+(gamma-1)*M^2/2)^((gamma+1)/(2*(gamma-1)))",
        "pressure_unit_and_kind": "Pa, absolute total (stagnation) pressure",
        "design_reference_mach_range": [MACH_MIN, MACH_MAX],
        "constants": {"gamma": GAMMA, "R_J_kg_K": R_AIR_J_KG_K, "T0_K": T0_K,
            "nominal_full_test_section_area_m2": TEST_SECTION_AREA_M2,
            "nominal_quarter_test_section_area_m2": TEST_SECTION_AREA_M2 / SYMMETRY_FACTOR,
            "design_reference_full_mass_flow_kg_s": REFERENCE_MASS_FLOW_KG_S,
            "design_reference_quarter_mass_flow_kg_s": REFERENCE_MASS_FLOW_KG_S / SYMMETRY_FACTOR,
            "outlet_static_pressure_Pa": OUTLET_STATIC_PRESSURE_PA,
            "historical_facility_pressure_limit_Pa": FACILITY_REFERENCE_LIMIT_PA},
        "source_files_sha256": {str(p.relative_to(ROOT)): _digest(p) for p in sources},
        "NASA_primary_equation_reference": {"url": NASA_EQUATION_URL, "equation": 10,
            "scope": "Ideal compressible isentropic mass-flow equation; local repository supplies the numerical design constants"},
        "phase1_csv_max_rounding_difference_Pa": discrepancy,
        "pressure_reference_at_M1_75_Pa": p0_reference(1.75),
        "pressure_reference_at_M1_75_atm": p0_reference(1.75) / 101325,
        "latest_CFD_actual_full_equivalent_mass_flow": [{"case_id": r["case_id"],
            "full_kg_s": float(r["full_section_mass_flow_kg_s"]),
            "quarter_kg_s": float(r["full_section_mass_flow_kg_s"]) / SYMMETRY_FACTOR,
            "difference_from_80kg_s_percent": 100 * (float(r["full_section_mass_flow_kg_s"]) / REFERENCE_MASS_FLOW_KG_S - 1)} for r in latest],
        "actual_mass_flow_definition": "4*integral_over_full_source_quarter_x0(rho*u*dA), not the adaptive core-only flux",
        "actual_mass_flow_fixed_at_80kg_s": False,
        "latest_solver_decks_independently_checked": False,
        "historical_BC_evidence": "Phase4 README documents prior BC7 P0/T0 and fixed outlet/static pressure audit; current latest case decks were not re-audited",
        "unseen_case_audit": {"searched_directory": str(raw_dir), "raw_volume_filenames": raw_names,
            "inventory_scope": "Bundled files plus external_raw_index snapshot; not a fresh search of the original source tree",
            "existing_M1_75_named_volume_files": existing_175,
            "existing_M17": {key: rows[[r["case_id"] for r in rows].index("M17")][key]
                for key in ["L_mm", "P0_Pa", "mean_mach"]},
            "novelty_definition": "Freeze a new L/P0 input pair before CFD; an existing response near1.75 is not a new validation case",
            "raw_volumes_hashed_in_this_audit": False},
        "coordinated_pressure_tracking_scope": "P0_ref(M_path(L_actual)) assumes ideal instantaneous pressure tracking of current actuator path; no pressure-regulator dynamics, no transient CFD and no actual mass-flow regulation",
        "support_requirement": "Check the final [L_mm,P0_Pa] path against ROM training support; analytical pressure alone does not establish support"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/pressure_schedule_provenance.json")
    args = parser.parse_args()
    result = write_provenance(args.output)
    print(f"P0_reference(1.75) = {result['pressure_reference_at_M1_75_Pa']:.9f} Pa; wrote {args.output}")
