"""Resolve preserved provenance paths inside this portable framework only.

Original absolute paths remain evidence; they are never an implicit fallback
for runtime data access. Full raw volumes are optional, explicitly restored
at their relative phase4 paths for Tier-2 re-extraction.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
ROOT_NAMES = {"phase1_quasi1D", "phase2_MOC", "phase3_FEA", "phase4_CFD", "phase5_Mach_del_L", "3DCAD_REF", "tools"}


def local_path(recorded):
    candidate = Path(recorded)
    if candidate.is_absolute():
        if candidate.is_relative_to(ROOT):
            relative = candidate.relative_to(ROOT)
        else:
            indices = [i for i, part in enumerate(candidate.parts) if part in ROOT_NAMES]
            if not indices:
                raise ValueError(f"Not a recognized framework provenance path: {recorded}")
            relative = Path(*candidate.parts[indices[0]:])
    else:
        relative = candidate
    result = (ROOT / relative).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("A provenance path must not escape the framework")
    return result


def raw_source_status(source, *, verify=False):
    """Never probe the original source location; verify local raw only on request."""
    result = {"raw_volume_hash_rechecked": False, "raw_stat_rechecked": False,
              "verification_mode": "verified_plane_cache_with_preserved_historical_raw_hash"}
    if verify:
        raw = local_path(source["path"])
        if not raw.is_file():
            raise FileNotFoundError(f"Tier-2 raw file is not bundled: {raw}. Restore it from external_raw_index.json first.")
        stat = raw.stat()
        with raw.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if stat.st_size != source["bytes"] or sha != source["sha256"]:
            raise ValueError("Restored raw volume does not match its preserved source hash")
        result.update(raw_volume_hash_rechecked=True, raw_stat_rechecked=True,
                      verification_mode="restored_local_raw_size_and_full_sha256_verified")
    return result


def external_raw_names(relative_directory):
    index = json.loads((ROOT / "external_raw_index.json").read_text())
    directory = Path(relative_directory)
    return sorted(Path(row["relative_path"]).name for row in index["files"]
                  if Path(row["relative_path"]).parent == directory)
