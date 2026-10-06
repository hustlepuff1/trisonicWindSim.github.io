#!/usr/bin/env python3
"""Rebuild the static simulator using only the files in this repository.

The default packages the saved payload. --from-models regenerates that payload
with the two unchanged trained ROMs before packaging. Outputs go to build/;
the published index.html and original frozen records are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
V09 = ROOT / "model_bundle/phase5_Mach_del_L/v09_continuous_control"


def preserved_predictions(output: Path) -> None:
    source = ROOT / "frozen_validation/M1p75"
    manifest = json.loads((source / "manifest.json").read_text())
    for name, expected in manifest["outputs_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen prediction changed: {name}")
    destination = output / "frozen_validation/M1p75"
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if not path.is_file():
            continue
        dest = destination / path.name
        if dest.exists() and dest.read_bytes() != path.read_bytes():
            raise ValueError(f"Refusing to replace a different frozen record: {dest}")
        if not dest.exists():
            shutil.copy2(path, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-models", action="store_true",
                        help="Regenerate the continuous payload from the two frozen PKL models.")
    parser.add_argument("--output-dir", default="build",
                        help="Output subdirectory of this repository (default: build).")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if output == ROOT or not output.is_relative_to(ROOT):
        parser.error("--output-dir must be a subdirectory of this repository")
    for protected in (ROOT / "model_bundle", ROOT / "frozen_validation"):
        if output.is_relative_to(protected):
            parser.error("--output-dir cannot overwrite model or frozen source files")
    output.mkdir(parents=True, exist_ok=True)
    preserved_predictions(output)
    # Keep local imports and bytecode inside the copied bundle. Scientific
    # packages are imported only when rebuilding from the trained models.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(V09))
    from build_simulator import build_simulator

    if args.from_models:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from build_continuous import build
        build(output=output)
        payload = output / "simulator_payload.json"
    else:
        payload = V09 / "results/simulator_payload.json"
    html = build_simulator(payload, output / "index.html")
    print(f"HTML: {html}")
    print(f"SHA256: {hashlib.sha256(html.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
