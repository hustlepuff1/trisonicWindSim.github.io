#!/usr/bin/env python3
"""Package the continuous L–P0 path, live control engine and UI in one offline HTML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_simulator(payload_path: Path, output_html: Path) -> Path:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    if not payload.get("cases") or not payload.get("field_roms"):
        raise ValueError("The simulator needs field_roms and a continuous operating path")
    if len(payload["cases"]) != 1:
        raise ValueError("The continuous simulator requires one coordinated L-P0 path")
    for key in ("longitudinal", "crossplane"):
        field = payload["field_roms"][key]
        if not field.get("mean") or not field.get("modes"):
            raise ValueError(f"Missing {key} POD mean or modes")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = (HERE / "simulator_template.html").read_text(encoding="utf-8")
    engine = (HERE / "control_engine.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    ui = (HERE / "simulator_ui.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    result = template.replace("__PAYLOAD__", encoded).replace("__ENGINE_CODE__", engine).replace("__UI_CODE__", ui)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(result, encoding="utf-8")
    return output_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload_path", type=Path)
    parser.add_argument("output_html", type=Path)
    args = parser.parse_args()
    print(build_simulator(args.payload_path, args.output_html))
