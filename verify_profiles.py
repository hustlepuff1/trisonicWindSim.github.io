#!/usr/bin/env python3
"""Validate live physical-axis profiles against the shipped 3D ROM binaries.

Uses the isolated Firefox Marionette helper from verify_browser.py. Start a
local HTTP server and Firefox with marionette.port=2841 before running. The
optional baseline URL is a saved, unmodified HEAD page for exact 2D comparison.
Generated records remain under the ignored build_profiles_review directory.
"""
import argparse
import base64
import csv
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from verify_browser import Browser

ROOT = Path(__file__).resolve().parent
VALUE_KEYS = ("coordinate_m", "x_m", "y_m", "z_m", "mach")
SNAPSHOT_JS = """const d=window.nozzleSimulatorDebug;
return {control:d.getState(),volume:d.getVolumeState(),profiles:d.getVolumeProfiles()};"""
SAMPLES_JS = """const d=window.nozzleSimulatorDebug,out=[];
for(const m of [1.5,1.75,2,2.5,3,3.5]){
  d.setCommand(m);d.step(2400);const f=d.getFields();
  out.push({state:d.getState(),longitudinal:Array.from(f.longitudinal),crossplane:Array.from(f.crossplane)});
}return out;"""


class ReferenceProfiles:
    """Independent NumPy reconstruction and extraction in physical coordinates."""
    def __init__(self, directory):
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.shape = tuple(self.manifest["grid"]["shape"])
        self.rank = self.manifest["rank"]
        self.mean = np.fromfile(directory / "mean.f32", dtype="<f4")
        self.modes = np.fromfile(directory / "modes.f32", dtype="<f4").reshape(self.rank, -1)
        self.geometry = np.fromfile(directory / "coordinates.f32", dtype="<f4").reshape(
            len(self.manifest["geometry_L_mm"]), *self.shape, 3)

    def at(self, length_mm):
        m = self.manifest
        coefficients = np.array([
            np.interp(length_mm, m["path"]["L_mm"], np.asarray(m["path"]["coefficients"])[:, r])
            for r in range(self.rank)
        ])
        # Float32 node storage is part of the exported display contract.
        field = self.mean.copy()
        for coefficient, mode in zip(coefficients, self.modes):
            field[:] = field.astype(np.float64) + coefficient * mode.astype(np.float64)
        field = field.reshape(self.shape)
        lengths = np.asarray(m["geometry_L_mm"])
        right = int(np.clip(np.searchsorted(lengths, length_mm, side="right"), 1, len(lengths) - 1))
        left = right - 1
        fraction = float(np.clip((length_mm - lengths[left]) / (lengths[right] - lengths[left]), 0, 1))
        coordinates = (self.geometry[left].astype(np.float64) + fraction * (
            self.geometry[right].astype(np.float64) - self.geometry[left].astype(np.float64))).astype(np.float32)
        result = {}
        node_sets = {
            "x": [(i, 0, 0) for i in range(self.shape[0])],
            "y": [(self.shape[0] - 1, j, 0) for j in range(self.shape[1])],
            "z": [(self.shape[0] - 1, 0, k) for k in range(self.shape[2])],
        }
        for axis, nodes in node_sets.items():
            dimension = "xyz".index(axis)
            points = []
            for node in nodes:
                xyz = coordinates[node].astype(float)
                points.append([xyz[dimension], *xyz, float(field[node])])
                if axis != "x" and abs(xyz[dimension]) > 1e-12:
                    mirror = xyz.copy()
                    mirror[dimension] *= -1
                    points.append([mirror[dimension], *mirror, float(field[node])])
            result[axis] = np.asarray(sorted(points, key=lambda point: point[0]))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8071/index.html")
    parser.add_argument("--port", type=int, default=2841)
    parser.add_argument("--output", type=Path, default=ROOT / "build_profiles_review")
    parser.add_argument("--baseline-url")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"url": args.url, "browser": "Firefox Marionette", "checks": [], "passed": False}
    reference = ReferenceProfiles(ROOT / "volume")
    browser = Browser(args.port)

    def check(name, passed, details=None):
        record = {"name": name, "passed": bool(passed)}
        if details is not None:
            record["details"] = details
        report["checks"].append(record)
        if not passed:
            raise AssertionError(name)

    def navigate(url):
        separator = "&" if "?" in url else "?"
        browser.call("WebDriver:Navigate", {"url": url + separator + "paused=1"})
        check("page initialized: " + url, browser.js(
            'return document.body.dataset.ready==="true" && !document.getElementById("errorBox").textContent;'))

    def check_snapshot(label, snapshot):
        state, volume, profiles = snapshot["control"], snapshot["volume"], snapshot["profiles"]
        check(label + ": profile provenance and current state", profiles["source"] == "3d_display_nodes"
              and abs(profiles["actual_L_mm"] - state["actual_L_mm"]) < 1e-12
              and abs(profiles["P0_Pa"] - state["P0_Pa"]) < 1e-8
              and abs(profiles["time_s"] - state["time_s"]) < 1e-10
              and abs(volume["actual_L_mm"] - state["actual_L_mm"]) < 1e-12)
        expected = reference.at(state["actual_L_mm"])
        errors = {}
        for axis, count in (("x", 201), ("y", 41), ("z", 45)):
            line = profiles["lines"][axis]
            actual = np.asarray([[point[key] for key in VALUE_KEYS] for point in line], dtype=float)
            check(label + ": " + axis + " node count and finite values", len(line) == count
                  and actual.shape == expected[axis].shape and np.isfinite(actual).all())
            error = float(np.max(np.abs(actual - expected[axis])))
            errors[axis] = error
            check(label + ": " + axis + " independent ROM and geometry agreement", error <= 2e-6,
                  {"max_absolute_error": error, "points": count})
            check(label + ": " + axis + " physical coordinate order", np.all(np.diff(actual[:, 0]) > 0))
            if axis == "x":
                check(label + ": axial line ends exactly at x=0 and y=z=0",
                      actual[0, 0] < -5.8 and abs(actual[-1, 0]) < 1e-9
                      and np.max(np.abs(actual[:, 2:4])) < 1e-9)
            else:
                dimension = "xyz".index(axis)
                other = [1 + d for d in range(3) if d != dimension]
                check(label + ": " + axis + " reflected center appears once",
                      np.count_nonzero(np.abs(actual[:, 0]) < 1e-12) == 1
                      and np.max(np.abs(actual[:, 0] + actual[::-1, 0])) < 1e-9
                      and np.max(np.abs(actual[:, 4] - actual[::-1, 4])) < 1e-9
                      and np.max(np.abs(actual[:, other])) < 1e-9)
        return {"control": state, "max_absolute_errors": errors}

    def check_controller_tuning():
        check("controller tuning API available", browser.js('return ["getControllerTuning","setControllerTuning","resetControllerTuning"].every(k=>typeof window.nozzleSimulatorDebug[k]==="function");'))
        keys = ("max_speed_mm_s", "kp_per_s", "ki_per_s2")

        def values(settings):
            return {key: settings[key] for key in keys}

        def set_form(settings, invalid_key=None, invalid_value=None):
            fields = {"controllerMaxSpeed": str(settings["max_speed_mm_s"]),
                      "controllerKp": str(settings["kp_per_s"]),
                      "controllerKi": str(settings["ki_per_s2"])}
            if invalid_key:
                fields[invalid_key] = invalid_value
            return browser.js("const fields=" + json.dumps(fields) + ";for(const [id,value] of Object.entries(fields)){const el=document.getElementById(id);el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));}document.getElementById('controllerTuningApply').click();return {state:window.nozzleSimulatorDebug.getState(),tuning:window.nozzleSimulatorDebug.getControllerTuning()};")

        browser.js('window.nozzleSimulatorDebug.setRunning(false);window.nozzleSimulatorDebug.setMode("mach");return true;')
        defaults_mach = browser.js("return window.nozzleSimulatorDebug.getControllerTuning();")
        browser.js('window.nozzleSimulatorDebug.setMode("position");return true;')
        defaults_position = browser.js("return window.nozzleSimulatorDebug.getControllerTuning();")
        browser.js('window.nozzleSimulatorDebug.setMode("mach");window.nozzleSimulatorDebug.setCommand(1.5);window.nozzleSimulatorDebug.step(20);return true;')
        state_before = browser.js("return window.nozzleSimulatorDebug.getState();")
        check("speed-clamp test starts above the requested new limit", abs(state_before["velocity_mm_s"]) > .12)
        browser.js('const f=window.nozzleSimulatorDebug.getFields();window.__beforeTuningFields={longitudinal:Array.from(f.longitudinal),crossplane:Array.from(f.crossplane)};return true;')
        custom_mach = {"max_speed_mm_s": .12, "kp_per_s": 1.7, "ki_per_s2": .23}
        applied = set_form(custom_mach)
        check("tuning form applies explicit speed and P/I values", values(applied["tuning"]) == custom_mach)
        check("tuning apply preserves physical location, time and command", all(
            state_before[key] == applied["state"][key]
            for key in ("actual_L_mm", "time_s", "command", "actual_P0_Pa", "true_core_mach")))
        check("lower speed limit clips current velocity immediately", abs(applied["state"]["velocity_mm_s"]) <= .12 + 1e-12
              and abs(applied["state"]["velocity_command_mm_s"]) <= .12 + 1e-12)
        check("controller tuning does not change predicted 2D fields", browser.js('const f=window.nozzleSimulatorDebug.getFields();return ["longitudinal","crossplane"].every(key=>Array.from(f[key]).every((v,i)=>v===window.__beforeTuningFields[key][i]));'))
        limited = browser.js('const d=window.nozzleSimulatorDebug;let max=0;for(let i=0;i<80;i++){const s=d.step(1);max=Math.max(max,Math.abs(s.velocity_mm_s),Math.abs(s.velocity_command_mm_s));}return {max,state:d.getState(),tuning:d.getControllerTuning()};')
        check("new speed limit is enforced on subsequent controller steps", limited["max"] <= .12 + 1e-12,
              {"max_observed_speed_mm_s": limited["max"]})
        browser.js('window.nozzleSimulatorDebug.setView("both");return true;')
        report["numerical_samples"].append(check_snapshot("tuned moving state", browser.js(SNAPSHOT_JS)))
        for field in ("controllerMaxSpeed", "controllerKp", "controllerKi"):
            for invalid in ("", "-1"):
                previous = browser.js("return window.nozzleSimulatorDebug.getControllerTuning();")
                result = set_form(custom_mach, field, invalid)
                check("invalid " + field + " " + repr(invalid) + " preserves settings", result["tuning"] == previous)
                check("invalid " + field + " " + repr(invalid) + " is marked locally", browser.js(
                    "return document.getElementById(" + json.dumps(field) + ").getAttribute('aria-invalid')==='true' && document.getElementById('controllerTuningStatus').textContent.trim().length>0;"))
        check("non-finite tuning API input is rejected without changing settings", browser.js('''const d=window.nozzleSimulatorDebug,before=JSON.stringify(d.getControllerTuning());let rejected=0;
          for(const key of ["max_speed_mm_s","kp_per_s","ki_per_s2"]){for(const value of [NaN,Infinity]){try{d.setControllerTuning({[key]:value});}catch(e){rejected++;}}}
          return rejected===6 && JSON.stringify(d.getControllerTuning())===before;'''))
        set_form(custom_mach)
        browser.call("WebDriver:SetWindowRect", {"width": 1500, "height": 1100})
        time.sleep(.1)
        browser.screenshot(args.output / "controller_tuning_desktop.png")
        browser.js('window.nozzleSimulatorDebug.setMode("position");return true;')
        position = browser.js("return window.nozzleSimulatorDebug.getControllerTuning();")
        check("switching mode retains its own gains and the shared speed limit", position["mode"] == "position"
              and position["max_speed_mm_s"] == custom_mach["max_speed_mm_s"]
              and position["kp_per_s"] == defaults_position["kp_per_s"]
              and position["ki_per_s2"] == defaults_position["ki_per_s2"])
        custom_position = {"max_speed_mm_s": .12, "kp_per_s": 2.6, "ki_per_s2": .65}
        browser.js("window.nozzleSimulatorDebug.setControllerTuning(" + json.dumps(custom_position) + ");window.nozzleSimulatorDebug.setMode('mach');return true;")
        check("Mach gains survive separate position-mode editing", values(browser.js("return window.nozzleSimulatorDebug.getControllerTuning();")) == custom_mach)
        browser.js('window.nozzleSimulatorDebug.setControllerTuning({ki_per_s2:0});return true;')
        pure_p = browser.js('const d=window.nozzleSimulatorDebug;const immediate=d.getState();d.step(80);return {immediate,state:d.getState(),tuning:d.getControllerTuning()};')
        check("Ki=0 disables the integral state immediately and during motion", pure_p["tuning"]["ki_per_s2"] == 0
              and abs(pure_p["immediate"]["integral_velocity_mm_s"]) < 1e-12
              and abs(pure_p["state"]["integral_velocity_mm_s"]) < 1e-12)
        before_reset = browser.js("return window.nozzleSimulatorDebug.getState();")
        browser.js('document.getElementById("controllerTuningReset").click();return true;')
        reset = browser.js("return {state:window.nozzleSimulatorDebug.getState(),tuning:window.nozzleSimulatorDebug.getControllerTuning()};")
        check("reset button restores current Mach gains and shared speed", values(reset["tuning"]) == values(defaults_mach))
        check("reset preserves current position, time and command", all(reset["state"][key] == before_reset[key]
              for key in ("actual_L_mm", "time_s", "command")))
        browser.js('window.nozzleSimulatorDebug.setMode("position");return true;')
        after_reset_position = browser.js("return window.nozzleSimulatorDebug.getControllerTuning();")
        check("reset leaves other-mode gains intact", after_reset_position["kp_per_s"] == custom_position["kp_per_s"]
              and after_reset_position["ki_per_s2"] == custom_position["ki_per_s2"]
              and after_reset_position["max_speed_mm_s"] == defaults_mach["max_speed_mm_s"])
        browser.js('window.nozzleSimulatorDebug.resetControllerTuning();window.nozzleSimulatorDebug.setMode("mach");return true;')
        check("mode switch refreshes visible tuning controls", browser.js('const d=window.nozzleSimulatorDebug,t=d.getControllerTuning();return Number(document.getElementById("controllerMaxSpeed").value)===t.max_speed_mm_s && Number(document.getElementById("controllerKp").value)===t.kp_per_s && Number(document.getElementById("controllerKi").value)===t.ki_per_s2 && document.getElementById("controllerTuningMode").textContent.trim().length>0;'))
        controller_csv = browser.js("return window.nozzleSimulatorDebug.getCSV();")
        (args.output / "controller_tuning_history.csv").write_text(controller_csv)
        rows = list(csv.DictReader(io.StringIO(controller_csv.lstrip("\ufeff"))))
        setting_rows = [row for row in rows if row["event"] in ("controller_settings_change", "controller_settings_reset")]
        check("history CSV records tuning changes and reset events", len(setting_rows) >= 5 and
              {row["event"] for row in setting_rows} == {"controller_settings_change", "controller_settings_reset"})
        csv_keys = ("time_s", "max_speed_mm_s", "controller_kp_per_s", "controller_ki_per_s2")
        check("history CSV stores finite active speed and gains at each tuning event",
              all(np.isfinite(float(row[key])) for row in setting_rows for key in csv_keys))
        check("history CSV preserves the applied custom tuning", any(
              row["mode"] == "mach" and abs(float(row["max_speed_mm_s"]) - .12) < 1e-12
              and abs(float(row["controller_kp_per_s"]) - 1.7) < 1e-12
              and abs(float(row["controller_ki_per_s2"]) - .23) < 1e-12 for row in setting_rows))
        report["controller_tuning"] = {"mach_defaults": defaults_mach, "position_defaults": defaults_position,
                                       "custom_mach": custom_mach, "custom_position": custom_position,
                                       "setting_events": setting_rows}
        browser.call("WebDriver:SetWindowRect", {"width": 550, "height": 1000})
        time.sleep(.2)
        check("550px tuning form has no horizontal overflow", browser.js("return document.documentElement.scrollWidth<=innerWidth;"))
        browser.screenshot(args.output / "controller_tuning_mobile.png")

    try:
        browser.call("WebDriver:NewSession", {"capabilities": {"alwaysMatch": {"acceptInsecureCerts": True}}})
        browser.call("WebDriver:SetWindowRect", {"width": 1500, "height": 1100})
        baseline = None
        if args.baseline_url:
            navigate(args.baseline_url)
            baseline = browser.js(SAMPLES_JS)
            (args.output / "baseline_2d_states_fields.json").write_text(json.dumps(baseline) + "\n")
        navigate(args.url)
        browser.js('window.__profileErrors=[];window.addEventListener("error",e=>window.__profileErrors.push(e.message));window.addEventListener("unhandledrejection",e=>window.__profileErrors.push(String(e.reason)));return true;')
        check("profile API available", browser.js('return ["getVolumeProfiles","getVolumeProfileCSV","exportVolumeProfilesPNG"].every(k=>typeof window.nozzleSimulatorDebug[k]==="function");'))
        check("2D view hides profiles and leaves 3D assets lazy", browser.js('return document.getElementById("volumeProfiles").getBoundingClientRect().width===0 && !window.nozzleSimulatorDebug.getVolumeState().ready;'))
        if baseline is not None:
            current = browser.js(SAMPLES_JS)
            check("2D fields and controller states exactly equal unmodified HEAD", baseline == current,
                  {"conditions": len(current), "field_values": sum(len(s["longitudinal"]) + len(s["crossplane"]) for s in current)})
        before_volume = browser.js("return window.nozzleSimulatorDebug.getState();")
        browser.js('window.nozzleSimulatorDebug.setView("3d");return true;')
        deadline = time.monotonic() + 60
        volume = None
        while time.monotonic() < deadline:
            volume = browser.js("return window.nozzleSimulatorDebug.getVolumeState();")
            if volume["ready"] or volume["error"]:
                break
            time.sleep(.1)
        check("3D data ready", volume is not None and volume["ready"] and not volume["error"], volume)
        check("3D selection preserves state", browser.js("return window.nozzleSimulatorDebug.getState();") == before_volume)
        check("3D view exposes profiles", browser.js('return document.getElementById("volumeProfiles").getBoundingClientRect().width>0;'))
        states = [check_snapshot("initial 3D", browser.js(SNAPSHOT_JS))]
        before_command = browser.js(SNAPSHOT_JS)
        browser.js("window.nozzleSimulatorDebug.setCommand(2.25);return true;")
        commanded = browser.js(SNAPSHOT_JS)
        check("new command alone does not change current-L profiles",
              before_command["control"]["actual_L_mm"] == commanded["control"]["actual_L_mm"]
              and before_command["profiles"]["lines"] == commanded["profiles"]["lines"])
        browser.js("window.nozzleSimulatorDebug.step(240);return true;")
        moved = browser.js(SNAPSHOT_JS)
        check("actuator movement updates profiles", moved["control"]["actual_L_mm"] != commanded["control"]["actual_L_mm"]
              and moved["profiles"]["lines"] != commanded["profiles"]["lines"])
        states.append(check_snapshot("moving state", moved))
        for target in (1.75, 2.5, 3.25):
            browser.js(f"window.nozzleSimulatorDebug.setCommand({target});window.nozzleSimulatorDebug.step(6000);return true;")
            states.append(check_snapshot(f"target {target}", browser.js(SNAPSHOT_JS)))
        report["numerical_samples"] = states
        frozen = browser.js(SNAPSHOT_JS)
        browser.js('const d=window.nozzleSimulatorDebug;d.setVolumeSlice(0);const s=document.getElementById("volumeSymmetry");s.value="quarter";s.dispatchEvent(new Event("change",{bubbles:true}));return true;')
        controls = browser.js(SNAPSHOT_JS)
        check("slice and quarter display leave physical profile lines unchanged", frozen["profiles"]["lines"] == controls["profiles"]["lines"]
              and frozen["control"] == controls["control"])
        browser.js('const d=window.nozzleSimulatorDebug;d.setVolumeSlice(d.getVolumeState().grid_shape[0]-1);const s=document.getElementById("volumeSymmetry");s.value="full";s.dispatchEvent(new Event("change",{bubbles:true}));return true;')
        time.sleep(.25)
        check("paused state and profiles remain fixed", browser.js(SNAPSHOT_JS)["profiles"] == frozen["profiles"])
        browser.js('window.nozzleSimulatorDebug.setCommand(2);document.getElementById("pauseButton").click();return true;')
        time.sleep(.45)
        browser.js('window.nozzleSimulatorDebug.setRunning(false);return true;')
        resumed = browser.js(SNAPSHOT_JS)
        check("resume advances state and plotted profiles", resumed["control"]["time_s"] > frozen["control"]["time_s"]
              and resumed["profiles"]["lines"] != frozen["profiles"]["lines"])
        states.append(check_snapshot("resumed state", resumed))
        browser.screenshot(args.output / "profiles_desktop.png")
        csv_text = browser.js("return window.nozzleSimulatorDebug.getVolumeProfileCSV();")
        (args.output / "profiles_export.csv").write_text(csv_text)
        rows = list(csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff"))))
        check("CSV contains all three physical lines", len(rows) == 287 and {row["profile"] for row in rows} == {"x", "y", "z"})
        export_profiles = browser.js("return window.nozzleSimulatorDebug.getVolumeProfiles();")
        csv_error = 0.0
        for axis in "xyz":
            actual = np.asarray([[float(row["Mach" if key == "mach" else key]) for key in VALUE_KEYS]
                                 for row in rows if row["profile"] == axis])
            expected = np.asarray([[point[key] for key in VALUE_KEYS] for point in export_profiles["lines"][axis]])
            check("CSV " + axis + " count matches", actual.shape == expected.shape)
            csv_error = max(csv_error, float(np.max(np.abs(actual - expected))))
        check("CSV numeric values match the plotted profiles", csv_error <= 2e-6, {"max_absolute_error": csv_error})
        png = browser.js('const c=window.nozzleSimulatorDebug.exportVolumeProfilesPNG();return {width:c.width,height:c.height,png:c.toDataURL("image/png").split(",")[1]};')
        image_bytes = base64.b64decode(png["png"])
        (args.output / "profiles_export.png").write_bytes(image_bytes)
        pixels = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        check("profile PNG exports nonempty canvas", png["width"] >= 600 and png["height"] >= 300
              and np.unique(pixels.reshape(-1, 3), axis=0).shape[0] > 50, {"width": png["width"], "height": png["height"]})
        browser.js('window.nozzleSimulatorDebug.setView("both");return true;')
        time.sleep(.15)
        check("combined view exposes 2D, 3D and profiles", browser.js('return ["longCanvas","volumeCanvas","volumeProfiles"].every(id=>document.getElementById(id).getBoundingClientRect().width>0);'))
        browser.screenshot(args.output / "profiles_both.png")
        browser.call("WebDriver:SetWindowRect", {"width": 550, "height": 1000})
        time.sleep(.2)
        check("550px layout has no horizontal overflow", browser.js("return document.documentElement.scrollWidth<=innerWidth;"))
        browser.screenshot(args.output / "profiles_mobile.png")
        browser.js('window.nozzleSimulatorDebug.setView("2d");return true;')
        check("switching back to 2D hides profiles", browser.js('return document.getElementById("volumeProfiles").getBoundingClientRect().width===0;'))
        check_controller_tuning()
        report["browser_errors"] = browser.js("return window.__profileErrors;")
        check("no UI or runtime errors", not report["browser_errors"] and browser.js('return !document.getElementById("errorBox").textContent && !window.nozzleSimulatorDebug.getVolumeState().error;'))
        report["passed"] = True
    finally:
        (args.output / "profiles_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        try:
            browser.call("WebDriver:DeleteSession")
        finally:
            browser.socket.close()
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]), "report": str(args.output / "profiles_validation.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
