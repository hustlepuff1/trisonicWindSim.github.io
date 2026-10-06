#!/usr/bin/env python3
"""Build a continuous, coordinated L/P0 operating path from frozen steady ROMs.

P0 uses the existing nominal 80 kg/s design equation. At each reference Mach,
invert the reconstructed cross-section's physical central mean for L. Neither
the field models nor the design-pressure formula are fitted to new CFD here.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
V08 = HERE.parent / "v08_interactive_control"
V07 = HERE.parent / "v07_flowfield_rom"
V05 = HERE.parent / "v05_surrogate_training"
sys.path.insert(0, str(V08))
from build_payload import load_checked_longitudinal, model_payload
from train_crossplane import core_weights
from field_rom import predict_coefficients, reconstruct
from prepare_snapshots import digest, write_json
from pressure_schedule import p0_reference


class FrozenModels:
    def __init__(self):
        self.long_model, self.long = load_checked_longitudinal()
        source = V08 / "results/crossplane"
        manifest = json.loads((source / "manifest.json").read_text())
        for name in ("field_model.pkl", "registered_snapshots.npz"):
            if digest(source / name) != manifest["outputs_sha256"][name]:
                raise ValueError(f"Frozen crossplane source changed: {name}")
        self.cross_model = pickle.loads((source / "field_model.pkl").read_bytes())
        self.cross = dict(np.load(source / "registered_snapshots.npz"))
        np.testing.assert_array_equal(self.long["case_ids"], self.cross["case_ids"])
        np.testing.assert_array_equal(self.long["X"], self.cross["X"])
        self.X = self.long["X"]
        self.ids = self.long["case_ids"].tolist()
        order = np.argsort(self.X[:, 0])
        length = self.X[order, 0]
        if not np.all(np.diff(length) > 0):
            raise ValueError("Geometry interpolation requires distinct observed L")
        self.bounds = (float(length[0]), float(length[-1]))
        self.wall = PchipInterpolator(length, self.long["wall_heights_m"][order], axis=0, extrapolate=False)
        self.crosswall = PchipInterpolator(length, self.cross["wall_heights_m"][order], axis=0, extrapolate=False)
        self.width = PchipInterpolator(length, self.cross["width_m"][order], extrapolate=False)
        self.flat_cross_section = bool(np.ptp(self.cross["wall_heights_m"], axis=1).max() < 1e-14)
        features = np.column_stack((self.X[:, 0], np.log(self.X[:, 1])))
        self.center, self.scale = features.mean(axis=0), features.std(axis=0)
        self.equations = ConvexHull((features-self.center)/self.scale).equations

    def supported(self, query):
        q = np.asarray(query)
        features = np.column_stack((q[:, 0], np.log(q[:, 1])))
        return np.all(((features-self.center)/self.scale) @ self.equations[:, :2].T + self.equations[:, 2] <= 1e-10, axis=1)

    def weights(self, length):
        width, wall = float(self.width(length)), self.crosswall(length)
        if not self.flat_cross_section:
            return core_weights(width, wall, self.cross["y_eta_edges"], self.cross["z_eta_edges"])
        # For this dataset the quarter section is rectangular. Exact clipped
        # rectangle areas equal the general quadrilateral clipping routine.
        dy = np.diff(np.clip(width*self.cross["y_eta_edges"], 0., .05))
        dz = np.diff(np.clip(float(wall[0])*self.cross["z_eta_edges"], 0., .05))
        area = np.outer(dz, dy).ravel()
        if abs(area.sum()-.0025) > 1e-13:
            raise ValueError("Predicted geometry does not contain the physical core")
        return area/area.sum()

    def core(self, length, pressure):
        coefficients = predict_coefficients(self.cross_model, [[length, pressure]])[0]
        weights = self.weights(length)
        return float(self.cross_model["mean"] @ weights + coefficients @ (self.cross_model["modes"] @ weights))

    def inverse(self, mach):
        # Match the historical CFD pressure-table resolution, including the
        # endpoint hull boundaries, before solving for L at that exact pressure.
        pressure = float(np.round(p0_reference(mach), 3))
        length = brentq(lambda x: self.core(x, pressure)-mach, *self.bounds, xtol=1e-10)
        return length, pressure


def build(output=HERE / "results", count=801):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    models = FrozenModels()
    for length in np.linspace(*models.bounds, 11):
        np.testing.assert_allclose(models.weights(length), core_weights(float(models.width(length)), models.crosswall(length),
            models.cross["y_eta_edges"], models.cross["z_eta_edges"]), rtol=1e-12, atol=1e-14)
    targets = np.linspace(1.5, 3.5, count)
    query = np.array([models.inverse(float(m)) for m in targets])
    residuals = np.array([models.core(l, p)-m for (l, p), m in zip(query, targets)])
    if abs(residuals).max() > 1e-10 or not (np.diff(query[:, 0]) < 0).all():
        raise ValueError("Continuous inverse path is not accurate and monotonic")
    # Browser lookup interpolation uses ascending actual L.
    query, targets = query[::-1], targets[::-1]
    length, pressure = query.T
    lc = predict_coefficients(models.long_model, query)
    cc = predict_coefficients(models.cross_model, query)
    wall, crosswall, width = models.wall(length), models.crosswall(length), models.width(length)
    lf, cf = reconstruct(models.long_model, lc), reconstruct(models.cross_model, cc)
    core = np.array([f @ models.weights(l) for f, l in zip(cf, length)])
    throat = 2000*wall.min(axis=1)
    if not (np.diff(throat)>0).all() or not (np.diff(core)<0).all():
        raise ValueError("Non-monotonic geometry or physical core response")
    if not all(np.isfinite(a).all() for a in (lf,cf,wall,crosswall,width,core)) or min(wall.min(),crosswall.min(),width.min()) <= 0:
        raise ValueError("Nonfinite fields or invalid predicted geometry")
    midq=(query[:-1]+query[1:])/2
    direct_lc=predict_coefficients(models.long_model,midq)
    direct_cc=predict_coefficients(models.cross_model,midq)
    approx_lc,approx_cc=(lc[:-1]+lc[1:])/2,(cc[:-1]+cc[1:])/2
    long_error=(approx_lc-direct_lc) @ models.long_model["modes"]
    cross_error=(approx_cc-direct_cc) @ models.cross_model["modes"]
    true_mid_core=np.array([models.core(l,p) for l,p in midq])
    audit={"lookup_nodes":len(length),"inverse_max_core_residual_mach":float(abs(residuals).max()),
           "inside_observed_hull_nodes":int(models.supported(query).sum()),
           "inside_observed_hull_midpoints":int(models.supported(midq).sum()),
           "midpoint_count":len(midq),"training_hull_space":"standardized [L_mm, ln(P0_Pa)]",
           "runtime_hull_test":"exact halfspace test at current L and interpolated pressure",
           "longitudinal_max_lookup_error_mach":float(abs(long_error).max()),
           "crossplane_max_lookup_error_mach":float(abs(cross_error).max()),
           "core_mean_lookup_error_mach":float(abs((core[:-1]+core[1:])/2-true_mid_core).max()),
           "wall_lookup_error_m":float(abs((wall[:-1]+wall[1:])/2-models.wall(midq[:,0])).max()),
           "reference_pressure_lookup_error_Pa":float(abs(midq[:,1]-p0_reference((targets[:-1]+targets[1:])/2)).max()),
           "minimum_predicted_mach":float(min(lf.min(),cf.min())),
           "L_range_mm":[float(length.min()),float(length.max())],
           "throat_height_range_mm":[float(throat.min()),float(throat.max())],
           "validation_location":"all interval midpoints compared with original GP at current L and interpolated P0",
           "geometry_method":"PCHIP vs linear interpolation of observed geometry as a function of actual L, no extrapolation",
           "scope":"lookup numerical error; not independent CFD model validation"}
    if max(audit["longitudinal_max_lookup_error_mach"],audit["crossplane_max_lookup_error_mach"])>1e-4:
        raise ValueError("Increase path lookup resolution to meet 1e-4 Mach numerical error")
    long_rom=model_payload(models.long_model,nx=len(models.long["x_edges_m"])-1,neta=len(models.long["eta_edges"])-1,
        x_edges_m=models.long["x_edges_m"].tolist(),eta_edges=models.long["eta_edges"].tolist(),
        ordering="eta rows, physical x columns; y=0 upper half")
    cross_rom=model_payload(models.cross_model,ny=len(models.cross["y_eta_edges"])-1,nz=len(models.cross["z_eta_edges"])-1,
        y_eta_edges=models.cross["y_eta_edges"].tolist(),z_eta_edges=models.cross["z_eta_edges"].tolist(),
        ordering="z_eta rows, y_eta columns; quarter mirrored for display",plane_x_m=0.,core_half_width_m=.05,core_half_height_m=.05)
    case={"id":"continuous","target_mach":1.5,"P0_Pa":float(pressure[-1]),
        "L_min_mm":float(length[0]),"L_max_mm":float(length[-1]),"L_initial_mm":float(length[-1]),
        "h_min_mm":float(throat.min()),"h_max_mm":float(throat.max()),"mach_min":1.5,"mach_max":3.5,
        "color_min":0.,"color_max":float(max(3.85,lf.max(),cf.max())),
        "long_delta_limit":float(abs(lf-lf[-1]).max()),"cross_delta_limit":float(abs(cf-cf[-1]).max()),
        "observed_case_ids":models.ids,"samples":{"L_mm":length.tolist(),"throat_height_mm":throat.tolist(),
        "core_mach":core.tolist(),"long_coefficients":lc.tolist(),"cross_coefficients":cc.tolist(),
        "wall_heights_m":wall.tolist(),"cross_wall_heights_m":crosswall.tolist(),"cross_width_m":width.tolist(),
        "P0_Pa":pressure.tolist(),"schedule_mach":targets.tolist(),"training_hull_supported":models.supported(query).tolist()}}
    payload={"schema_version":2,"meta":{"title":"연속 운전 가변노즐 제어 시뮬레이터",
        "training_case_count":len(models.ids),"field_physics":"quasi-steady Mach ROM at actual L and coordinated P0",
        "feedback_definition":"physical central full 100x100mm area mean of the reconstructed x=0 Mach field",
        "time_dynamics":"assumed actuator velocity lag and measurement filter; no learned transient CFD",
        "pressure_tracking":"ideal instantaneous pressure tracking of the current actual actuator path",
        "pressure_schedule":"nominal full test-section design mass flow 80kg/s, T0=300K, A=0.25m2, gamma=1.4, R=287; rounded 0.001Pa",
        "actual_mass_flow_regulated":False,"geometry_model":"PCHIP in observed actual L; no new structural solution",
        "path_policy":"single continuous path; command and mode changes do not reset actual actuator position or time",
        "source_longitudinal_model_sha256":digest(V07/'results/field_model.pkl'),
        "source_crossplane_model_sha256":digest(V08/'results/crossplane/field_model.pkl')},
        "support_hull":{"center":models.center.tolist(),"scale":models.scale.tolist(),"equations":models.equations.tolist(),"tolerance":1e-10},
        "field_roms":{"longitudinal":long_rom,"crossplane":cross_rom},"cases":[case],
        "config":{"dt_s":.05,"actuator_tau_s":.3,"max_speed_mm_s":.5,"measurement_tau_s":.2,
        "position_kp":2.,"position_ki":.5,"mach_kp":1.2,"mach_ki":.2,"antiwindup_tau_s":.5}}
    write_json(output/'simulator_payload.json',payload)
    write_json(output/'lookup_validation.json',audit)
    with (output/'operating_schedule.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['target_mach','L_mm','P0_Pa','full_throat_height_mm','predicted_core_mean_mach','training_hull_supported'])
        writer.writerows(zip(targets[::-1],length[::-1],pressure[::-1],throat[::-1],core[::-1],models.supported(query)[::-1]))
    print(json.dumps(audit,indent=2),flush=True)
    print(f"Payload ready: {output/'simulator_payload.json'}",flush=True)
    freeze_prediction(models,1.75,output/'frozen_validation/M1p75')
    return payload


def freeze_prediction(models,mach,output):
    """Freeze one prospective input/field bundle; never revise an existing one."""
    output=Path(output)
    if output.exists():
        manifest=json.loads((output/'manifest.json').read_text())
        for name,expected in manifest['outputs_sha256'].items():
            if digest(output/name)!=expected:
                raise ValueError(f"Frozen prediction changed: {output/name}")
        old=json.loads((output/'prediction.json').read_text())
        if old['inputs']['target_mach']!=mach or old['source_models']['longitudinal']['sha256']!=digest(V07/'results/field_model.pkl') or old['source_models']['crossplane']['sha256']!=digest(V08/'results/crossplane/field_model.pkl'):
            raise ValueError("Use a new frozen prediction directory for a new model/input")
        print(f"Preserved existing frozen prediction: {output}",flush=True)
        return
    output.mkdir(parents=True)
    length,pressure=models.inverse(mach)
    q=[[length,pressure]]
    lf=reconstruct(models.long_model,predict_coefficients(models.long_model,q))[0]
    cf=reconstruct(models.cross_model,predict_coefficients(models.cross_model,q))[0]
    wall,cwall,width=models.wall(length),models.crosswall(length),float(models.width(length))
    provenance=json.loads((V05/'results/dataset_provenance.json').read_text())
    prediction={"status":"prediction_frozen_before_new_CFD","frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "inputs":{"target_mach":mach,"L_mm":length,"P0_Pa":pressure,"T0_K":300.,"outlet_static_pressure_Pa":10132.5},
        "design_reference_full_mass_flow_kg_s":80.,"mass_flow_boundary_condition":False,
        "core_definition":"full100x100mm central","predicted_core_mean_mach":float(cf@models.weights(length)),
        "predicted_full_throat_height_mm":float(2000*wall.min()),"training_hull_supported":bool(models.supported(np.array(q))[0]),
        "CFD_comparison_status":"pending_new_independent_CFD","CFD_error_metrics":None,
        "training_case_ids":models.ids,"training_crossplane_sha256":[s['plane_sha256'] for s in provenance['sources']],
        "training_raw_volume_sha256":[s['source']['sha256'] for s in provenance['sources']],
        "source_models":{"longitudinal":{"path":str(V07/'results/field_model.pkl'),"sha256":digest(V07/'results/field_model.pkl')},
            "crossplane":{"path":str(V08/'results/crossplane/field_model.pkl'),"sha256":digest(V08/'results/crossplane/field_model.pkl')}},
        "source_geometry_sha256":{"longitudinal":digest(V07/'results/registered_snapshots.npz'),"crossplane":digest(V08/'results/crossplane/registered_snapshots.npz')},
        "source_schedule_sha256":digest(HERE/'pressure_schedule.py'),"source_builder_sha256":digest(Path(__file__)),
        "geometry_assumption":"PCHIP interpolation of existing deformed wall coordinates by actual L, not a new FEA result",
        "independent_case_definition":"new [L_mm,P0_Pa] input pair; nearby output Mach in existing CFD is not a held-out case",
        "validation_protocol":["Keep this prediction and the trained ROMs fixed before running new CFD.",
            "Generate geometry at this exact L using the same structural/CAD procedure; retain the actual geometry for separate comparison.",
            "Use the declared P0/T0/outlet conditions and consistent solver/mesh/convergence practice; record actual input deck and mass flow.",
            "Extract y=0 longitudinal and x=0 quarter cross-section caches using the existing extraction tools.",
            "Compare both fields and physical central mean; report geometry differences and grid/convergence evidence.",
            "Do not retrain on this case before reporting the prospective validation; do not claim steady CFD validates physical transient control."]}
    write_json(output/'prediction.json',prediction)
    np.savez_compressed(output/'predicted_fields.npz',
        longitudinal_Mach=lf.reshape(len(models.long['eta_edges'])-1,-1),
        crossplane_Mach=cf.reshape(len(models.cross['z_eta_edges'])-1,-1),
        long_x_edges_m=models.long['x_edges_m'],long_eta_edges=models.long['eta_edges'],
        cross_y_eta_edges=models.cross['y_eta_edges'],cross_z_eta_edges=models.cross['z_eta_edges'],
        long_wall_heights_m=wall,cross_wall_heights_m=cwall,cross_width_m=width,
        core_mean_mach=prediction['predicted_core_mean_mach'],long_weights=models.long['weights'],cross_weights=models.cross['weights'])
    with (output/'cfd_inputs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(prediction['inputs']))
        writer.writeheader();writer.writerow(prediction['inputs'])
    np.savetxt(output/'predicted_longitudinal_upper_wall.csv',np.column_stack((models.long['x_edges_m'],wall)),delimiter=',',header='x_m,z_upper_wall_m',comments='')
    np.savetxt(output/'predicted_crossplane_upper_wall.csv',np.column_stack((-width*models.cross['y_eta_edges'],cwall)),delimiter=',',header='y_m,z_upper_wall_m',comments='')
    write_json(output/'manifest.json',{"status":"immutable prospective steady-field prediction","generated_at_utc":prediction['frozen_at_utc'],
        "outputs_sha256":{p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest.json'}})
    print(f"Frozen M{mach}: L={length:.12f} mm, P0={pressure:.3f} Pa; {output}",flush=True)


if __name__=='__main__':
    build()
