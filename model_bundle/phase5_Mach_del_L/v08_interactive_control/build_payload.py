#!/usr/bin/env python3
"""Export bounded steady ROM lookup tables for an offline live PI simulator."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
V07 = HERE.parent / "v07_flowfield_rom"
V05 = HERE.parent / "v05_surrogate_training"
sys.path.insert(0, str(V07))
from field_rom import predict_coefficients, reconstruct
from prepare_snapshots import digest, write_json
sys.path.insert(0, str(V05))
from surrogate_models import supported_mask


def load_checked_longitudinal():
    source = V07 / "results"
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ("field_model.pkl", "registered_snapshots.npz"):
        if digest(source / name) != manifest["outputs_sha256"][name]:
            raise ValueError(f"v07 input changed: {name}")
    return pickle.loads((source / "field_model.pkl").read_bytes()), dict(np.load(source / "registered_snapshots.npz"))


def model_payload(model, **grid):
    return {**grid, "mean": model["mean"].tolist(), "modes": model["modes"].tolist()}


def build(output=HERE / "results", sample_count=129):
    # New cross-section model is independent of the longitudinal model; both
    # use exactly the same 27 CFD input conditions and the same actual L/P0.
    from train_crossplane import core_weights
    output = Path(output)
    long_model, longitudinal = load_checked_longitudinal()
    cross_dir = output / "crossplane"
    cross_manifest = json.loads((cross_dir / "manifest.json").read_text())
    for name in ("field_model.pkl", "registered_snapshots.npz"):
        if digest(cross_dir / name) != cross_manifest["outputs_sha256"][name]:
            raise ValueError(f"Crossplane input changed: {name}")
    cross_model = pickle.loads((cross_dir / "field_model.pkl").read_bytes())
    cross = dict(np.load(cross_dir / "registered_snapshots.npz"))
    np.testing.assert_array_equal(longitudinal["case_ids"], cross["case_ids"])
    np.testing.assert_array_equal(longitudinal["X"], cross["X"])
    X, ids = longitudinal["X"], longitudinal["case_ids"].tolist()
    long_rom = model_payload(long_model, nx=len(longitudinal["x_edges_m"])-1,
                             neta=len(longitudinal["eta_edges"])-1,
                             x_edges_m=longitudinal["x_edges_m"].tolist(),
                             eta_edges=longitudinal["eta_edges"].tolist(),
                             ordering="eta rows, physical x columns; y=0 upper half")
    cross_rom = model_payload(cross_model, ny=len(cross["y_eta_edges"])-1,
                              nz=len(cross["z_eta_edges"])-1,
                              y_eta_edges=cross["y_eta_edges"].tolist(),
                              z_eta_edges=cross["z_eta_edges"].tolist(),
                              ordering="z_eta rows, y_eta columns; quarter mirrored for display",
                              plane_x_m=0., core_half_width_m=.05, core_half_height_m=.05)
    cases, audits = [], []
    for target, case_id in ((1.5,"M15_new2"),(2.,"M20_new2"),(2.5,"M25_new"),
                             (3.,"M30_new"),(3.5,"M35_new2")):
        initial = ids.index(case_id)
        pressure, initial_L = float(X[initial,1]), float(X[initial,0])
        selected = np.flatnonzero(X[:,1] == pressure)
        selected = selected[np.argsort(X[selected,0])]
        length = np.unique(np.r_[np.linspace(X[selected[0],0],X[selected[-1],0],sample_count),X[selected,0]])
        query = np.column_stack((length,np.full_like(length,pressure)))
        if not supported_mask(X,query).all():
            raise ValueError("Lookup queries outside observed input support")
        lc, cc = predict_coefficients(long_model,query), predict_coefficients(cross_model,query)
        long_wall_fn = PchipInterpolator(X[selected,0],longitudinal["wall_heights_m"][selected],axis=0,extrapolate=False)
        cross_wall_fn = PchipInterpolator(X[selected,0],cross["wall_heights_m"][selected],axis=0,extrapolate=False)
        width_fn = PchipInterpolator(X[selected,0],cross["width_m"][selected],extrapolate=False)
        wall, cwall, width = long_wall_fn(length),cross_wall_fn(length),width_fn(length)
        lf, cf = reconstruct(long_model,lc),reconstruct(cross_model,cc)
        core = np.array([field @ core_weights(float(w),h,cross["y_eta_edges"],cross["z_eta_edges"])
                         for field,w,h in zip(cf,width,cwall)])
        throat = 2000*wall.min(axis=1)
        if not (np.diff(throat)>0).all():
            raise ValueError(f"Non-monotonic throat geometry at target {target}")
        if not (np.diff(core)<0).all():
            raise ValueError(f"Non-monotonic core response at target {target}: check Mach mode support")
        # A midpoint comparison measures added error from using dense linear
        # browser lookups rather than silently treating the lookup as exact GP.
        middle=(length[:-1]+length[1:])/2
        mq=np.column_stack((middle,np.full_like(middle,pressure)))
        true_lc,true_cc=predict_coefficients(long_model,mq),predict_coefficients(cross_model,mq)
        approx_lc,approx_cc=(lc[:-1]+lc[1:])/2,(cc[:-1]+cc[1:])/2
        long_error=reconstruct(long_model,approx_lc)-reconstruct(long_model,true_lc)
        cross_error=reconstruct(cross_model,approx_cc)-reconstruct(cross_model,true_cc)
        direct_core=np.array([f @ core_weights(float(w),h,cross["y_eta_edges"],cross["z_eta_edges"])
                              for f,w,h in zip(reconstruct(cross_model,true_cc),width_fn(middle),cross_wall_fn(middle))])
        audit={"target_mach":target,"case_id":case_id,"same_pressure_CFD_cases":[ids[i] for i in selected],
               "lookup_nodes":len(length),"all_queries_inside_observed_hull":True,
               "longitudinal_max_lookup_error_mach":float(abs(long_error).max()),
               "crossplane_max_lookup_error_mach":float(abs(cross_error).max()),
               "core_mean_lookup_error_mach":float(abs((core[:-1]+core[1:])/2-direct_core).max()),
               "wall_lookup_error_m":float(abs((wall[:-1]+wall[1:])/2-long_wall_fn(middle)).max()),
               "minimum_predicted_mach":float(min(lf.min(),cf.min())),
               "Mach_range":[float(core.min()),float(core.max())],
               "throat_height_range_mm":[float(throat.min()),float(throat.max())]}
        if max(audit["longitudinal_max_lookup_error_mach"],audit["crossplane_max_lookup_error_mach"])>1e-4:
            raise ValueError("Dense lookup is insufficiently accurate; increase sample_count")
        if not all(np.isfinite(a).all() for a in (lf,cf,wall,cwall,width,core)) or min(width.min(),wall.min(),cwall.min())<=0:
            raise ValueError("Invalid simulated geometry or fields")
        audits.append(audit)
        initial_index=int(np.flatnonzero(length==initial_L)[0])
        cases.append({"id":case_id,"target_mach":target,"P0_Pa":pressure,
                      "L_min_mm":float(length[0]),"L_max_mm":float(length[-1]),"L_initial_mm":initial_L,
                      "h_min_mm":float(throat.min()),"h_max_mm":float(throat.max()),
                      "mach_min":float(core.min()),"mach_max":float(core.max()),
                      "color_min":0.,"color_max":float(max(target*1.1,lf.max(),cf.max())),
                      "long_delta_limit":float(max(.002,abs(lf-lf[initial_index]).max())),
                      "cross_delta_limit":float(max(.002,abs(cf-cf[initial_index]).max())),
                      "observed_case_ids":[ids[i] for i in selected],
                      "samples":{"L_mm":length.tolist(),"throat_height_mm":throat.tolist(),
                                 "core_mach":core.tolist(),"long_coefficients":lc.tolist(),
                                 "cross_coefficients":cc.tolist(),"wall_heights_m":wall.tolist(),
                                 "cross_wall_heights_m":cwall.tolist(),"cross_width_m":width.tolist()}})
    payload={"schema_version":1,"meta":{"title":"가변노즐 실시간 조작 시뮬레이터",
              "training_case_count":27,"field_physics":"quasi-steady Mach ROM at actual actuator L and fixed P0",
              "feedback_definition":"x=0 reconstructed Mach field, physical central 100x100mm area mean (full symmetry)",
              "time_dynamics":"assumed actuator velocity lag and measurement filter, not transient CFD",
              "case_change_policy":"independent fixed-P0 runs; selecting a case resets simulation",
              "source_longitudinal_model_sha256":digest(V07/'results/field_model.pkl'),
              "source_crossplane_model_sha256":digest(cross_dir/'field_model.pkl')},
             "field_roms":{"longitudinal":long_rom,"crossplane":cross_rom},"cases":cases,
             "config":{"dt_s":.05,"actuator_tau_s":.3,"max_speed_mm_s":.5,"measurement_tau_s":.2,
                       "position_kp":2.,"position_ki":.5,"mach_kp":1.2,"mach_ki":.2,"antiwindup_tau_s":.5}}
    write_json(output/'simulator_payload.json',payload)
    write_json(output/'lookup_validation.json',{"cases":audits,"interpolation":"linear in actual L, within one fixed-P0 observed interval",
               "validation_location":"every lookup-interval midpoint compared with original trained GP and PCHIP geometry",
               "live_feedback":"runtime integrates its reconstructed cross-section field, not interpolated core_mach scalar",
               "field_accuracy_scope":"lookup error is additional numerical approximation error, not CFD model error"})
    print(f"Built live simulator payload: {output/'simulator_payload.json'}",flush=True)
    return payload


if __name__=="__main__":
    build()
