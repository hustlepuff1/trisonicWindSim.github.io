#!/usr/bin/env python3
"""Build the Korean inlet-to-x=0 validation page from the new nozzle results.

The builder rejects the previous full-domain model and never reads its error
figures or geometry report. Counts, retained rank and validation statistics
come from the matching completed v11 model and browser export.
"""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGIONS = (
    ("nozzle_volume", "입구 → 시험부 x = 0"),
    ("upstream_x_lt_minus_0p5", "입구측 · 기준 x < −0.5 m"),
    ("terminal_x_minus_0p5_to_0", "시험부 앞 · 기준 −0.5 ≤ x ≤ 0 m"),
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def case_label(case_id: str) -> str:
    return escape({"M175": "M1.75", "M325": "M3.25"}.get(case_id, case_id))


def html_table(headings, rows):
    head = "".join(f'<th scope="col">{escape(label)}</th>' for label in headings)
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render(report, export, manifest, protocol, field_rows):
    if report.get("status") != "completed" or report.get("fresh_nozzle_only_fit") is not True:
        raise ValueError("A completed freshly fitted nozzle model is required")
    domain = manifest.get("domain", {})
    if domain.get("id") != "nozzle_x_le_0" or domain.get("x_max_m") != 0 or domain.get("endpoint") != "physical_x_plane":
        raise ValueError("The active display must end at the physical x=0 nozzle plane")
    if protocol.get("fresh_nozzle_only_fit") is not True or protocol.get("cropped_existing_full_domain_POD_or_GP") is not False:
        raise ValueError("The training protocol must record a fresh nozzle-only fit")
    training = manifest["training"]
    if training.get("fresh_fit") is not True or training.get("previous_full_domain_model_used") is not False:
        raise ValueError("Browser data must come from the new nozzle model")
    cells = int(report["native_cell_count"])
    if cells != int(training["cell_count"]) or cells != int(export["native_cell_count"]):
        raise ValueError("Model and display training-cell counts disagree")
    if int(report["rank"]) != int(manifest["rank"]):
        raise ValueError("Model and display ranks disagree")
    count = int(report["training_condition_count"])
    if count != int(training["case_count"]):
        raise ValueError("Model and display training-condition counts disagree")
    held = [r for r in field_rows if r["scheme"] == "independent_retrospective_heldout"]
    folds = [r for r in field_rows if r["scheme"] == "leave_one_condition_out"]
    if len(held) != int(report["heldout_condition_count"]) or len(folds) != count:
        raise ValueError("Held-out or LOCO metric rows are incomplete")
    expected_ids = set(protocol["heldout_case_ids"])
    if {r["case_id"] for r in held} != expected_ids:
        raise ValueError("Held-out case IDs do not match the training protocol")
    if expected_ids & set(protocol["training_case_ids"]):
        raise ValueError("A held-out case appears in the training cases")
    loco = next(r for r in report["validation_summary"]
                if r["scheme"] == "leave_one_condition_out" and r["subset"] == "all")
    outside = sum(str(r["in_training_input_hull"]).lower() == "false" for r in folds)
    held_rows = []
    for row in held:
        for key, label in REGIONS:
            held_rows.append((case_label(row["case_id"]), escape(label),
                              f'{float(row[key + "_weighted_rmse_mach"]):.5f}',
                              f'{float(row[key + "_relative_weighted_L2_percent"]):.2f}%'))
    held_table = html_table(["조건", "평가 영역", "RMSE [Mach]", "상대 L₂ 오차"], held_rows)
    range_rows = []
    for row in report["heldout_native_cell_metrics"]:
        range_rows.append((case_label(row["case_id"]),
                           f'{float(row["minimum_predicted_mach"]):.5f}',
                           f'{float(row["maximum_predicted_mach"]):.5f}',
                           f'{int(row["negative_prediction_cell_count"]):,}',
                           f'{float(row["max_abs_error_mach"]):.5f}'))
    range_table = html_table(["조건", "예측 최솟값", "예측 최댓값", "음수 예측 셀", "최대 절대 오차 [Mach]"], range_rows)
    geometry_rows = []
    for row in export["heldout_geometry_interpolation"]:
        geometry_rows.append((case_label(row["case_id"]),
                              f'{int(row["node_count"]):,}',
                              f'{1000 * float(row["node_displacement_max_m"]):.3f} mm',
                              escape(" / ".join(row["bracketing_cases"])) ))
    geometry_table = html_table(["조건", "비교 절점 수", "최대 대응 절점 차이", "보간에 사용한 형상"], geometry_rows)
    original = export.get("unchanged_native_cell_count")
    terminal = export.get("terminal_aggregated_cell_count")
    if original is None or terminal is None:
        raise ValueError("Export audit must state unchanged and terminal-aggregated cell counts")
    original, terminal = int(original), int(terminal)
    if original < 1 or terminal < 1 or original + terminal != cells:
        raise ValueError("Nozzle registration cell counts do not add up")
    layer_text = f" {original // terminal:,}개 층의" if original % terminal == 0 else ""
    cut_note = (f"입구측{layer_text} 원래 셀 {original:,}개는 값을 유지했다. 끝단에서는 x = 0까지 남는 "
                f"셀 부분을 횡방향 열별로 합쳐 마지막 층 {terminal:,}개를 구성했다. 이때 셀 체적과 "
                "마하수의 체적적분을 보존하는 체적 가중 평균을 사용했다. 마지막 표시 단면의 모든 절점은 정확히 x = 0이다.")
    final_converged = report["full_model_all_GP_optimizers_converged"] is True
    failed_folds = int(report["folds_with_nonconverged_GP_optimizers"])
    if final_converged and failed_folds == 0:
        convergence = "최종 모델과 모든 교차검증 fold에서 선택된 GP 최적화가 수렴했다."
    else:
        convergence = ("최종 모델의 모든 GP 최적화 수렴 여부: " + ("예" if final_converged else "아니오") +
                       f". 미수렴 GP를 포함한 교차검증 fold는 {failed_folds:,}개이며, 자세한 상태는 수치 검증 자료에 기록했다.")
    color_low, color_high = map(float, manifest["color_range"])
    held_names = " · ".join(case_label(row["case_id"]) for row in held)
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>노즐 3D 마하수 모델 검증 · 입구부터 x=0</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f7;color:#193747;font:15px/1.7 system-ui,sans-serif}}main{{max-width:1040px;margin:auto;padding:34px 22px 44px}}a{{color:#206da1;text-underline-offset:3px}}a:focus-visible{{outline:3px solid #7cb6d8;outline-offset:3px}}h1{{font-size:30px;line-height:1.35;margin:10px 0}}h2{{font-size:20px;margin-top:0}}p{{margin:10px 0}}.eyebrow{{color:#567583;font-size:12px;letter-spacing:.03em}}.card{{background:white;padding:24px;border:1px solid #d4e0e5;border-radius:12px;margin:18px 0}}.scope{{border-left:5px solid #367a91}}.small{{font-size:13px;color:#55707d}}.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:10px 12px;border-bottom:1px solid #dfe8ec;text-align:left;white-space:nowrap}}thead{{background:#edf4f7}}.numbers{{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}}.numbers strong{{font-size:27px;display:block;white-space:nowrap}}.numbers span{{font-size:12px;color:#55707d}}.boundary{{display:flex;align-items:center;gap:14px;margin:16px 0 3px;color:#315b6e;font-size:14px}}.boundary .line{{height:3px;flex:1;background:linear-gradient(90deg,#96c4d3,#367a91);position:relative}}.boundary .line:after{{content:"";position:absolute;right:-1px;top:-4px;border-top:5px solid transparent;border-bottom:5px solid transparent;border-left:8px solid #367a91}}.formula{{font-family:system-ui,sans-serif;white-space:normal}}code{{font-size:13px}}@media(max-width:700px){{main{{padding:20px 12px}}.card{{padding:17px}}h1{{font-size:25px}}.numbers{{grid-template-columns:repeat(2,1fr);gap:13px}}.numbers strong{{font-size:24px}}.boundary{{font-size:12px}}}}
</style></head><body><main>
<a href="../?view=3d">← 노즐 3D 시뮬레이터</a>
<p class="eyebrow">입구 → 시험부 x = 0 · 정상 CFD 기반 POD–GPR</p>
<h1>노즐 3D 마하수 모델 검증</h1>
<p>사용자가 지정한 노즐 범위에 대해 {count:,}개 CFD 조건의 평균장·POD 기저·차수·모든 GPR을 새로 학습했다. 기존 2D 모델과 제어 계산은 유지한다.</p>
<section class="card scope"><h2>학습과 표시의 끝점: 시험부 x = 0</h2>
<p>평가 대상은 입구부터 x = 0까지의 ¼ 노즐 체적이다. 학습 조건을 대칭 복사하지 않고, 표시할 때 y = 0과 z = 0에서 형상을 반사한다.</p>
<div class="boundary" aria-label="학습 범위는 입구부터 정확한 x=0 시험부 평면까지"><strong>입구</strong><span class="line" aria-hidden="true"></span><strong>시험부 · x = 0</strong></div>
<p class="small">{cut_note}</p></section>
<section class="card"><div class="numbers"><div><strong>{count:,}</strong><span>학습 CFD 조건</span></div><div><strong>{cells:,}</strong><span>조건별 공통 노즐 체적 셀</span></div><div><strong>{int(report['rank']):,}</strong><span>노즐 POD 모드</span></div><div><strong>{100 * float(report['achieved_energy_fraction']):.4f}%</strong><span>학습장 변동 에너지</span></div></div><p class="small">에너지 보존율은 새로운 조건의 예측 정확도가 아니다. 아래 오차는 이 노즐 영역에서 계산했으며, 평가 영역과 정규화 분모가 다른 모델의 오차와 직접적인 개선율로 비교하지 않는다.</p></section>
<section class="card"><h2>학습에서 제외한 CFD 조건</h2>
<p>{held_names}는 이 모델의 학습·차수 선택에서 제외했다. 이미 보유한 CFD로 수행한 사후 독립 조건 평가이며, 과거 2D 모델의 사전 고정 예측과 구분한다.</p>
{held_table}
<p class="small formula">w는 정밀한 M17 기준 형상에서 계산한 양의 노즐 셀 체적이다. RMSE = √(Σw·오차²/Σw), 상대 L₂ = 100√(Σw·오차²/Σw·CFD²). 세부 영역은 M17 기준 셀 중심 좌표로 나눈다. 각 변형 조건 자체의 체적 적분과는 다르며, 이 표의 장 오차는 별도 형상 보간 오차를 포함하지 않는다.</p></section>
<section class="card"><h2>조건 단위 교차검증</h2>
<p>{len(folds):,}회 LOCO에서 매번 CFD 조건 하나 전체를 제외하고 평균장·POD 기저·차수·입력 정규화·GPR을 다시 학습했다. 노즐 체적의 조건 간 제곱평균 RMSE는 <strong>{float(loco['nozzle_volume_weighted_rmse_mach']):.5f} Mach</strong>, 조건별 상대 L₂ 오차 평균은 <strong>{float(loco['mean_nozzle_volume_relative_weighted_L2_percent']):.2f}%</strong>였다.</p>
<p>제외한 조건 중 {outside:,}개는 해당 fold의 학습 입력 볼록껍질 밖에 있었다. {convergence} 공간 셀을 무작위로 나누는 검증은 사용하지 않았다.</p></section>
<section class="card"><h2>형상 보간의 별도 검사</h2>
<p>표시 형상은 관측된 실제 3D 절점을 L에 따라 보간한다. 입구부터 x = 0까지의 대응 절점에서 다음 차이를 확인했다. 이 값에는 내부 격자 배치 차이도 포함되며, 마하수 오차와는 별도의 지표다.</p>
{geometry_table}
<p class="small">정확한 x = 0 끝점에서도 y·z 방향 형상 보간 차이는 남을 수 있다. 이 검사는 새로운 FEA나 CFD 계산을 수행한 결과가 아니다.</p></section>
<section class="card"><h2>예측값의 수치 범위</h2>
<p>GPR은 마하수의 비음수 조건을 강제하지 않는다. 다음 값은 각 독립 조건의 모든 공통 노즐 셀에서 확인했으며, 오차 계산과 저장에서 음수를 절삭하지 않았다.</p>
{range_table}
<p>3D 색범위는 {color_low:g}–{color_high:g}이며, 범위 밖 값은 색상표의 끝색으로 표시한다. 두 모델을 함께 볼 때는 2D와 3D의 각 범례 숫자를 확인한다.</p></section>
<section class="card"><h2>웹 표시와 조회 오차</h2>
<p>학습에 사용한 {cells:,}개 공통 노즐 셀로 만든 모델을 웹에서는 {int(export['display_node_count']):,}개 표시 절점으로 재구성한다. 표시 절점의 마하수는 인접 셀 평균이다. 내부 횡단면은 격자 좌표면이므로 x≈ 값과 실제 x 범위를 함께 표시하고, 마지막 시험부 단면은 정확한 x = 0 평면으로 표시한다.</p>
<p>브라우저 경로 조회값을 구간 중점 {int(export['lookup_midpoint_count']):,}개에서 직접 GP 계산과 비교했다. 최대 기준 체적 RMSE는 {float(export['lookup_max_reference_volume_rmse_mach']):.2e} Mach, 최대 표시 절점 차이는 {float(export['lookup_max_display_node_error_mach']):.2e} Mach였다. 이는 모델 자체의 CFD 예측 오차와 별개의 구현 검사다.</p></section>
<section class="card"><h2>해석 범위와 재현 자료</h2>
<p>이 모델은 정상 CFD 스냅샷을 이용한 준정상 마하수 예측이다. 중앙 Mach 제어 피드백은 기존 2D 단면 모델을 사용하며, 구동기·센서 동특성과 이상적인 총압 추종은 기존 가정이다. 비정상 유동 학습이나 풍동 실험 동기화를 검증한 결과로 해석하지 않는다. 입력 조건은 기존 L–P₀ 연동 운전 경로에 집중되어 있다.</p>
<p><a href="validation.json">수치 검증 JSON</a> · <a href="field_metrics.csv">조건·영역별 CSV</a> · <a href="export_audit.json">표시·형상 보간 검사</a> · <a href="../model_bundle/phase5_Mach_del_L/v11_nozzle_volume/README.md">노즐 모델 학습 절차</a></p>
<p class="small">원본 CFD의 출처와 정합·x = 0 경계 처리 과정은 학습 절차 및 추출 manifest에 기록한다. 전체 해상도 모델은 별도 연구 결과로 보관하고, 사이트에는 이 모델에서 생성한 표시용 데이터를 배포한다.</p></section>
</main></body></html>'''


def build(model_dir: Path, volume_dir: Path) -> Path:
    report = load_json(model_dir / "validation.json")
    protocol = load_json(model_dir / "training_protocol.json")
    export = load_json(volume_dir / "export_audit.json")
    manifest = load_json(volume_dir / "manifest.json")
    with (model_dir / "field_metrics.csv").open(encoding="utf-8", newline="") as stream:
        fields = list(csv.DictReader(stream))
    html = render(report, export, manifest, protocol, fields)
    path = volume_dir / "validation.html"
    path.write_text(html, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE / "results/model")
    parser.add_argument("--volume", type=Path, required=True)
    arguments = parser.parse_args()
    print(build(arguments.model, arguments.volume))
