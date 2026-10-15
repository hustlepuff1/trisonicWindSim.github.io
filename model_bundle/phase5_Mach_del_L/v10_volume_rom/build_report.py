#!/usr/bin/env python3
"""Build a readable static validation page from measured 3D results."""
import argparse
import csv
from html import escape
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(model_dir, volume_dir):
    report = json.loads((model_dir / 'validation.json').read_text())
    export = json.loads((volume_dir / 'export_audit.json').read_text())
    with (model_dir / 'field_metrics.csv').open() as f:
        held = [r for r in csv.DictReader(f) if r['scheme'] == 'independent_retrospective_heldout']
    rows = ''
    for row in held:
        name = {'M175': 'M1.75', 'M325': 'M3.25'}[row['case_id']]
        for key, region in [('full_volume', '전체 체적'), ('upstream_x_le_0', '상류 · 기준 x ≤ 0'),
                            ('downstream_x_gt_0', '하류 · 기준 x > 0'), ('x_band_minus_0p5_to_0p5', '시험부 주변 · 기준 |x| ≤ 0.5 m')]:
            rows += f'<tr><th>{name}</th><td>{region}</td><td>{float(row[key+"_weighted_rmse_mach"]):.5f}</td><td>{float(row[key+"_relative_weighted_L2_percent"]):.2f}%</td></tr>'
    negative = '; '.join(f'{r["case_id"]}: 최솟값 {r["minimum_predicted_mach"]:.5f}, 음수 {r["negative_prediction_cell_count"]:,}셀' for r in report['heldout_native_cell_metrics'])
    loco = next(r for r in report['validation_summary'] if r['scheme'] == 'leave_one_condition_out' and r['subset'] == 'all')
    geometry_rows = ''.join(f'<tr><th>{r["case_id"]}</th><td>{1000*r["node_displacement_max_m"]:.1f} mm</td><td>{escape(" / ".join(r["bracketing_cases"]))}</td></tr>' for r in export['heldout_geometry_interpolation'])
    geometry_file = HERE / 'results/geometry_validation.json'
    geometry_note = ''
    if geometry_file.exists():
        wall = json.loads(geometry_file.read_text())
        detail = '; '.join(f'{r["case_id"]}: 노즐 {1000*r["all_span_tracks"]["inlet_and_nozzle"]["max_abs_height_error_m"]:.2f} mm, 디퓨저 {1000*r["all_span_tracks"]["diffuser"]["max_abs_height_error_m"]:.1f} mm' for r in wall['cases'])
        geometry_note = '<p>같은 물리 x에서 비교한 실제 상부 벽의 최대 높이 오차는 ' + detail + '이다. 큰 하류 차이는 실제 외벽 형상에서도 확인된다. <a href="geometry_validation.json">상부 벽 검사 원자료</a></p>'
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>전체 체적 3D Mach 모델 검증</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f7;color:#193747;font:15px/1.7 system-ui,sans-serif}}main{{max-width:1000px;margin:auto;padding:36px 22px}}a{{color:#206da1}}h1{{font-size:30px;line-height:1.35}}h2{{font-size:20px;margin-top:0}}.card{{background:white;padding:24px;border:1px solid #d4e0e5;border-radius:12px;margin:20px 0}}.notice{{border-left:5px solid #c58a27}}.small{{font-size:13px;color:#55707d}}.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:10px 12px;border-bottom:1px solid #dfe8ec;text-align:left;white-space:nowrap}}thead{{background:#edf4f7}}.numbers{{display:flex;gap:24px;flex-wrap:wrap}}.numbers strong{{font-size:27px;display:block}}.numbers span{{font-size:13px;color:#55707d}}code{{font-size:13px}}@media(max-width:600px){{main{{padding:20px 12px}}.card{{padding:17px}}h1{{font-size:25px}}}}
</style><main><a href="../?view=3d">← 3D 시뮬레이터</a><h1>전체 체적 3D 마하수 모델 검증</h1>
<p>27개 정상 CFD의 전체 체적을 POD–GPR로 학습한 첫 3D 확장판이다. 기존 2D 모델과 제어 계산은 유지했다.</p>
<section class="card notice"><h2>전체 학습 완료 · 하류 예측 개선 필요</h2><p>새 운전 조건의 상류 예측보다 하류 디퓨저의 오차가 크다. 아래 독립 조건 비교와 형상 보간 오차를 함께 확인해야 한다. 이 결과를 완성된 실험 디지털 트윈이나 비정상 유동 검증으로 해석하지 않는다.</p></section>
<section class="card"><div class="numbers"><div><strong>27</strong><span>학습 CFD 조건</span></div><div><strong>3,533,741</strong><span>각 조건의 전체 ¼ 체적 셀</span></div><div><strong>{report['rank']}</strong><span>체적 POD 모드</span></div><div><strong>{100*report['achieved_energy_fraction']:.4f}%</strong><span>학습장 변동 에너지</span></div></div><p class="small">에너지 보존율은 새로운 조건의 예측 정확도가 아니다. 대칭 복원은 표시 단계에만 적용한다.</p></section>
<section class="card"><h2>학습에서 제외한 두 CFD 조건</h2><p>M1.75와 M3.25는 이 3D 모델의 학습·차수 선택에서 제외했다. 이미 보유한 CFD로 수행한 사후 독립 조건 평가이며, 과거 2D의 사전 고정 예측과 구분한다.</p><div class="table"><table><thead><tr><th>조건</th><th>영역</th><th>RMSE [Mach]</th><th>상대 L₂ 오차</th></tr></thead><tbody>{rows}</tbody></table></div><p class="small">w는 정밀한 M17 기준 형상의 양의 셀 체적이다. RMSE = √(Σw·오차²/Σw), 상대 L₂ = 100√(Σw·오차²/Σw·CFD²). 영역도 M17 기준 좌표에서 정의한다. 각 변형 조건 자체의 체적 적분과는 다르며, 이 표의 장 오차는 별도 형상 보간 오차를 포함하지 않는다.</p></section>
<section class="card"><h2>조건 단위 교차검증</h2><p>27회 LOCO에서 매번 조건 하나 전체를 제외하고 평균장·POD 기저·차수·정규화·GP를 다시 학습했다. 전체 체적의 조건 간 제곱평균 RMSE는 {loco['full_volume_weighted_rmse_mach']:.5f} Mach, 조건별 상대 L₂ 오차 평균은 {loco['mean_full_volume_relative_weighted_L2_percent']:.2f}%였다. 제외 조건 23개는 해당 fold의 학습 입력 볼록껍질 밖에 있었다.</p><p>모든 최종 모델 및 교차검증 GP 최적화가 수렴했다. 공간 셀을 무작위로 나누는 검증은 사용하지 않았다.</p></section>
<section class="card"><h2>형상 보간은 별도로 평가</h2><p>표시 형상은 관측된 실제 3D CFD 절점을 L에 따라 보간한다. 두 제외 조건의 전체 대응 절점에서 아래 최대 위치 차이가 남았다. 이 값에는 내부 격자 배치 차이도 포함되며, CFD 마하수 오차와 같은 지표가 아니다.</p><div class="table"><table><thead><tr><th>조건</th><th>최대 대응 절점 차이</th><th>보간에 사용한 두 형상</th></tr></thead><tbody>{geometry_rows}</tbody></table></div>{geometry_note}<p>하류 형상 차이를 추가 입력으로 다루거나 형상 생성 규칙을 맞추는 작업이 후속 개선에 필요하다. 현재 L·P₀만으로 임의의 디퓨저 형상까지 검증한 것은 아니다.</p></section>
<section class="card"><h2>표시와 수치 범위</h2><p>학습에는 모든 셀을 사용하고 웹 표시만 {export['display_node_count']:,}개 절점으로 줄였다. 표시 절점은 인접 셀 평균이며 실제 3D 격자 좌표를 사용한다. 횡단면은 격자 좌표면이므로 화면의 x≈ 값과 실제 x 범위를 함께 표시한다.</p><p>GPR에는 Mach ≥ 0 제약이 없다. 독립 조건의 원래 예측값 검사: {negative}. 오차 계산에서 이 값을 잘라내지 않았다. 3D 색범위는 0–{export['display_color_range'][1]:g}이며 범위 밖 값은 끝 색으로 표시한다. 2D 색범위와 다를 수 있다.</p><p>브라우저 경로 조회 오차는 구간 중점 {export['lookup_midpoint_count']:,}개에서 직접 GP와 비교했다. 최대 기준 체적 RMSE는 {export['lookup_max_reference_volume_rmse_mach']:.2e} Mach, 최대 표시 절점 차이는 {export['lookup_max_display_node_error_mach']:.2e} Mach였다. 이는 모델 자체의 CFD 예측 오차와 별개의 구현 검사다.</p></section>
<section class="card"><h2>재현 자료</h2><p><a href="validation.json">수치 검증 JSON</a> · <a href="field_metrics.csv">조건·영역별 CSV</a> · <a href="export_audit.json">표시·형상 보간 검사</a> · <a href="../model_bundle/phase5_Mach_del_L/v10_volume_rom/README.md">학습 절차</a></p><p class="small">원본 29개 CFD의 전체 SHA-256과 격자 일대일 대응을 확인했다. 기존 2D에서 보관한 1,405,874개 원본 셀 값과도 대조했다. 3D 모델은 정상 CFD 기반 준정상 예측이며, 중앙 Mach 제어 피드백은 기존 2D 단면 모델을 사용한다.</p></section></main></html>'''
    (volume_dir / 'validation.html').write_text(html)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=HERE / 'results/model')
    p.add_argument('--volume', type=Path, required=True)
    a = p.parse_args()
    build(a.model, a.volume)
