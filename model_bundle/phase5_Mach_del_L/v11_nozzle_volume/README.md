# 입구–x=0 노즐 체적 마하수 POD–GPR

사용자가 지정한 입구부터 x=0까지의 1/4 노즐 체적만 사용해 새 모델을 학습한다. 하류 디퓨저는 학습과 표시 영역에서 제외한다. 기존 v10 전체 영역 모델의 그림이나 기저를 잘라 사용하는 방식이 아니라, 선택한 CFD 체적 데이터에서 평균장, POD 기저·차수와 모든 GPR을 다시 학습한다. 기존 2D 모델과 v10 연구 결과는 보존한다.

정확한 x=0 절단, 보존적 병합과 캐시 구성은 [domain_readme.txt](domain_readme.txt)에 설명한다. 재현에는 외부에 보관한 감사 완료 v10 전체 해상도 캐시를 `../v10_volume_rom/results/cache/`에 복원해야 한다. 이 캐시에는 29개 조건의 원래 격자·정합 마하수와 체적 감사 결과가 필요하다. 웹 내보내기는 기존 v09에서 생성한 `../v09_continuous_control/results/simulator_payload.json`의 운전 경로를 사용한다.

이 README가 있는 디렉터리에서 다음 순서로 실행한다.

```bash
python3 prepare_domain.py --source-cache ../v10_volume_rom/results/cache
python3 verify_domain.py
OPENBLAS_NUM_THREADS=4 python3 train_nozzle.py --cache results/cache --output results/model
python3 export_nozzle.py --cache results/cache --model results/model \
  --payload ../v09_continuous_control/results/simulator_payload.json --output results/web
python3 build_report.py --model results/model --volume results/web
python3 -m unittest test_nozzle.py
```

새 전체 해상도 학습 결과는 `results/model/`, 웹용 표시 자료와 검증 페이지 `validation.html`은 `results/web/`에 생성된다. 웹용 출력 디렉터리는 `export_nozzle.py --output`과 `build_report.py --volume`에 같은 경로를 지정한다.

공통 논리 격자로 정합된 27개 학습 CFD와 M1.75·M3.25 독립 조건 두 개를 사용한다. M17 기준 형상의 양의 셀 체적을 고정 가중치로 사용한다. 선택 영역의 정확한 셀 대응과 x=0 경계 처리, 원본 파일 SHA-256은 `results/cache/extraction_manifest.json`에 기록한다. POD 에너지 목표는 0.9999, 최대 차수는 26이고 독립 평가 결과로 차수나 하이퍼파라미터를 조정하지 않는다.

첫 286개 축방향 층의 원래 셀 1,441,726개는 보존한다. 끝부분은 원래 286·287번 층에서 x=0 상류에 해당하는 부분만 보존적으로 합쳐 5,041개 마지막 층 셀을 만든다. 따라서 총 1,446,767개는 **등록 체적 셀**이며 모두 원래 셀의 단순 순열이라고 표현하지 않는다. 끝단은 정확히 x=0이다. 영역 변경은 사용자가 새 학습 전에 지정한 표시·학습 목적이며, 독립 조건 오차를 보고 POD 차수나 GPR을 조정한 것이 아니다.

마지막 층의 병합은 조각별 일정한 원래 셀 마하수의 체적 적분을 보존한다. 이는 재표본화의 보존 성질이며, POD–GPR에 질량·운동량 보존식을 강제했다는 의미가 아니다.

각 조건 전체를 제외하는 27회 LOCO 검증에서 평균장·기저·차수·정규화·GPR을 다시 학습한다. 평가 영역은 전체 노즐, x<−0.5 m, −0.5≤x≤0 m이다. 기준 좌표는 M17 형상이며 오차는 해당 기준 형상의 체적 내적이다. 다른 조건 자체의 물리 체적 오차나 형상 보간 오차를 측정한 것으로 표현하지 않는다. v10 전체 영역과 v11 노즐 영역의 오차는 평가 영역과 정규화 분모가 달라 직접적인 개선율 계산에 사용하지 않는다.

`results/model/`에 새 평균장·기저·GPR, 조건별 평가와 한국어 `report.txt`를 저장한다. 전체 해상도 모델은 로컬에 보관하고, 웹에는 이 모델에서 추출한 표시용 데이터만 배포한다. Gram 재사용은 실제 마하수·체적·좌표·조건 파일과 생성 코드의 SHA-256이 일치할 때만 허용한다. 정상 CFD 기반 준정상 모델이며 비정상 유동 학습이나 실험 동기화를 의미하지 않는다.
