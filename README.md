# 가변노즐 연속 제어 시뮬레이터 · 2D / 3D Mach

- 사이트: https://hustlepuff1.github.io/trisonicWindSim.github.io/
- 저장소: https://github.com/hustlepuff1/trisonicWindSim.github.io

GitHub Pages 화면과 그 화면을 재생성하는 학습 완료 모델·데이터·코드를 함께 보관한다.
이 폴더 전체가 하나의 저장소가 되도록 구성했다. 기존 연구 폴더를 함께 올릴 필요가 없다.

화면 위에서 **2D / 3D / 함께**를 선택한다. 기존 2D 모델과 제어기는 유지하고,
전체 CFD 체적의 마하수를 새로 학습한 3D POD-GPR을 같은 현재 L·P₀에 연결했다.
3D에서는 회전·확대, ¼/전체 대칭 복원, 내부 격자 횡단면 이동을 지원한다.

2D 데이터와 화면 코드는 `index.html`에 포함된다. 3D를 처음 선택할 때
`volume/`의 이진 모델 자료를 추가로 불러오므로 GitHub Pages 또는 로컬 HTTP
서버로 열어야 한다. 외부 CDN이나 서버 추론은 사용하지 않는다.
화면 사용법은 [조작 안내](model_bundle/phase5_Mach_del_L/v09_continuous_control/UI_USAGE.md)에 있다.

## 3D 학습과 표시

- 학습: 기존 27개 운전 조건의 전체 ¼ CFD 마하수 **3,533,741셀/조건**. 입구부터 하류 디퓨저까지 포함한다.
- 격자 대응: 케이스마다 다른 셀 번호를 원래 701×71×71 체적 격자 주소로 일대일 재정렬했다. 학습용 셀을 삭제하거나 2D 단면을 돌출시키지 않았다.
- POD: 정밀도가 높은 M17의 양의 물리 셀 체적을 고정 기준 가중치로 사용한다. 계수는 기존 [L, ln P₀] Matérn 5/2 GPR로 학습한다.
- 검증: 운전 조건 단위 LOCO와 학습에서 제외한 M1.75·M3.25의 전체 체적 비교. 새 3D 모델의 검증이며 과거 2D 사전 고정 예측을 수정하지 않는다.
- 웹 표시: 전체 체적 모델의 평균·모드를 표시 격자에 샘플링한다. 인접 셀 평균으로 표시 절점 값을 만들고 실제 3D CFD 좌표를 L에 따라 보간한다. 학습 해상도와 표시 해상도는 다르다.
- 대칭: ¼ 영역만 학습하고 y=0, z=0에서 Mach 스칼라와 형상을 반사한다. 비대칭 유동이나 새로운 비정상 CFD를 예측하는 기능은 아니다.
- 제어: 중앙 100×100 mm Mach 피드백은 기존 2D 단면 모델을 유지한다. 3D 화면은 같은 현재 운전 조건의 별도 체적 예측이다.

수치 검증 결과는 [3D 검증 보고서](volume/validation.html), 표시·보간 검사 결과는
[내보내기 검사](volume/export_audit.json)에 있다. 오차의 체적 가중치는 M17 기준
격자에 고정되며 각 케이스의 현재 체적 적분과 구별한다.

## 포함 파일

| 경로 | 내용 |
|---|---|
| `index.html` | 기존 2D와 새 3D를 선택하는 배포 화면 |
| `volume/` | 3D 표시용 평균·POD 모드·관측 형상, 운전 경로 계수와 검증 기록 |
| `volume_viewer.js`, `volume_viewer.css` | 외부 라이브러리 없는 3D 표시 코드와 스타일 |
| `model_bundle/phase5_Mach_del_L/v10_volume_rom/` | 전체 체적 추출·등록·학습·검증·내보내기 코드와 소형 모델 기록 |
| `frozen_validation/M1p75/` | 사전 고정 예측 JSON, 예측 유동장 NPZ, 형상·입력 CSV, 해시 목록 |
| `model_bundle/phase5_Mach_del_L/v07_flowfield_rom/results/` | 종단면 모델 PKL, 형상·격자·스냅숏 NPZ와 모델 설명 |
| `model_bundle/phase5_Mach_del_L/v08_interactive_control/results/crossplane/` | 시험부 단면 모델 PKL, 형상·격자·스냅숏 NPZ와 모델 설명 |
| `model_bundle/phase5_Mach_del_L/v09_continuous_control/` | 연속 운전 경로 생성, HTML 빌더, JS 제어기·화면 코드, HTML 템플릿, 저장된 payload·운전표 |
| `model_bundle/`의 나머지 Python·자료 | 위 코드가 사용하는 GP 계산, 기하 적분, 파일 처리 및 학습 입력 출처 |
| `rebuild.py`, `requirements.txt` | 재생성 실행 파일과 Python 패키지 버전 |
| `bundle_manifest.json` | 최초 2D 배포 당시 복사본의 출처·SHA-256·용량 기록 |
| `release_manifest.json` | 현재 2D/3D 배포 파일의 SHA-256·용량 |
| `verification.json` | 최초 2D 배포의 독립 재생성 검사 기록 |
| `.nojekyll`, `.gitignore` | 정적 배포 및 로컬 생성물 제외 설정 |

모델 파일은 다음 두 개다.

- [종단면 field_model.pkl](model_bundle/phase5_Mach_del_L/v07_flowfield_rom/results/field_model.pkl)
- [시험부 단면 field_model.pkl](model_bundle/phase5_Mach_del_L/v08_interactive_control/results/crossplane/field_model.pkl)

기존 2D 모델·운전 payload·고정 예측의 원본 바이트는 보존했다. `model_bundle` 내부의 폴더 구조를
유지하여 Python import가 이 저장소 안에서 해결된다. 최초 bundle manifest에는 전체 연구의
다른 출력들도 기재되어 있으며, 이번 저장소에 포함한 파일 목록은 `bundle_manifest.json`이다.

## GitHub Pages에 올리기

1. 이 폴더 **안의 파일과 하위 폴더**를 저장소의 `gh-pages` 브랜치 최상위에 올린다. `index.html`이 저장소 최상위에 있어야 한다.
2. 저장소의 **Settings → Pages → Build and deployment**에서 Source를 **Deploy from a branch**로 선택한다.
3. **gh-pages** 브랜치와 **/(root)**를 선택하고 **Save**를 누른다.
4. 배포가 끝나면 Pages 화면에 표시되는 주소를 연다.

설정 방법의 기준은 [GitHub 공식 배포 안내](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)이다.
파일 참조가 상대 경로여서 저장소 이름이 붙는 사이트 하위 경로에서도 사용할 수 있다.

배포 후 수정 사항은 이 폴더에서 커밋하고 푸시한다. `gh-pages` 브랜치에
새 커밋이 올라가면 GitHub Pages가 사이트를 다시 배포한다.

```bash
git add .
git commit -m "Update nozzle simulator"
git push origin gh-pages
```

## 로컬에서 확인

이 폴더에서 아래 명령을 실행한 뒤 `http://localhost:8000/`을 연다.

```bash
python3 -m http.server 8000
```

## HTML 다시 만들기

아래 명령은 이 저장소의 최상위에서 실행한다. Python 3.11 이상을 사용하며,
검사는 Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1에서 수행했다.

저장된 payload와 화면 코드를 묶고 3D 자료의 해시를 확인해 복사할 때는 표준 Python만 필요하다.

```bash
python3 rebuild.py
```

PKL 두 개와 등록된 형상·격자로 801개 운전점을 다시 계산하고 HTML을 만들려면:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 rebuild.py --from-models
```

결과는 `build/index.html`과 `build/volume/`에 저장된다. 모델 재생성에서는 `build/` 아래에
payload, 운전표, 조회 오차 검사도 생성한다. 두 방식 모두 `build/frozen_validation/`
아래에 기존 고정 예측을 함께 복사하여 화면 링크를 유지한다. 이미 있는 고정 예측을
다른 내용으로 덮어쓰지 않는다. 저장소 최상위의 배포용 `index.html`도 자동 교체하지 않는다.

생성물을 확인한 뒤 `build/index.html`을 최상위 `index.html`로 복사하면 배포 화면을 갱신할 수 있다.
`build/`는 `.gitignore`에 포함되어 있다.

`rebuild.py`는 학습 완료 자료로 배포 화면을 재생성한다. `--from-models`는 기존
2D 운전 경로를 재계산하며 3D 모델을 재학습하지 않는다. 3D 재학습과 전체 해상도
복원은 `v10_volume_rom/README.md`의 연구 절차를 따른다. 약 27 GB의 원본 CFD와
전체 해상도 로컬 캐시는 이 정적 사이트 저장소에 중복 업로드하지 않는다.
브라우저용 3D 자료는 이 저장소 안에 완전히 포함된다.

## 원본과 보존 범위

다음 원본에서 시뮬레이터와 모델 재생성에 필요한 파일을 복사했다.

```text
01_Masters_degree/framework/
```

기존 2D PKL·NPZ·고정 예측과 제어 계산은 수정하지 않았다. 3D 버전은 화면 코드와
HTML 빌더를 확장하고 새 모델·자료를 추가했다. 예측 JSON의 과거 절대 경로는
출처 기록이며, 재생성할 때 그 경로의 파일을 읽지 않는다. 고정 예측 JSON은 생성 당시의
상태를 그대로 유지한다.
