# 가변노즐 연속 제어 시뮬레이터

- 사이트: https://hustlepuff1.github.io/trisonicWindSim.github.io/
- 저장소: https://github.com/hustlepuff1/trisonicWindSim.github.io

GitHub Pages 화면과 그 화면을 재생성하는 학습 완료 모델·데이터·코드를 함께 보관한다.
이 폴더 전체가 하나의 저장소가 되도록 구성했다. 기존 연구 폴더를 함께 올릴 필요가 없다.

`index.html`을 열면 시뮬레이터가 실행된다. CSS, JavaScript, 유동장 모델의
브라우저용 표현과 운전 데이터가 HTML 안에 포함되어 있다. PKL은 아래 Python
재생성 과정에서 읽는다. 화면 사용법은 [조작 안내](model_bundle/phase5_Mach_del_L/v09_continuous_control/UI_USAGE.md)에 있다.

## 포함 파일

| 경로 | 내용 |
|---|---|
| `index.html` | 원본 시뮬레이터와 내용이 같은 첫 화면 |
| `frozen_validation/M1p75/` | 사전 고정 예측 JSON, 예측 유동장 NPZ, 형상·입력 CSV, 해시 목록 |
| `model_bundle/phase5_Mach_del_L/v07_flowfield_rom/results/` | 종단면 모델 PKL, 형상·격자·스냅숏 NPZ와 모델 설명 |
| `model_bundle/phase5_Mach_del_L/v08_interactive_control/results/crossplane/` | 시험부 단면 모델 PKL, 형상·격자·스냅숏 NPZ와 모델 설명 |
| `model_bundle/phase5_Mach_del_L/v09_continuous_control/` | 연속 운전 경로 생성, HTML 빌더, JS 제어기·화면 코드, HTML 템플릿, 저장된 payload·운전표 |
| `model_bundle/`의 나머지 Python·자료 | 위 코드가 사용하는 GP 계산, 기하 적분, 파일 처리 및 학습 입력 출처 |
| `rebuild.py`, `requirements.txt` | 재생성 실행 파일과 Python 패키지 버전 |
| `bundle_manifest.json` | 복사한 파일의 출처·SHA-256·용량 |
| `verification.json` | 원래 연구 경로 접근을 차단한 재생성 검사 결과 |
| `.nojekyll`, `.gitignore` | 정적 배포 및 로컬 생성물 제외 설정 |

모델 파일은 다음 두 개다.

- [종단면 field_model.pkl](model_bundle/phase5_Mach_del_L/v07_flowfield_rom/results/field_model.pkl)
- [시험부 단면 field_model.pkl](model_bundle/phase5_Mach_del_L/v08_interactive_control/results/crossplane/field_model.pkl)

모델과 데이터는 원본 바이트를 보존했다. `model_bundle` 내부의 원래 폴더 구조도
유지하여 Python import가 이 저장소 안에서 해결된다. 원본 manifest에는 전체 연구의
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

저장된 payload와 화면 코드를 묶어 HTML만 다시 만들 때는 표준 Python만 필요하다.

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

결과는 `build/index.html`에 저장된다. 모델 재생성에서는 `build/` 아래에
payload, 운전표, 조회 오차 검사도 생성한다. 두 방식 모두 `build/frozen_validation/`
아래에 기존 고정 예측을 함께 복사하여 화면 링크를 유지한다. 이미 있는 고정 예측을
다른 내용으로 덮어쓰지 않는다. 저장소 최상위의 배포용 `index.html`도 자동 교체하지 않는다.

생성물을 확인한 뒤 `build/index.html`을 최상위 `index.html`로 복사하면 배포 화면을 갱신할 수 있다.
`build/`는 `.gitignore`에 포함되어 있다.

재생성 범위는 **학습 완료 모델로 운전 경로·유동장 조회 데이터·HTML을 생성하는 과정**이다.
모델 재학습, CFD 원자료 재추출, 새 CFD/FEA 해석은 별도의 연구 자료와 환경을 사용한다.

## 원본과 보존 범위

다음 원본에서 시뮬레이터와 모델 재생성에 필요한 파일을 복사했다.

```text
01_Masters_degree/framework/
```

원본 HTML·PKL·NPZ·고정 예측과 연구 코드는 수정하지 않았다.
한글 사용 안내의 링크만 이 저장소의 경로로 연결했다. 예측 JSON의 과거 절대 경로는
출처 기록이며, 재생성할 때 그 경로의 파일을 읽지 않는다. 고정 예측 JSON은 생성 당시의
상태를 그대로 유지한다.
