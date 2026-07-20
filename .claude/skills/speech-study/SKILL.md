---
name: speech-study
description: >-
  speech-study 저장소에서 Speech AI 논문·아이디어 실험을 관리하는 스킬. 새 실험 폴더
  스캐폴딩(new), 머신별 환경 셋업(setup home=Windows·CUDA / work=Mac·MPS), 실험
  레지스트리 갱신(index), 실험 로그 기록(log)을 담당한다. 사용자가 이 레포에서
  "새 실험/논문 실습 시작", "encodec/hubert/moshi 같은 논문 폴더 만들어줘",
  "환경 세팅 / setup home / setup work", "인덱스 갱신", "실험 로그 남겨줘" 같은
  것을 요청할 때 반드시 사용. speech / 오디오 / ASR / TTS / codec / speech-LM /
  full-duplex 실험을 이 레포에서 시작·정리하려 할 때도 사용.
---

# speech-study 실험 관리

Speech AI 논문/아이디어를 hands-on으로 실습하는 저장소를 일관되게 관리한다.
핵심 목표는 **미래의 나(또는 Claude)가 폴더만 열고 실험을 이어받을 수 있게** 하는 것.
그래서 모든 실험은 같은 골격(로그·의존성·다운로드 스크립트·outputs)을 갖고,
루트 `INDEX.md`가 전체를 한눈에 보여준다.

## 저장소 컨벤션 (설계 근거)

```
speech-study/
├── INDEX.md                       # 실험 레지스트리 (build_index.py가 자동 생성)
├── shared/env.py                  # device 자동감지 헬퍼 (cuda→mps→cpu)
├── study/                         # 논문·메커니즘 해부
│   └── <topic>/<paper-or-idea>/   # 실험 단위
│       ├── LOG.md                 # 프론트매터 + 본문 (INDEX의 단일 소스)
│       ├── requirements.txt       # torch 제외 의존성
│       ├── download.py            # 대용량 자산 다운로드 (자산은 git 제외)
│       ├── run.py                 # 실험 코드
│       ├── .venv/  · outputs/     # git 제외
└── train/                         # 학습·전처리 실무 실습
    └── <topic>/<name>/            # 같은 골격
```

- **조직 단위 = 실험 폴더**(대개 논문 1개), 그 위를 **주제 폴더**로 묶는다.
- **최상위 구분**: `study/`(해부) · `train/`(학습 실무). 경로는 항상 `<section>/<topic>/<name>` 3단계.
- **논문 폴더명**은 짧은 kebab-case(`encodec`, `hubert`, `moshi`) — 노션 논문 리뷰 허브와 1:1.
- **아이디어 실험**은 관련 주제 안에 `_idea-<name>`, 주제가 애매하면 `sandbox/`.
- **venv는 실험마다 분리**해서 의존성 충돌(특히 `transformers` 핀)을 격리한다.
  단, **HF 캐시(`~/.cache/huggingface`)는 기본 위치를 공유**하므로 모델은 한 번만 받는다.
- **재현성은 엄격하지 않다.** 가중치·대용량·outputs는 git에 넣지 않는다.

## 두 실습 환경

| 프로필 | 머신 | device | torch | 메모 |
|--------|------|--------|-------|------|
| `home` | Windows(WSL2) · RTX 3060 Ti **8GB** | CUDA | cu121 (2.5.1 고정) | 8GB → 경량 모델·공식 데모. `transformers==4.46.3` 핀 |
| `work` | MacBook Pro M5 **32GB** | MPS | mac 기본 빌드 | 32GB → 더 큰 모델 가능. 핀 완화 여지 |

`home`의 `transformers==4.46.3` 핀은, transformers 4.50+/5.x가 CVE-2025-32434 가드로
torch>=2.6에서만 `.bin` 체크포인트 로드를 허용하는데 설치 torch가 cu121(2.5.1)이고
HuBERT 등 일부 모델은 `.bin`만 있기 때문. 코드에서 device는 항상 `shared/env.py`의
`get_device()`로 얻어 두 환경 모두에서 그대로 돌아가게 한다.

## 명령 디스패치

사용자 요청을 아래 4개 중 하나로 매핑해 해당 스크립트를 실행한다. 스크립트 경로는
이 스킬의 `scripts/` 아래. 스크립트는 표준 라이브러리만 쓰므로 시스템 파이썬으로 실행한다.
레포 루트는 스크립트가 `git rev-parse --show-toplevel`로 자동 탐지한다(실패 시 cwd).

### 1) new — 새 실험 스캐폴딩

트리거: "새 실험", "encodec 폴더 만들어줘", "논문 실습 시작".

```bash
python .claude/skills/speech-study/scripts/new_experiment.py <section>/<topic>/<name> \
  --title "<사람이 읽는 제목>" \
  [--idea] [--paper-link <url>] [--notion-link <url>]
```

- `<section>/<topic>/<name>` 예: `train/recognition/whisper-finetune-ko`, `study/audio-tokenization/encodec`.
- 폴더와 `LOG.md`·`requirements.txt`·`download.py`·`run.py`·`outputs/`를 만들고 INDEX를 갱신한다.
- 만든 뒤 사용자에게 다음 할 일을 알려준다: LOG.md 목표 채우기 → `setup` → 코드 작성.

### 2) setup — 머신별 환경 셋업

트리거: "setup home", "환경 세팅", "work 환경 준비".

**반드시 대상 실험 폴더 안(cwd)에서 실행**한다. venv가 실험별로 분리되기 때문.

```bash
cd <section>/<topic>/<name>
python <스킬경로>/scripts/setup_env.py [home|work]
```

- 프로필 생략 시 OS로 자동 추정(Linux→home 후보, Darwin→work). 추정값을 사용자에게 확인.
- `.venv` 생성 → 프로필에 맞는 torch/torchaudio 설치 → `requirements.txt` 설치 →
  `get_device()`로 device를 출력해 검증.
- `uv`가 있으면 uv를, 없으면 표준 `venv`+`pip`를 쓴다.

### 3) index — 레지스트리 재생성

트리거: "인덱스 갱신", "INDEX 다시 만들어".

```bash
python <스킬경로>/scripts/build_index.py
```

모든 `LOG.md` 프론트매터를 스캔해 `INDEX.md` 표를 다시 쓴다. `new`가 자동 호출하지만,
LOG의 status/env_tested를 손으로 바꾼 뒤 수동으로 돌려도 된다.

### 4) log — 실험 로그에 관찰 추가

트리거: "실험 로그 남겨", "오늘 결과 기록".

`log` 전용 스크립트는 없다. 대상 `LOG.md`를 열어 본문 섹션(📊 결과 / 🧱 막힌 점 등)에
날짜와 함께 관찰을 추가하고, 프론트매터의 `updated`를 오늘 날짜로, 필요하면 `status`·
`env_tested`도 갱신한다. 그 뒤 `build_index.py`를 돌려 INDEX를 맞춘다. 오늘 날짜가
불확실하면 사용자에게 묻거나 시스템 날짜를 쓴다.

## 로그 템플릿

`assets/LOG_template.md`가 원본. 프론트매터가 INDEX 생성의 단일 소스이므로 키를
임의로 바꾸지 말 것(`title, topic, paper|idea, status, env_tested, created, updated, links`).
`status`는 `planned|wip|done|parked`, `env_tested`는 `[]`/`[home]`/`[work]`/`[home, work]`.

## 작업 원칙

- 사용자가 "논문 실습 시작"처럼 말하면 먼저 어떤 주제 폴더에 넣을지, 폴더명(kebab-case)을
  확인한 뒤 `new`를 돌린다. 이미 명확하면 바로 실행한다.
- 대용량 자산(가중치·데이터셋)은 코드에 URL/로더만 남기고 `download.py`에 모은다.
  실제 파일은 `.gitignore`가 막으므로 커밋 걱정 없이 받는다.
- 8GB(home)에서 큰 모델을 요구하는 실험은 경량 대체 모델이나 공식 데모로 우회하고,
  그 사실을 LOG.md의 "막힌 점"에 남겨 다음에 참고하게 한다.
