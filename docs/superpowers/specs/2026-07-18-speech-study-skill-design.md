# speech-study 실험 관리 스킬 & 레포 refresh — 설계

- 작성일: 2026-07-18
- 상태: 승인 대기

## 배경

`speech-study`는 Speech AI 논문/주제를 hands-on으로 실습하는 개인 저장소.
기존의 단계형 커리큘럼(`stage1~5`)은 방향이 바뀌어 폐기하고(git 커밋에는 잔존),
**논문 중심 + 주제 경로 하이브리드** 구조로 새로 출발한다.

실습 환경은 두 대:
- **home** — Windows(WSL2) · RTX 3060 Ti **8GB VRAM** · CUDA
- **work** — MacBook Pro M5 **32GB unified** · MPS

목표: 폴더만 열면(또는 미래의 Claude가) 실험을 이어받을 수 있도록 일관된
구조·로그·인덱스를 유지하고, 이를 `/speech-study` 스킬로 반자동화한다.

## 확정된 결정

1. **조직 단위** = 실험 폴더(대개 논문 1개). 그 위를 주제 폴더로 묶는 하이브리드.
2. **venv** = 실험 폴더마다 분리. **HF 캐시(`~/.cache/huggingface`)는 공유**.
3. **스킬 위치** = 레포 안 `.claude/skills/speech-study/`.
4. **refresh** = 완전 새 출발. 옛 stage 구조는 되살리지 않고 빈 구조 + 문서만 커밋.
5. **공유 자산** = 폴더별. 대용량은 `download.py`만 커밋, 실제 파일은 git 제외.
6. **재현성** = 엄격하지 않음. 가중치·대용량은 git에서 제외.

## 레포 구조

```
speech-study/
├── README.md              # 저장소 개요 + 스킬 사용법 + 환경(home/work) 설명
├── INDEX.md               # 실험 레지스트리 (스킬이 자동 유지)
├── .gitignore             # outputs/, *.bin, *.safetensors, *.wav, .venv/, __pycache__ 등
├── docs/superpowers/specs/  # 설계 문서
├── shared/
│   └── env.py             # device 자동감지(cuda→mps→cpu) + 공통 헬퍼 (환경 무관)
│
├── <topic>/                       # 주제 폴더 (kebab-case: audio-tokenization, asr, tts, ...)
│   └── <paper-or-idea>/           # 실험 단위 (논문: encodec / 아이디어: _idea-xxx)
│       ├── LOG.md                 # 실험 로그 (프론트매터 + 본문)
│       ├── requirements.txt       # 이 실험 전용 의존성 (torch 제외 — setup이 설치)
│       ├── download.py            # 대용량 자산 다운로드 스크립트 (자산 자체는 ignore)
│       ├── run.py …               # 실험 코드
│       ├── .venv/                 # 실험 전용 (gitignore)
│       └── outputs/               # 결과물/가중치/오디오 (gitignore)
│
└── sandbox/               # 주제가 애매한 순수 아이디어 실험
```

- 논문 폴더명: 짧은 kebab-case 식별자(`encodec`, `hubert`, `moshi`) — 노션 논문 리뷰 허브와 1:1.
- 아이디어 실험: 관련 주제 안에 `_idea-<name>`, 애매하면 `sandbox/`.

## 로그 템플릿 (`LOG.md`)

프론트매터를 INDEX 생성의 단일 소스로 사용.

```markdown
---
title: <실험 제목>
topic: <주제 폴더명>
paper: <논문 식별자>        # 아이디어면 idea: <이름>
status: planned            # planned | wip | done | parked
env_tested: []             # [home], [work], [home, work]
created: YYYY-MM-DD
updated: YYYY-MM-DD
links:
  paper: <arxiv 등>
  notion: <리뷰 페이지>
---

## 🎯 목표
왜 이 실험을 하는가, 무엇을 확인하려는가

## ⚙️ 실행 방법
​```bash
/speech-study setup home        # 또는 work
python run.py
​```

## 📊 결과 / 관찰

## 🧱 막힌 점 / TODO

## 📝 메모
- 관련 실험 [[topic/paper]]
```

## 인덱스 컨벤션 (`INDEX.md`)

모든 `LOG.md` 프론트매터를 스캔해 표 재생성.

| 주제 | 실험 | 유형 | 상태 | 검증환경 | 갱신일 |
|------|------|------|------|----------|--------|

- 상태 이모지: 🌱 planned · 🚧 wip · ✅ done · 🧊 parked
- 유형: 📄 논문 · 💡 아이디어

## 환경 처리

`shared/env.py` — 코드 레벨 device 자동 감지:
```python
def get_device():  # "cuda" → "mps" → "cpu" 순으로 감지
```

`/speech-study setup [home|work]` — 머신별 부트스트랩:

| 인자 | 머신 | device | torch | 비고 |
|------|------|--------|-------|------|
| home | Windows WSL2 · 3060 Ti 8GB | CUDA | cu121 (2.5.1) | `transformers==4.46.3` 핀(CVE 가드) |
| work | MacBook Pro M5 32GB | MPS | mac 기본 빌드 | 32GB로 더 큰 모델 가능, 핀 완화 가능 |

- 인자 생략 시 `platform.system()` + `torch.cuda.is_available()`로 자동 추정 후 확인.
- 흐름: 실험 폴더에 venv 생성 → torch를 환경에 맞게 설치 → `pip install -r requirements.txt` → device 검증.
- HF 캐시는 기본 위치(`~/.cache/huggingface`)를 공유하므로 venv가 분리돼도 모델 재다운로드 없음.

## `/speech-study` 스킬 명령

1. `setup [home|work]` — 현재(또는 지정) 실험의 머신 부트스트랩 + device 검증.
2. `new <topic>/<name>` — 실험 폴더 스캐폴딩(LOG.md·requirements·download.py·outputs/) + INDEX 갱신.
3. `index` — 프론트매터 스캔해 INDEX.md 재생성.
4. `log` — 지정 실험 LOG.md에 날짜별 관찰 append + `updated` 갱신.

## 산출물

- `.claude/skills/speech-study/SKILL.md` (+ 필요 시 헬퍼 스크립트)
- `README.md`, `INDEX.md`(빈 상태), `.gitignore`, `shared/env.py`
- 옛 stage 구조 삭제 커밋 정리

## 비목표 (YAGNI)

- 자동 하이퍼파라미터 추적/실험 매트릭스
- CI, 테스트 스위트
- 엄격한 재현성(seed 고정 강제, lockfile)
