# speech-study

Speech AI 논문·아이디어를 hands-on으로 실습하는 저장소.
LLM 엔지니어가 오디오 기초 → speech understanding/generation → full-duplex로
올라가며, **논문 하나 = 실험 폴더 하나**를 원칙으로 그때그때 폴더를 만들어 실험한다.

전체 실험 목록은 [INDEX.md](INDEX.md) 참고.

## 구조

```
study/<topic>/<paper-or-idea>/   # 논문·메커니즘 해부 (예: recognition/whisper)
train/<topic>/<name>/            # 학습·전처리 실무 실습 (예: recognition/whisper-finetune-ko)
├── LOG.md                 # 목표·실행법·결과·막힌 점 (INDEX의 단일 소스)
├── requirements.txt       # torch 제외 의존성
├── download.py            # 대용량 자산 다운로드 (자산 자체는 git 제외)
├── run.py                 # 실험 코드
├── .venv/                 # 실험 전용 (git 제외)
└── outputs/               # 결과물·가중치 (git 제외)

shared/env.py              # device 자동감지 (cuda→mps→cpu)
```

- **venv는 실험마다 분리**해 의존성 충돌을 격리. **HF 캐시는 공유**되어 모델은 한 번만 받는다.
- 가중치·대용량·outputs는 git에 넣지 않는다(재현성은 엄격하지 않음).

## 두 실습 환경

| 프로필 | 머신 | device | torch |
|--------|------|--------|-------|
| `home` | Windows(WSL2) · RTX 3060 Ti **8GB** | CUDA | cu121 (2.5.1) |
| `work` | MacBook Pro M5 **32GB** | MPS | mac 기본 빌드 |

## 워크플로 (`/speech-study` 스킬)

이 레포에는 실험 관리를 반자동화하는 Claude Code 스킬이 들어있다
(`.claude/skills/speech-study/`). Claude에게 다음처럼 요청하면 된다:

- **새 실험** — "audio-tokenization에 encodec 실험 만들어줘"
  → `new_experiment.py`가 폴더+템플릿 생성 후 INDEX 갱신
- **환경 셋업** — 실험 폴더에서 "setup home" / "setup work"
  → `setup_env.py`가 venv 생성·torch 설치·device 검증
- **인덱스 갱신** — "INDEX 다시 만들어" → `build_index.py`
- **실험 로그** — "오늘 결과 기록해줘" → 해당 `LOG.md`에 관찰 추가

스크립트를 직접 쓰려면:

```bash
# 새 실험
python .claude/skills/speech-study/scripts/new_experiment.py \
    train/recognition/whisper-finetune-ko --title "Whisper 한국어 파인튜닝"

# 환경 셋업 (실험 폴더 안에서)
cd train/recognition/whisper-finetune-ko
python ../../../.claude/skills/speech-study/scripts/setup_env.py work

# 인덱스 갱신
python .claude/skills/speech-study/scripts/build_index.py
```
