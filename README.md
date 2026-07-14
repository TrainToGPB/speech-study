# speech-study — Voice AI 실습 커리큘럼

LLM 엔지니어가 **오디오 기초 → speech understanding/generation → full-duplex**까지
순서대로 올라가는 hands-on 학습 프로젝트.

- 이론/논문 리뷰는 Notion([Voice AI 기술 이해](https://app.notion.com/p/39d74b55ec7c80a2885df06e314d0eb4))에서.
- **코드 실습은 이 저장소에서.**
- 도착점: **SpeakerLM**(이해 축) · **BayLing-Duplex**(대화 축) 두 논문.

전체 커리큘럼(스테이지별 논문 2 + 코드 1, 그리고 왜 골랐는지)은 [`CURRICULUM.md`](./CURRICULUM.md) 참고.

## 이 커리큘럼의 한 줄 요약

> 음성도 결국 **discrete token 시퀀스**다. 텍스트 LLM에서 배운 next-token prediction이
> 이해·생성·대화까지 그대로 확장된다. 큰 흐름은 **모듈 cascade → 통합 end-to-end LLM**.

## 스테이지 지도

| Stage | 주제 | 논문 ① | 논문 ② | 실습 코드 | 로컬 실행(8GB) |
|---|---|---|---|---|---|
| [1](./stage1_audio_tokens/) | 오디오 표현 & discrete token | HuBERT | EnCodec | acoustic vs semantic 토큰 비교 | ✅ 완전 실행 |
| [2](./stage2_understanding/) | 음성 이해 (ASR·diarization) | Whisper | SALMONN | Whisper + pyannote 이어붙이기 | ✅ 실행 가능 |
| [3](./stage3_generation/) | 음성 생성 (TTS) | VALL-E | CosyVoice | CosyVoice2 vs 2단 baseline | ✅ 실행 가능 |
| [4](./stage4_speechlm/) | Speech LLM (통합) | GLM-4-Voice | LLaMA-Omni | Mini-Omni S2S + cascade 대조 | ⚠️ 경량모델만 |
| [5](./stage5_fullduplex/) | Full-duplex (도착점) | SpeakerLM | BayLing-Duplex | Moshi vs turn-based 대조 | ⚠️ 데모 위주 |

**현재 상태**: Stage 1 실습 코드 완성(실행 검증 완료). Stage 2~5는 README(커리큘럼·논문·코드 설계·자가점검)와
실행 명령 가이드가 채워져 있고, 코드는 진행하며 채웁니다.

## 환경

- WSL2 · Python 3.10 · [uv](https://docs.astral.sh/uv/) · NVIDIA RTX 3060 Ti (**8GB VRAM**)
- 8GB 제약: Stage 1~3 실습은 로컬에서 충분. Stage 4~5의 큰 모델(GLM-4-Voice 9B, Moshi 7B,
  Qwen2.5-Omni 7B)은 full inference가 빠듯 → **경량 모델 실습 + 공식 데모 관찰** 조합.

## 셋업 — 스테이지별로 독립

venv와 출력물을 **스테이지마다 분리**합니다. 스테이지끼리 의존성 충돌(특히 torch·transformers
버전)이 잦고, 실습 결과도 섞이지 않게 하기 위함입니다.

```
stageN_.../
├── .venv/        ← 그 스테이지 전용 가상환경
├── requirements.txt
├── outputs/      ← 그 스테이지 실습 결과 (PNG·WAV·npy)
└── *.py
```

각 스테이지 진입 시:
```bash
cd stage1_audio_tokens
uv venv --python 3.10 .venv          # (이미 만들어져 있으면 생략)
source .venv/bin/activate
uv pip install -r requirements.txt   # 그 스테이지 의존성만
```

공통 샘플 오디오는 한 번만 받으면 됩니다(어느 스테이지 venv에서든):
```bash
python ../shared/sample_audio/download_samples.py   # → shared/sample_audio/*.wav
```

그다음 Stage 1 실습:
```bash
python 01_waveform_mel.py
python 02_encodec_tokens.py
python 03_hubert_features.py
python 04_compare_acoustic_vs_semantic.py
# 결과는 stage1_audio_tokens/outputs/ 에 저장됨
```

> 5개 스테이지 venv는 이미 생성돼 있고, **Stage 1만 의존성 설치·실행 검증 완료**입니다.
> Stage 2~5는 해당 스테이지에 진입할 때 `uv pip install -r requirements.txt`로 설치하세요
> (torch를 스테이지마다 받으므로 디스크를 아끼려 미리 깔지 않았습니다).

## 진행 방식 (권장)

1. Notion에서 해당 스테이지의 **논문 2개**를 읽고 리뷰 초안을 채운다.
2. 이 저장소에서 **실습 코드**를 돌리며 논문의 개념을 눈으로 확인한다.
3. 각 스테이지 README 맨 아래 **자가 점검** 질문에 답할 수 있으면 다음 스테이지로.
4. cascade 실무 경험이 있으면 Stage 2·3은 빠르게 통과, Stage 4·5에 시간 투자.
