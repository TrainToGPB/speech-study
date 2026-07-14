# Stage 2 — 음성 이해 (ASR · diarization)

> 음성을 텍스트·의미로. cascade의 **앞단**이자 도착점 **SpeakerLM**의 빌드업.

## 목표
- cascade의 ASR 단을 이루는 기술 계보를 안다.
- audio-LLM이 vision-LLM과 **같은 `encoder+adapter+LLM` 패턴**으로 음성을 이해함을 설명.
- diarization+ASR을 따로 돌릴 때의 **error propagation**을 몸으로 이해 → SpeakerLM의 동기.

## 논문 2개 (Notion에서 리뷰)

| # | 논문 | arXiv | 왜 |
|---|---|---|---|
| ① | **Whisper** | 2212.04356 | 실무 표준 ASR. "68만 시간 weak supervision" 데이터 스케일 철학이 포인트 |
| ② | **SALMONN** | 2310.13289 | `speech encoder+adapter+LLM` 패턴의 대표 = vision-LLM의 오디오판 |

배경: Conformer(2005.08100, ASR 인코더 기본기), Qwen2-Audio(2407.10759, 대규모 audio-LLM 레퍼런스).

## 실습 코드 — `Whisper 전사 + pyannote diarization 이어붙이기`

**설계** (`pipeline_asr_diarization.py`로 구현 예정):
1. `faster-whisper`(CTranslate2, int8)로 음성 전사 → 단어별 타임스탬프.
2. `pyannote/speaker-diarization-3.1`로 "누가 언제"(화자 세그먼트).
3. 둘을 타임스탬프로 병합 → "화자 A: ..., 화자 B: ..." 스크립트 생성.
4. **관찰 포인트**: 겹쳐 말하는(overlap) 구간·화자 경계에서 정렬이 어긋나는 걸 확인.
   두 모듈의 오차가 **곱해지는**(error propagation) 지점을 로그로 남긴다.

**셋업 메모** (이 스테이지 전용 venv, 결과는 `stage2_understanding/outputs/`):
```bash
cd stage2_understanding
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# pyannote 3.1은 HuggingFace 토큰 + 모델 약관 동의 필요:
#   huggingface-cli login
#   https://hf.co/pyannote/speaker-diarization-3.1 약관 accept
python ../shared/sample_audio/download_samples.py   # two_speakers.wav 사용
```

**8GB 메모**: faster-whisper `small`/`base` int8 + pyannote 3.1 → 넉넉히 실행 가능.
Qwen2-Audio-7B(SALMONN 계열 직접 실행)는 8GB 초과 → **읽기 + 공식 데모 관찰**.

## 자가 점검
- [ ] CTC / seq2seq / RNN-T의 차이와 스트리밍 적합성을 말할 수 있나
- [ ] audio-LLM이 vision-LLM과 구조적으로 같은 점을 설명할 수 있나
- [ ] diarization과 ASR을 따로 돌릴 때 생기는 문제를 예로 들 수 있나

## 다음
→ [Stage 3 — 음성 생성](../stage3_generation/).
