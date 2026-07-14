# Stage 1 — 오디오 표현 & discrete token

> waveform이 어떻게 **"LLM이 먹을 수 있는 discrete token"**이 되는가. 모든 것의 토대.

## 목표
끝나면 이걸 할 수 있어야 함:
- mel-spectrogram과 discrete audio token의 차이를 설명
- **semantic token**(내용 중심)과 **acoustic token**(복원 중심)을 구분
- frame rate가 왜 latency와 직결되는지 설명

## 논문 2개 (Notion에서 리뷰)

| # | 논문 | arXiv | 왜 |
|---|---|---|---|
| ① | **HuBERT** | 2106.07447 | **semantic token**의 표준. 클러스터 타깃 masked prediction. GLM-4-Voice tokenizer 계보의 뿌리 |
| ② | **EnCodec** | 2210.13438 | **acoustic token**의 대표. RVQ neural codec. VALL-E·Moshi가 사용 |

두 논문은 **짝으로** 읽는다: `semantic(HuBERT)` vs `acoustic(EnCodec)`. 이 대비가 이후 모든
SpeechLM 설계의 첫 갈림길. (배경: wav2vec 2.0, WavLM, SoundStream, DAC는 여유될 때.)

## 실습 코드

먼저 셋업 (이 스테이지 전용 venv):
```bash
cd stage1_audio_tokens
uv venv --python 3.10 .venv          # 이미 있으면 생략
source .venv/bin/activate
uv pip install -r requirements.txt
python ../shared/sample_audio/download_samples.py   # 공통 샘플 오디오
```
결과물은 `stage1_audio_tokens/outputs/` 에 저장됩니다.

순서대로:
```bash
python 01_waveform_mel.py                  # waveform·mel·frame rate 감각
python 02_encodec_tokens.py                # EnCodec: encode→token→decode 복원
python 03_hubert_features.py               # HuBERT: semantic feature→k-means unit
python 04_compare_acoustic_vs_semantic.py  # ★ 핵심: 둘을 나란히 비교
```

출력물은 `../outputs/`에 PNG·WAV로 저장된다. `02`와 `04`의 복원 WAV를 원음과 **들어보며**
"acoustic은 복원되고 semantic은 안 된다"를 귀로 확인하는 게 이 스테이지의 핵심 체험.

### 각 스크립트가 보여주는 것
- **01**: 16kHz waveform → STFT → mel(80). hop=10ms면 frame rate 100Hz. 같은 발화가
  frame rate에 따라 토큰 수가 어떻게 변하는지(100→12.5Hz) = latency 예산.
- **02**: EnCodec은 한 시점을 **여러 codebook 정수**로 표현(RVQ). 디코딩하면 **원음 복원**.
  이 다중 스트림 구조가 Moshi(Stage 5) parallel-stream의 근거.
- **03**: HuBERT는 연속 hidden state → k-means로 **discrete unit**. 내용은 담지만 **복원 불가**.
  중간 레이어(≈9)가 음소/내용에 가깝다.
- **04**: 둘을 한 표·한 그림으로. 복원 가능성·frame rate·표현 형태·소비처 대비.

## 자가 점검
- [ ] mel-spectrogram과 discrete audio token의 차이를 말할 수 있나
- [ ] ASR엔 어느 토큰(semantic), TTS 음질엔 어느 토큰(acoustic)을 쓸지 답할 수 있나
- [ ] frame rate가 왜 latency와 직결되는지 말할 수 있나
- [ ] RVQ에서 codebook을 여러 개 쌓는 이유를 설명할 수 있나

## 트러블슈팅
- **CUDA OOM 없음**: 이 스테이지 모델(EnCodec, HuBERT-base)은 <2GB. 3060 Ti 여유.
- **CPU만 쓰고 싶다**: `requirements.txt`의 `--extra-index-url` 줄 지우고 재설치.
- **HuBERT 다운로드 느림**: 최초 1회 HF에서 ~360MB 받는다. 이후 캐시.
- **k-means 느림**: 3초 클립엔 즉시. 긴 오디오면 `n_clusters`·`n_init` 낮추기.

## 다음
→ [Stage 2 — 음성 이해](../stage2_understanding/). 이 token들 위에서 Whisper가 ASR을 하는 법.
