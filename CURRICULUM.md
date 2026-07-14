# CURRICULUM — 스테이지별 핵심 논문 2 + 실습 코드 1

Notion 로드맵의 자료 DB(43개)를 **스테이지당 논문 2개 + 실습 코드 1개**로 압축한 최단 경로.
선정 기준:

1. Notion에서 `우선순위 = 필독`으로 태깅된 것 우선.
2. 두 **도착점 논문**(SpeakerLM, BayLing-Duplex)으로 이어지는 **계보**를 만드는 것 우선.
3. 실습 코드는 8GB VRAM(RTX 3060 Ti)에서 **직접 돌려 개념을 눈으로 확인**할 수 있는 것 우선.
   큰 모델은 "읽기 + 공식 데모 관찰"로 대체.

> 표기: arXiv ID는 Notion 로드맵 기준. BayLing-Duplex(2506.14528)·SpeakerLM(2508.06372)은
> 도착점 논문이라 원문 확인 후 리뷰 권장.

---

## Stage 1 — 오디오 표현 & discrete token

**한 문장**: waveform이 어떻게 "LLM이 먹을 수 있는 discrete token"이 되는가. 모든 것의 토대.

### 논문 ① HuBERT (arXiv:2106.07447)
- **왜**: **semantic token**의 사실상 표준. 클러스터 타깃으로 masked prediction → 라벨 없이
  "내용/발음" 중심 표현을 학습. GLM-4-Voice tokenizer 계보의 뿌리 = Stage 4·5로 직결.
- **핵심 포인트**: offline k-means로 만든 discrete target → iterative refinement. wav2vec 2.0의
  contrastive 대비 "왜 클러스터 타깃이 이해 태스크에 강한가".

### 논문 ② EnCodec (arXiv:2210.13438)
- **왜**: **acoustic token**의 대표. RVQ 기반 neural codec. Moshi·VALL-E가 그대로 사용 →
  Stage 3·5로 직결. HuBERT(semantic)와 짝을 이뤄 이 스테이지의 핵심 갈림길을 완성.
- **핵심 포인트**: encoder–RVQ–decoder 구조, 코드북 여러 개(계단식 잔차 정제)가 한 프레임을
  여러 token으로 표현 → parallel-stream 설계(Moshi)의 근거.

> 두 논문을 **짝으로** 읽는 게 포인트: semantic(HuBERT) vs acoustic(EnCodec). 이 대비가
> 이후 모든 SpeechLM 설계의 첫 번째 결정.

### 실습 코드 — `acoustic vs semantic 토큰 비교`
같은 음성을 EnCodec(acoustic token)과 HuBERT(semantic feature)로 각각 뽑아 차이를 체감.
- `01_waveform_mel.py` — waveform·STFT·mel-spectrogram 시각화 (frame rate 감각)
- `02_encodec_tokens.py` — EnCodec encode → discrete token → decode 복원, 토큰 shape·bitrate 확인
- `03_hubert_features.py` — HuBERT 레이어별 feature 추출, k-means로 discrete unit 만들기
- `04_compare_acoustic_vs_semantic.py` — 두 표현을 나란히 비교(복원 가능성 vs 내용 보존)

---

## Stage 2 — 음성 이해 (ASR · diarization)

**한 문장**: 음성을 텍스트·의미로 바꾸는 이해 계열. cascade의 앞단이자 SpeakerLM의 빌드업.

### 논문 ① Whisper (arXiv:2212.04356)
- **왜**: 오늘날 실무 ASR의 디폴트. 아키텍처보다 **"68만 시간 weak supervision"이라는 데이터
  스케일 철학**이 포인트. cascade voice agent의 첫 스테이지 = 전체 파이프라인 성능·latency 상한.

### 논문 ② SALMONN (arXiv:2310.13289)
- **왜**: `speech encoder + adapter + LLM` 패턴의 대표 = **vision-LLM의 오디오판**. "이해"가
  단순 전사를 넘어 QA·요약·instruction까지 확장되는 지점. audio-LLM의 뼈대를 잡는 논문.
- (대안: Qwen-Audio/Qwen2-Audio도 필독이지만, 구조를 가장 깔끔하게 보여주는 SALMONN을 선택.)

### 실습 코드 — `Whisper 전사 + pyannote diarization 이어붙이기`
- faster-whisper(CTranslate2)로 전사 → pyannote 3.1로 "누가 언제" → 둘을 수동으로 병합.
- **핵심**: 두 모듈을 따로 돌려 붙이면 생기는 **error propagation·overlap 취약성**을 몸으로 확인.
  → 왜 통합 모델(SpeakerLM)이 필요한지 Stage 5 도착점의 동기를 여기서 체득.

---

## Stage 3 — 음성 생성 (TTS)

**한 문장**: token·텍스트를 음성으로. 최신 흐름이 codec-LM과 flow-matching 두 갈래로 갈리는 이유.

### 논문 ① VALL-E (arXiv:2301.02111)
- **왜**: TTS를 **"audio codec token의 language modeling"**으로 재정의 → in-context voice
  cloning. **SpeechLM으로 가는 사고 전환**의 핵심. Stage 1의 EnCodec token 위에서 동작.

### 논문 ② CosyVoice (arXiv:2407.05407)
- **왜**: **flow-matching** zero-shot TTS. GLM-4-Voice·**BayLing-Duplex의 speech decoder 계보** →
  도착점으로 직결. VALL-E(codec-LM)와 짝을 이뤄 생성의 두 축을 완성.

### 실습 코드 — `CosyVoice2 zero-shot cloning vs 2단 baseline`
- CosyVoice2-0.5B로 reference 몇 초 → zero-shot 클로닝.
- 비교군: FastSpeech2+HiFi-GAN(고전 2단 구조) 또는 F5-TTS.
- **핵심**: 같은 문장을 두 방식으로 뽑아 **품질·속도·화자 유사도** 비교.

---

## Stage 4 — Speech LLM (통합)

**한 문장**: 이해(2)와 생성(3)을 하나의 LLM으로. cascade의 3단이 단일 모델로 접힘.

### 논문 ① GLM-4-Voice (arXiv:2412.02612)
- **왜**: 저 frame-rate(12.5Hz) **native SpeechLM**. **BayLing-Duplex의 백본** = 도착점의 몸통.
  음성 token을 LLM vocab에 직접 넣어 이해·생성을 전부 next-token prediction으로.

### 논문 ② LLaMA-Omni (arXiv:2409.06666)
- **왜**: **modular·저지연**(226ms) speech interaction의 대표. **도착점 그룹(ICT/CAS)의 시작점**:
  LLaMA-Omni → LLaMA-Omni2 → BayLing-Duplex. native(GLM-4-Voice) vs modular(LLaMA-Omni)
  트레이드오프를 이 두 논문으로 대비.
- (배경 필독: **AudioLM** 2209.03143 — semantic+acoustic 계층 생성, SpeechLM 사고의 기틀. 여유되면.)

### 실습 코드 — `Mini-Omni 로컬 speech-to-speech + cascade 대조`
- 8GB 제약: GLM-4-Voice 9B·Qwen2.5-Omni 7B는 full inference 빠듯 → **공식 데모로 관찰**.
- 로컬 실행은 경량 **Mini-Omni**(2408.16725)로 speech-to-speech 흐름 체험.
- **핵심**: 같은 질문을 cascade(Whisper+LLM+CosyVoice, Stage 2·3 재활용)와 end-to-end로
  응답시켜 **latency·자연스러움·paralinguistic** 차이 비교.

---

## Stage 5 — Full-duplex & 도착점 두 논문

**한 문장**: 듣기와 말하기가 동시에. "cascaded → 통합 end-to-end" 테제의 종착.

> 배경 필독: **Moshi**(2410.00037, parallel RVQ + Inner Monologue) — full-duplex의 대표작,
> 실습 코드로도 사용. **dGSLM**(2203.16502)은 dual-channel 개척으로 함께 읽으면 좋음.

### 논문 ① SpeakerLM (arXiv:2508.06372) — 이해 축 도착점
- End-to-End Versatile Speaker Diarization and Recognition with Multimodal LLMs (AAAI 2026).
- **왜**: SD+ASR을 하나의 multimodal LLM으로 통합 → "누가·언제·무엇을"을 end-to-end로.
  Stage 2에서 체감한 cascade의 error propagation을 통합으로 해소. **이해 축에서 테제를 증명**.

### 논문 ② BayLing-Duplex (arXiv:2506.14528) — 대화 축 도착점
- Native Full-Duplex Speech Dialogue with a Single Autoregressive LLM (ICT/CAS, 2026.06).
- **왜**: GLM-4-Voice(9B) 백본에 **dialogue-state token 4개**(`[SILENCE]/[ASSISTANT]/[PAD]/[EPAD]`)
  + **3채널 block interleave**만으로 turn-taking을 평범한 next-token prediction으로 환원.
  400K 샘플 SFT + 경량 DPO → **academic team도 재현 가능한 full-duplex 전환 레시피**.
- 결과(vs Moshi): turn-taking 71.9→92.0%, 끼어들기 81.9→100%, S2S 2.17→3.39.

### 실습 코드 — `Moshi full-duplex vs GLM-4-Voice turn-based 대조`
- Moshi(Kyutai) 공개 데모/모델로 full-duplex 실시간 대화(끼어들기·backchannel) 체험.
- GLM-4-Voice 데모로 turn-based를 대조.
- **핵심**: "끼어들기가 되고 안 되고"의 UX 차이를 직접 느끼기. 8GB에선 Moshi q8 로컬 시도 +
  공식 hosted 데모 병행.

### 최종 산출
두 논문까지 읽으면 → 메인 페이지 의사결정 표 + BayLing 레시피 기반으로
**자체 voice agent의 half/full-duplex 전환 RFC** 작성.

---

## 논문 전체 목록 (복사용)

| Stage | 논문 | arXiv |
|---|---|---|
| 1 | HuBERT | 2106.07447 |
| 1 | EnCodec | 2210.13438 |
| 2 | Whisper | 2212.04356 |
| 2 | SALMONN | 2310.13289 |
| 3 | VALL-E | 2301.02111 |
| 3 | CosyVoice | 2407.05407 |
| 4 | GLM-4-Voice | 2412.02612 |
| 4 | LLaMA-Omni | 2409.06666 |
| 5 | SpeakerLM | 2508.06372 |
| 5 | BayLing-Duplex | 2506.14528 |
