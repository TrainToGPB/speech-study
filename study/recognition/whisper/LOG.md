---
title: Whisper 인코더 동작 메커니즘 순서대로 파악
topic: speech-recognition
paper: whisper
status: wip
env_tested: [work]
created: 2026-07-18
updated: 2026-07-18
links:
  paper: https://arxiv.org/abs/2212.04356
  notion: 
---

## 🎯 목표
Whisper 인코더가 raw waveform을 어떻게 `(1500 x d_model)` 문맥 시퀀스로 바꾸는지,
예시 오디오 하나를 잡고 **STEP 1~6을 순서대로** 실제 텐서로 따라간다:
log-Mel → conv stem(다운샘플) → positional → encoder blocks(self-attn) → final LN →
(capstone) 그 출력이 디코더 cross-attention으로 전사에 쓰이는 모습. 각 단계의 shape·중간값·
시각화를 확인하고, STEP 2~5는 HF WhisperEncoder와 bit-exact로 일치하는지 검증한다.

## ⚙️ 실행 방법
```bash
cd speech-recognition/whisper
python <스킬경로>/scripts/setup_env.py work   # 또는 home
python download.py     # (선택) 오디오·모델 미리 캐시
python run.py          # STEP 1~6 쭉 실행 → outputs/*.png
# 단계 개별 실행도 가능: python 04_encoder_blocks.py (컨텍스트 자동 빌드)
```

모델은 `openai/whisper-base`(enc 6층·d_model 512·head 8·mel 80) — 메커니즘이 또렷하고
Mac/MPS에서 가볍다. 예시 오디오는 `hf-internal-testing/librispeech_asr_dummy` 첫 샘플.
오프라인이면 합성 사인파로 폴백(shape는 보이지만 음소 구조는 무의미).

## 파일 구성
- `common.py` — 모델·프로세서·오디오·log-Mel 공유 로더 (SEED 고정, eager attention)
- `01_logmel.py` — waveform → log-Mel `(1,80,3000)` (30s pad, 100Hz)
- `02_conv_stem.py` — 2×Conv1d+GELU → `(1,1500,512)` (시간축 3000→1500, 50Hz)
- `03_positional.py` — 고정 sinusoidal positional embedding 더하기
- `04_encoder_blocks.py` — pre-LN transformer 블록 ×6 (self-attn, 마스크 없음)
- `05_final_ln.py` — final LayerNorm → encoder 출력, **HF와 allclose 검증**
- `06_cross_attention.py` — 인코더 출력 → 디코더 cross-attention 정렬 (capstone)
- `viz_attention.py` — STEP 4 보조 시각화(기본 L5): crop+PowerNorm+행정규화+head별. `python viz_attention.py [layer] [gamma]`
- `viz_encoder_out.py` — STEP 5 보조 시각화: 발화구간 crop + 대칭클립 / dim별 z-score / frame×frame cosine. `python viz_encoder_out.py`

## 📊 결과 / 관찰
2026-07-18 · work(MacBook Pro M5, MPS, torch 2.13.0, transformers 4.46.3, Python 3.12)에서 `run.py` 완주.
예시: LibriSpeech dummy `"MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSE"` (5.86s).

- **STEP 1** waveform `(93,680,)` → log-Mel `(1, 80, 3000)`. 30s로 pad돼 frame은 항상 3000
  (**100Hz, 10ms hop**). 실제 발화 5.9s = **586 frame**, 나머지 2414개는 무음 패딩.
- **STEP 2** conv1(80→512, k3 s1 p1)→`(1,512,3000)`, conv2(512→512, **k3 s2 p1**)→`(1,512,1500)`,
  permute→`(1,1500,512)`. **stride 2 한 번**이 프레임레이트를 100Hz→**50Hz(20ms/frame)**로 낮춘다.
- **STEP 3** `embed_positions.weight (1500,512)`, **requires_grad=False** → 학습 안 하는 고정 sinusoid.
  pos 0의 dim 0~5가 전부 0인 건 앞쪽 dim이 sin 블록이라 `sin(0)=0`이기 때문(정상).
- **STEP 4** 6블록·8head·d512·MLP2048, **causal mask 없음(양방향)**. residual stream ‖h‖가
  layer 따라 **13.95 → 50.63**로 누적 증가(특히 L3~L5). attn 엔트로피 7~9.5bit(최대 log2 1500≈10.55) —
  L0가 가장 집중(7.10), 중간층은 더 퍼짐.
- **STEP 4 시각화**(`viz_attention.py`, L5): 1500² attention은 softmax(행합=1)라 평균 셀값 ~1/301,
  선형 스케일이면 안 보임 → 실제발화 crop + `PowerNorm(γ=0.4)`로 가시화. **head별로 역할이 갈림**:
  head 0·2·4·**7**은 선명한 국소 대각(7은 거의 identity), head 3·6은 전역 격자, head 5는 넓은 대각 밴드,
  head 1은 대각+오프셋 밴드. 평균 엔트로피 8.45는 이 초집중·초분산 head들을 뭉갠 값 → head-평균 그림이
  흐릿했던 이유. (`LogNorm`은 0값·노이즈 바닥 때문에 확률맵엔 부적합, PowerNorm이 표준.)
- **STEP 5** `layer_norm`→`(1,1500,512)`. 우리가 STEP 2~5에서 손으로 재현한 파이프라인 vs
  HF `enc(mel).last_hidden_state` → **max|Δ| = 0.00e+00, 완전 일치 ✅**(재현 정확). residual ‖h‖ 50.6이
  LN 후 std 1.46으로 재정규화(std≠1은 학습 affine γ).
- **STEP 5 시각화**(`viz_encoder_out.py`, 발화구간 crop): ① 대칭 diverging에서 **outlier dim 5~6개**
  (dim ~85/150/320/345/395)가 상수 오프셋으로 크게 켜짐 → raw 히트맵이 흐릿했던 이유. ② dim별 z-score로
  시간변화 증폭 → 무음 구간(5.86s 이후)이 확연히 매끈. ③ **frame×frame cosine**: 대각 띠 두께 = 표현
  변화 속도(두꺼움=지속음/안정, 가늘음=전이), 우하단 밝은 블록=무음 frame 균일.
- **STEP 6** 전사 `'Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'`
  = 정답과 대소문자·문장부호 빼고 일치. cross-attn `(6층, 8head, 26토큰, 1500)`. **alignment_heads 8개**
  (layer 3~5)만 평균하면 **단조 계단 정렬**: `Mr 0.94s → Qu 0.98 → apostle 2.16 → classes 3.34 →
  gospel 5.74s`. 인코더 출력이 "시간이 인덱싱된 표현"이고 디코더가 그 시각을 짚어 단어를 뽑음이 드러남.

그림 6종 → `outputs/*.png` (git 제외).

## 🧱 막힌 점 / TODO
- **STEP 6 정렬은 alignment_heads가 정답.** 모든 cross-attn head를 평균하면 오디오 시작부
  frame~6(**0.12s**)의 attention-sink에 대부분 토큰이 몰려 정렬이 안 보인다. `sink 질량<0.6` 필터는
  base에선 안 걸림(sink가 한 frame에 뾰족). `generation_config.alignment_heads`
  `[[3,1],[4,2],[4,3],[4,7],[5,1],[5,2],[5,4],[5,6]]`(단어 타임스탬프/DTW용)만 평균해야 깨끗함.
  마지막 `.`는 여전히 0.12s로 튐(종결부호는 정렬 신호 약함).
- **requirements 레이스**: `setup_env.py`가 requirements.txt를 읽은 뒤 내가 갱신해서
  datasets/soundfile/matplotlib가 빠졌다 → venv에 수동 `uv pip install -r`로 보충. 다음엔 setup 전에
  requirements부터 확정할 것.
- ~~그림 한글 폰트 깨짐 경고(DejaVu Sans, `Glyph missing`)~~ → **해결**: `common.use_korean_font()`가
  import 시 `AppleGothic` 등록(work). home(Linux)엔 AppleGothic이 없으니 `NanumGothic` 등 대체 필요할 수 있음.
- TODO: `large-v3`(mel 128·enc 32층)로 바꿔 mel bin·깊이 차이 관찰(work 32GB면 가능). 진짜 DTW로
  word-level timestamp 뽑아 STEP 6 정렬을 정량화.

## 📝 메모
- 논문: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- 인코더 프레임레이트 **50Hz(20ms)** 는 [[speech-representation/wav2vec2]]의 feature encoder 49Hz와
  우연히 비슷. 단 whisper는 **log-Mel + conv 2층**, wav2vec2는 **raw waveform + CNN 7층**으로 도달 경로가 다름.
- 인코더는 causal mask가 없어 30초 전체를 양방향으로 본다(오프라인 전사). 디코더만 causal.
