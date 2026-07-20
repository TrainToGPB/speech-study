---
title: wav2vec 2.0 — 메커니즘 step-by-step
topic: representation
paper: wav2vec2
status: wip
env_tested: [home]
created: 2026-07-18
updated: 2026-07-18
links:
  paper: https://arxiv.org/abs/2006.11477
  notion: https://app.notion.com/p/39d74b55ec7c81eb81f1fed78d055a74
---

## 🎯 목표
노션 정리("wav2vec 2.0")의 7단계 pre-training 메커니즘을, 예시 오디오 하나를 잡고
실제 텐서로 step-by-step 확인한다: 음성 처리(feature encoder) → masking → context
network → quantizer → contrastive → diversity → 전체 objective. 각 단계의 shape와
중간값, 시각화를 노션 설명과 1:1 대조하는 게 목적.

## ⚙️ 실행 방법
```bash
cd study/representation/wav2vec2
python <스킬경로>/scripts/setup_env.py home   # 또는 work
python download.py     # (선택) 오디오·모델 미리 캐시
python run.py          # STEP 1~7 쭉 실행 → outputs/*.png
# 단계 개별 실행도 가능: python 01_feature_encoder.py
```

모델은 사전학습 SSL 체크포인트 `facebook/wav2vec2-base`(quantizer 포함, ~360MB).
예시 오디오는 `hf-internal-testing/librispeech_asr_dummy`의 첫 샘플. 오프라인이면
합성 사인파로 대체(메커니즘 shape는 확인되지만 음소 구조는 무의미).

## 파일 구성
- `common.py` — 모델·오디오·마스크 공유 로더 (SEED 고정)
- `01_feature_encoder.py` — waveform X → latent Z (CNN, 49Hz)
- `02_masking.py` — context 경로 span 마스킹 (p=0.065, M=10)
- `03_context_network.py` — Transformer → 문맥표현 C
- `04_quantizer.py` — product quantization (G=2, V=320) → discrete q_t
- `05_contrastive.py` — InfoNCE (K=100, κ=0.1), positive vs distractor
- `06_objective.py` — diversity loss + 전체 L = L_m + αL_d (α=0.1)

## 📊 결과 / 관찰
2026-07-18 · home(RTX 3060 Ti, torch 2.5.1+cu121, Python 3.12)에서 `run.py` 완주.
예시: LibriSpeech dummy `"MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSE"` (5.86s).

- **STEP 1** waveform (1, 93680) → Z (1, 292, 512), 실측 **49.9Hz**(20.1ms/frame) — 노션의 49Hz와 일치.
- **STEP 2** p=0.065·M=10로 span 2개(각 10 frame ≈ 200ms), 짧은 발화라 6.8%만 가려짐
  (노션의 ~49%는 긴 발화·많은 span 기준). Transformer=masked / quantizer=원본 비대칭 확인.
- **STEP 3** context C (1, 292, 768), masked 위치는 MASK 벡터 입력 → 문맥으로 c_t 추론.
- **STEP 4** G=2×V=320, 이 발화에서 고유 codeword 조합 **248/292**, perplexity 177.9(전체 frame).
- **STEP 5** masked 평균 cosine: **positive 0.487 vs distractor 0.034**, t=55에서 positive **순위 1/101**
  → 사전학습 모델이 진짜 target을 정확히 식별(그림 `05_contrastive_sim.png`에 명확히 분리).
- **STEP 6·7** `L = L_m + αL_d = 59.897 + 0.1×19.299 = 61.827` = HF `out.loss` **정확히 일치**.

그림 6종 → `outputs/*.png` (git 제외).

## 🧱 막힌 점 / TODO
- 최신 `datasets`가 오디오 디코딩에 `torchcodec` 요구 → `decode=False` + `soundfile` 직접
  디코딩으로 우회(common.py). 안 그러면 합성 사인파로 폴백돼 값이 무의미해짐(예: positive 순위 88/101).
- STEP 4 perplexity(전체 frame, 177.9)와 STEP 6 perplexity(masked-only, 22.4)는 측정 대상이 달라
  값이 다름. loss에 쓰이는 건 masked-only 쪽.
- HF `diversity_loss`는 노션의 negative-entropy 식과 형태가 다름(perplexity 기반 정규화).
  방향(다양성↑)은 동일 — 06_objective.py 주석 참고.
- TODO: STEP 4 discrete unit이 실제 음소 구조를 담는지(노션 Figure 3)는 단일 발화론 부족.
  여러 발화 모아 code↔음소 co-occurrence 그려보기.

## 📝 메모
- 노션 정리: [wav2vec 2.0](https://app.notion.com/p/39d74b55ec7c81eb81f1fed78d055a74)
- 같은 계보 continuous SSL feature → discrete unit: 다음은 [[study/representation/hubert]] (예정)
