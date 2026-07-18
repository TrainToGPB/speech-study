---
title: Conformer 블록 구조: macaron FFN + MHSA + conv module 순서대로
topic: speech-recognition
paper: conformer
status: wip
env_tested: [work]
created: 2026-07-18
updated: 2026-07-18
links:
  paper: https://arxiv.org/abs/2005.08100
  notion: 
---

## 🎯 목표
Conformer 블록이 오디오 feature 시퀀스를 어떻게 처리하는지, 예시 오디오 하나로 **블록 내부
서브모듈을 순서대로 텐서로 따라간다**. `encoder.layers[0]` 한 블록을 macaron 순서
(½FFN → rel-pos MHSA → conv module → ½FFN → LN)대로 손으로 재조립하고, HF 공식 forward와
**bit-exact(allclose)** 로 일치하는지 검증한다. 특히 논문 시그니처인 **상대위치 MHSA**
(Transformer-XL content/position 분해 + rel-shift)를 깊게 판다. 마지막에 24층 전체 + CTC로
실제 LibriSpeech 전사까지 확인(capstone).

## ⚙️ 실행 방법
```bash
cd speech-recognition/conformer
python <스킬경로>/scripts/setup_env.py work   # 또는 home
python download.py     # (선택) 오디오·모델(~600M) 미리 캐시
python run.py          # STEP 1~6 쭉 실행 → outputs/*.png
# 단계 개별 실행도 가능: python 03_mhsa_relpos.py (선행 스텝 자동 빌드)
python viz_relpos_attention.py [layer=0] [gamma=0.4]   # STEP 3 심화
python viz_conv_module.py                              # STEP 4 심화
```

모델은 `facebook/wav2vec2-conformer-rel-pos-large-960h-ft` (hidden 1024·head 16·24층·rel-pos·
CTC). 예시 오디오는 `hf-internal-testing/librispeech_asr_dummy` 첫 샘플. device는 `shared/env.py`.

## 파일 구성
- `common.py` — 모델·프로세서·오디오 공유 로더 + `build_context`(front-end → conformer 입력 `(1,T,1024)`
  + rel-pos 임베딩 + `layers[0]`). SEED 고정.
- `01_frontend.py` — waveform → feature_extractor+feature_projection → `(1,292,1024)` + rel-pos `(1,583,1024)`
- `02_ffn_macaron1.py` — ½ FFN #1 (LN→Linear1024→4096→swish→4096→1024, residual **×0.5**)
- `03_mhsa_relpos.py` — **상대위치 MHSA**: `matrix_ac`(content)+`matrix_bd`(position)+rel_shift, HF 대조
- `04_conv_module.py` — LN→PWConv(→2048)→GLU→depthwise(k31,groups1024)→BN→swish→PWConv
- `05_ffn2_and_verify.py` — ½ FFN #2 → final LN → 블록 출력, **HF allclose 검증**
- `06_stack_and_transcribe.py` — 24층 스택 + CTC 전사 (capstone), 층별 ‖h‖ 성장
- `viz_relpos_attention.py` — content vs position 분해 + head별 attention. `python viz_relpos_attention.py [layer] [gamma]`
- `viz_conv_module.py` — GLU 게이트 분포 + depthwise 커널(수용영역)

## 📊 결과 / 관찰
2026-07-18 · work(MacBook Pro M5, MPS, torch 2.13.0, transformers 4.46.3, Python 3.12)에서 `run.py` 완주.
예시: LibriSpeech dummy `"MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES..."` (5.86s).

- **STEP 1** waveform `(1,93680)` → feature_extractor `(1,292,512)` → feature_projection `(1,292,1024)`.
  CNN 7층이 시간축을 93680→**292 frame(~50Hz, 20ms/frame)** 로 줄임. rel-pos 임베딩 `(1,583,1024)`
  = **2T-1**(±291 상대거리). conformer는 wav2vec2와 달리 **pos_conv를 안 쓰고** 위치정보는 전적으로
  상대위치로만 준다. 입력 frame별 ‖·‖ 105.1.
- **STEP 2** ½ FFN #1이 `‖·‖ 105→378`로 **크게 부풀림**(‖Δ‖/‖res‖ 3.55). 0.5 계수인데도 FFN 기여가
  큼 — attention 앞에서 표현을 강하게 전처리. (블록 끝 final LN이 다시 되돌림 → layer0 출력 ‖·‖ 72.9.)
- **STEP 3 (시그니처)** q/k/v `(1,16,292,64)`. content `matrix_ac`와 position `matrix_bd`로 분해:
  **position 항이 content보다 크다** — 원 score에서 `‖bd‖ 468k` vs `‖ac‖ 177k`, |bd| 평균 85.3 vs |ac| 38.9.
  즉 layer 0의 attention은 "무슨 내용인가"보다 **"얼마나 떨어졌나"(상대거리)** 에 더 크게 반응. rel_shift로
  `(292,583)`→`(292,292)` 정렬. 엔트로피 7.03bit(max 8.19) — 중간 수준. 손 재현 vs HF self_attn
  **max|Δ| = 0.00e+00 완전 일치 ✅**(rel-shift 인덱싱까지 정확).
- **STEP 3 시각화**(`viz_relpos_attention.py`, L0): **content**는 세로 줄무늬(특정 key frame이 query 무관하게
  정보성↑), **position**은 rel-shift 특유의 가로 밴드+대각 텍스처, **최종 probs**는 둘의 합 + attention-sink성
  세로줄. layer 0 attention은 의외로 **장거리**(평균 query–key 거리 **113 frame ≈ 2.3s**) — 대각 집중이 아님.
- **STEP 4** conv module: pointwise `1024→2048` → GLU가 절반으로(gate 평균 0.546) → **depthwise k31,
  groups=1024** → 수용영역 **±15 frame(±300ms)** 의 국소 시간 필터. 기여 ‖Δ‖/‖res‖ 0.38로 attention보다 온건.
  `‖·‖ 355→335`.
- **STEP 5** ½ FFN #2 → final LN → 블록 출력. **손 재조립 블록 vs HF `layers[0].forward` max|Δ| = 0.00e+00
  일치 ✅**. final LN이 블록 내부에서 부푼 ‖·‖(315)를 std 2.277로 재정규화(→ frame ‖·‖ ≈ 72.9).
- **STEP 6** 우리 블록 출력 == 모델 `hidden_states[1]`(layer0 출력) **max|Δ| 0.00e+00 ✅**. 층별 ‖h‖는
  105(입력)→73~87(중간층 안정)→마지막 몇 층에서 감소(23층 19.9)→final LN 후 6.6. whisper의 단조 증가와 달리
  **중간 안정 후 수축**. CTC 전사 `'MISTER QUILTER ... WELCOME HIS GOSPEL'` = **정답과 완전 일치 ✅**.

그림: `outputs/{01_frontend, 06_layer_norms, viz_relpos_attention_L0_decomp, viz_relpos_attention_L0_heads, viz_conv_module}.png` (git 제외).

## 🧱 막힌 점 / TODO
- **home 8GB 미검증**: `-960h-ft`는 **618.6M 파라미터**(fp32 ≈ 2.4GB). work(32GB)는 여유롭지만 home 8GB에서는
  실측 필요 — 짧은 클립 forward는 될 가능성이 크나 OOM 시 fp16 로드(`torch_dtype=torch.float16`)나 CPU 폴백 고려.
- **requirements 레이스 재현**: `setup_env.py`가 requirements.txt(초기엔 transformers만)를 읽어서
  datasets/soundfile/matplotlib가 빠졌다 → `uv pip install -r`로 보충. whisper LOG의 교훈과 동일(다음엔 setup 전 확정).
- **체크포인트 포맷**: `-960h-ft`는 safetensors로 로드됨(4.46.3 핀은 home CUDA<2.6 대비 유지).
- TODO: layer 0은 장거리였는데 **깊은 층(예: L12, L23)의 attention은 국소/음소성인지** `viz_relpos_attention.py 12`로 비교.
  conv module gate가 실제로 어떤 채널을 여닫는지 음소 경계와 대조.

## 📝 메모
- 논문: [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- 블록 순서: `x = ffn1(LN)·0.5+x → dropout(self_attn(LN,rel_pos))+x → conv_module(x)+x → ffn2(LN)·0.5+x → final_LN`.
  **macaron = ½FFN 두 개가 attention을 감쌈**. residual은 FFN만 ×0.5.
- front-end(CNN 7층·49~50Hz)는 [[speech-representation/wav2vec2]] STEP 1과 동일 계보. 같은 wav2vec2 feature encoder.
- 프레임레이트 **50Hz(20ms)** 는 [[speech-recognition/whisper]] 인코더와 같음. 단 whisper는 log-Mel+conv2층·양방향
  absolute, conformer는 raw+CNN7층·**상대위치** MHSA + conv module로 국소성까지 명시적으로 잡는 게 차이.
- rel-pos MHSA의 position 항이 content보다 크다는 건 흥미 — 사전학습이 상대거리 편향을 크게 실었다는 뜻.
