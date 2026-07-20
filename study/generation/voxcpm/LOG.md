---
title: VoxCPM: tokenizer-free TTS
topic: generation
paper: voxcpm
status: wip
env_tested: [work]
created: 2026-07-19
updated: 2026-07-20
links:
  paper: https://arxiv.org/abs/2509.24650
  notion: https://app.notion.com/p/39f74b55ec7c81f5bea6cc0fb6cb2980
---

## 🎯 목표
- tokenizer-free TTS 논문(VoxCPM)을 공식 0.5B 모델로 **step-by-step 해부** — 아키텍처의
  각 모듈을 실제 가중치로 하나씩 통과시키며 텐서로 동작을 파악.
- `TSLM`+`FSQ`(semantic skeleton) → `RALM`(acoustic residual) → `LocDiT`(flow-matching)
  → `Causal Audio VAE`(continuous latent) 흐름을 코드로 재현.
- work(Mac·MPS)에서 로드→합성→해부가 실제로 돌아가는지, semantic-acoustic 분리를 확인.

## 🧩 단계별 스크립트 (run.py가 ctx를 스레딩하며 01→06 순서 실행)
- `common.py` — 모델 1회 로드(`model.tts_model=VoxCPMModel`), 참조 오디오 준비, prefill 유틸.
- `01_audio_vae.py` — Causal Audio VAE: 16kHz waveform ↔ 25Hz×64d **연속 latent**(640×),
  patch_size 2 → 12.5Hz. round-trip 재구성으로 "tokenizer-free(이산 코드 없음)" 확인.
- `02_text_and_condition.py` — 텍스트 BPE + audio_start(101) + text/audio mask 조립.
- `03_tslm_fsq.py` — LocEnc → TSLM(MiniCPM 24L) → FSQ semi-discrete skeleton(tanh→round(*9)/9).
- `04_ralm_residual.py` — RALM(6L) acoustic 잔차 → `h_final = skeleton + residual`(핵심 등식).
- `05_locdit_diffusion.py` — LocDiT flow-matching: noise→다음 latent patch(Euler ODE), CFG·step 스윕.
- `06_generate_and_stop.py` — 전체 AR 루프 + Stop Predictor(hook)로 voice cloning 합성(capstone).
- `viz_decoupling_tsne.py` — (독립) semantic-acoustic decoupling 통제 실험(Fig 2).

## ⚙️ 실행 방법
```bash
cd study/generation/voxcpm
python <스킬경로>/scripts/setup_env.py work   # 최초 1회 (torch/torchaudio + requirements)
python download.py                            # 최초 1회 VoxCPM-0.5B 가중치(HF 캐시) + 참조 오디오
python run.py                                 # STEP 1~6 순차 실행 → outputs/*.png, *.wav
python viz_decoupling_tsne.py                 # (선택) 분리 증거 t-SNE
python 03_tslm_fsq.py                         # 개별 STEP도 단독 실행 가능(앞 STEP 자동 체이닝)
```

## 📊 결과 / 관찰
- 2026-07-20 (work·MPS, torch 2.13.0): **STEP 1~6 + viz 전부 성공**.
  - STEP1 VAE round-trip corr **0.926** / rmse 0.024 — 16kHz→25Hz(640×), patch 12.5Hz. 연속 latent(실수, 이산 코드 아님).
  - STEP3 FSQ: 오디오 위치 256d code가 정확히 **19개 정수레벨 {-9..9}** 에만 존재(=semi-discrete). hidden std 0.62→0.13로 안정화.
  - STEP4 h_final = skeleton(‖·‖≈6.5) + residual(‖·‖≈15.8) — 잔차가 표현력의 큰 몫.
  - STEP5 LocDiT: n_timesteps 2→30 갈수록 50-step 기준 L2 5.26→0.14로 수렴, cfg 1→3에서 |patch| 6.6→7.6(외삽 강해짐).
  - STEP6 voice cloning: 목표 문장 2.8~3.0s 합성, **RTF ~1.3~1.6**(MPS라 real-time보다 느림; 논문 4090은 0.17). Stop Predictor가 마지막 직전 스텝에서 P(stop)>0.5로 종료.
  - **decoupling 통제 실험**(내용 고정·목소리 6개): 목소리 라벨 silhouette **skeleton +0.00 vs residual +0.10**
    → skeleton은 목소리 무관(내용 축), residual은 목소리별 군집(음색 축). Notion Fig 2 재현 ✔ (`outputs/viz_decoupling_tsne.png`).
- (이전) 2026-07-19: 최소 데모(text-only 합성)만 성공 — 지금은 그 흐름을 STEP 6에 흡수.

## 🧱 막힌 점 / TODO
- **torchcodec/ffmpeg**: torchaudio 2.11이 `torchaudio.load`를 torchcodec 백엔드로 위임 →
  Mac에 ffmpeg(libavutil.56) 없으면 죽음(`backend="soundfile"`로도 우회 안 됨). voxcpm 내부가
  참조 오디오를 `torchaudio.load`로 읽으므로 `common.patch_torchaudio_load()`가 soundfile 로더로
  몽키패치해 우회(voice cloning 경로 필수). text-only 합성만 하면 안 걸림.
- **설치 voxcpm==2.0.3의 `generate`엔 `seed=` 인자 없음**(GitHub 최신 소스엔 있음). 목소리 seed 고정은
  `torch.manual_seed()`를 generate 직전에 호출해 대체.
- 공식 실습 데이터셋(librispeech dummy)은 **화자 1명(1272)뿐** → 다화자 Fig 2 대신 "내용 고정·목소리 변주" 통제 실험으로 재현.
- MPS는 dtype이 bf16→**fp32 자동 조정**(voxcpm가 처리). `PYTORCH_ENABLE_MPS_FALLBACK=1` 켜둠.
- home(8GB·CUDA) 미검증 — 0.5B면 8GB에 들어갈 가능성 높으나 확인 필요. CUDA면 `optimize=True`(triton)로 훨씬 빠를 것.
- 후속: denoiser(zipenhancer) 붙여 실제 화자 복제 품질, VoxCPM2(48kHz) 대비, cfg/step 스윕의 WER·음질 영향.

## 📝 메모
- 노션 정리(방법론 중심): [VoxCPM 자료 라이브러리 페이지](https://app.notion.com/p/39f74b55ec7c81f5bea6cc0fb6cb2980) — S3 실습 메인
- `model.tts_model`(VoxCPMModel) attribute ↔ 논문 모듈: `audio_vae`/`feat_encoder`(LocEnc)/`base_lm`(TSLM)/
  `fsq_layer`(FSQ)/`residual_lm`(RALM)/`feat_decoder`(LocDiT+UnifiedCFM)/`stop_head`(Stop Predictor).
- 관련 실험: [[study/generation/tacotron1]] (attention 기반 고전 TTS 대비)
