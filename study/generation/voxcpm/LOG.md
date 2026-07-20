---
title: VoxCPM: tokenizer-free TTS
topic: speech-generation
paper: voxcpm
status: wip
env_tested: [work]
created: 2026-07-19
updated: 2026-07-19
links:
  paper: https://arxiv.org/abs/2509.24650
  notion: https://app.notion.com/p/39f74b55ec7c81f5bea6cc0fb6cb2980
---

## 🎯 목표
- tokenizer-free TTS 논문(VoxCPM)을 공식 0.5B 모델로 hands-on.
- 아키텍처(`TSLM`+`FSQ` semantic → `RALM` acoustic 잔차 → `LocDiT` diffusion) 동작 감 잡기.
- work(Mac·MPS)에서 로드→합성이 실제로 돌아가는지, 속도/품질 확인.

## ⚙️ 실행 방법
```bash
cd speech-generation/voxcpm
python <스킬경로>/scripts/setup_env.py work   # 최초 1회 (torch/torchaudio + requirements)
python download.py                            # 최초 1회 VoxCPM-0.5B 가중치(HF 캐시)
python run.py                                 # outputs/voxcpm_0.5b_demo.wav 생성
```

## 📊 결과 / 관찰
- 2026-07-19 (work·MPS): VoxCPM-0.5B 로드+합성 **성공**.
  - `voxcpm==2.0.3` 패키지 기본값은 VoxCPM2지만 `from_pretrained(hf_model_id="openbmb/VoxCPM-0.5B")`로 논문 원본 0.5B 명시 로드 OK.
  - 패키지가 MPS에서 dtype을 bf16→fp32로 자동 조정. `LocDiT` 확산 148 step, ~16 it/s.
  - 샘플: `outputs/voxcpm_0.5b_demo.wav` — 6.16s @ 16kHz, peak 0.99 / rms 0.12 (정상 음성).
  - `transformers==4.46.3` 그대로 호환(업그레이드 불필요) — home 핀과 충돌 없음.

## 🧱 막힌 점 / TODO
- home(8GB·CUDA) 미검증 — 0.5B면 8GB에 들어갈 가능성 높으나 확인 필요.
- denoiser(zipenhancer, modelscope)는 `load_denoiser=False`로 미로딩 → voice cloning 실습 시 필요.
- MPS 대비로 `PYTORCH_ENABLE_MPS_FALLBACK=1` 켜둠(일부 확산 연산 CPU 폴백 가능성).
- 후속: 참조오디오 voice cloning, VoxCPM2(48kHz)와 품질/속도 비교, LocDiT step/cfg 스윕.

## 📝 메모
- 노션 정리(방법론 중심): [VoxCPM 자료 라이브러리 페이지](https://app.notion.com/p/39f74b55ec7c81f5bea6cc0fb6cb2980) — S3 실습 메인
- tokenizer-free TTS: `TSLM`+`FSQ` semantic 골격 + `RALM` acoustic 잔차 → `LocDiT` diffusion. MiniCPM-4-0.5B backbone
- 관련 실험: [[speech-generation/tacotron1]] (attention 기반 고전 TTS 대비)
