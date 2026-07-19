---
title: VoxCPM: tokenizer-free TTS
topic: speech-generation
paper: voxcpm
status: planned
env_tested: [work]
created: 2026-07-19
updated: 2026-07-19
links:
  paper: https://arxiv.org/abs/2509.24650
  notion: https://app.notion.com/p/39f74b55ec7c81f5bea6cc0fb6cb2980
---

## 🎯 목표
<!-- 왜 이 실험을 하는가, 무엇을 확인하려는가 (2~3줄) -->

## ⚙️ 실행 방법
```bash
cd speech-generation/voxcpm
python <스킬경로>/scripts/setup_env.py home   # 또는 work
python run.py
```

## 📊 결과 / 관찰
<!-- YYYY-MM-DD: 무엇이 나왔나, 수치·인상 -->

## 🧱 막힌 점 / TODO
<!-- 에러, 8GB 한계, 다음에 할 것 -->

## 📝 메모
- 노션 정리(방법론 중심): [VoxCPM 자료 라이브러리 페이지](https://app.notion.com/p/39f74b55ec7c81f5bea6cc0fb6cb2980) — S3 실습 메인
- tokenizer-free TTS: `TSLM`+`FSQ` semantic 골격 + `RALM` acoustic 잔차 → `LocDiT` diffusion. MiniCPM-4-0.5B backbone
- <!-- 참고 링크, 관련 실험 [[topic/paper]] -->
