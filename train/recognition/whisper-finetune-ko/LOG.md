---
title: Whisper 한국어 미니 파인튜닝 (CER 전/후, M5·MPS)
topic: recognition
idea: whisper-finetune-ko
status: planned
env_tested: []
created: 2026-07-20
updated: 2026-07-20
links:
  paper: 
  notion: 
---

## 🎯 목표
GPU 서버 없이 **M5(32GB·MPS)** 에서 기존 multilingual `openai/whisper-base`를
한국어(**Zeroth-Korean** 소량)에 full fine-tune 해서, **CER이 실제로 떨어지는지**를
전/후로 확인한다. 학습 자체가 목적이 아니라 "데이터 파이프라인 → 학습 루프 → 지표 개선"의
한 사이클을 손으로 도는 게 목표. 지표는 한국어 특성상 WER보다 **CER**을 주로 본다.

## ⚙️ 실행 방법
```bash
cd train/recognition/whisper-finetune-ko
python ../../../.claude/skills/speech-study/scripts/setup_env.py work
python download.py       # whisper-base + Zeroth subset + CER metric 캐시
python 01_data.py        # 전처리 한 샘플 해부 (mel·label)
python 02_baseline.py    # 파인튜닝 전 CER
python 03_train.py       # full fine-tune (MPS, max_steps 소량)
python 04_evaluate.py    # 파인튜닝 후 CER + 전/후 비교
# run.py 로 01→04 일괄 실행도 가능
```
모델 `openai/whisper-base`, 데이터 `Bingsu/zeroth-korean`. MPS 안전 설정(fp32,
num_workers 0)은 `common.py`에 모음.

## 📊 결과 / 관찰
<!-- YYYY-MM-DD: 무엇이 나왔나, 수치·인상 -->

## 🧱 막힌 점 / TODO
<!-- 에러, 8GB 한계, 다음에 할 것 -->

## 📝 메모
<!-- 참고 링크, 관련 실험 [[topic/paper]] -->
