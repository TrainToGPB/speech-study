---
title: Whisper 한국어 미니 파인튜닝 (CER 전/후, M5·MPS)
topic: recognition
idea: whisper-finetune-ko
status: wip
env_tested: [work]
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
2026-07-20 · work(M5, MPS)에서 01~04 완주. whisper-base + Zeroth-Korean
train 1000·eval 100, max_steps 200, fp32.
- baseline CER = 0.1769 → fine-tuned CER = 0.1387 (Δ +0.0382, 좋아짐).
- 예시 triple에서 숫자 표기가 눈에 띄게 개선됨: base는 "3월"·"2018년"처럼 아라비아
  숫자로 뽑는 반면 ft는 Zeroth 전사 관례대로 "삼 월"·"이천 십 팔년"으로 한글 숫자
  표기를 학습해 정답과 더 가깝게 맞춤. 문장 끝 마침표도 base는 붙이고 ft는 (정답처럼)
  생략하는 방향으로 스타일이 이동.
- 학습 시간 ~6.5분 (batch 8, MPS), loss 1.07 → 0.28 (train_loss 0.466).

## 🧱 막힌 점 / TODO
<!-- 에러, 8GB 한계, 다음에 할 것 -->

## 📝 메모
<!-- 참고 링크, 관련 실험 [[topic/paper]] -->
