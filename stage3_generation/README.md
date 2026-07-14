# Stage 3 — 음성 생성 (TTS)

> token·텍스트를 음성으로. 최신 흐름이 **codec-LM**과 **flow-matching** 두 갈래로 갈리는 이유.

## 목표
- cascade의 마지막 스테이지(TTS) 계보를 안다.
- **VALL-E(codec-LM)** vs **flow-matching(CosyVoice)** 두 축을 구분.
- SpeechLM의 speech decoder를 읽을 기반 마련.

## 논문 2개 (Notion에서 리뷰)

| # | 논문 | arXiv | 왜 |
|---|---|---|---|
| ① | **VALL-E** | 2301.02111 | TTS를 "codec token의 language modeling"으로 재정의 → in-context cloning. SpeechLM으로 가는 사고 전환 |
| ② | **CosyVoice** | 2407.05407 | flow-matching zero-shot TTS. GLM-4-Voice·**BayLing-Duplex의 speech decoder 계보** |

배경: HiFi-GAN(2010.05646, vocoder 실무 표준), VITS(2106.06103, end-to-end 분기점),
Tacotron2/FastSpeech2(2단 구조 고전), F5-TTS(2410.06885, DiT+flow-matching 최신 오픈).

## 실습 코드 — `CosyVoice2 zero-shot cloning vs 2단 baseline`

**설계** (`tts_compare.py`로 구현 예정):
1. reference 음성 몇 초 → **CosyVoice2-0.5B** zero-shot voice cloning.
2. 비교군: 고전 2단(FastSpeech2 + HiFi-GAN) 또는 **F5-TTS**.
3. 같은 문장을 두 방식으로 합성 → **품질·속도·화자 유사도** 비교.
4. Stage 1의 EnCodec 복원음과도 비교하면 "codec 복원 vs TTS 생성"의 차이가 보인다.

**셋업 메모** (이 스테이지 전용 venv, 결과는 `stage3_generation/outputs/`):
```bash
cd stage3_generation
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements.txt        # f5-tts 포함(가벼운 비교군)
# CosyVoice2(Alibaba)는 공식 repo 클론 후 설치, 모델은 modelscope/HF에서:
#   git clone https://github.com/FunAudioLLM/CosyVoice
```

**8GB 메모**: CosyVoice2-0.5B·F5-TTS는 3060 Ti에서 실행 가능. HiFi-GAN 단독은 매우 가벼움.

## 자가 점검
- [ ] acoustic model과 vocoder의 역할 분담을 설명할 수 있나
- [ ] VALL-E가 TTS를 "language modeling"으로 본 게 왜 중요한지 말할 수 있나
- [ ] flow-matching TTS가 기존 대비 뭘 개선했는지 답할 수 있나

## 다음
→ [Stage 4 — Speech LLM](../stage4_speechlm/). token·LLM·decoder가 하나로 묶인다.
