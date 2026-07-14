# Stage 4 — Speech LLM (통합)

> 이해(2)와 생성(3)을 하나의 LLM으로. cascade의 `ASR→LLM→TTS` 3단이 단일 모델로 접힘.

## 목표
- **native vs modular** SpeechLM을 구분.
- 음성 token을 LLM vocab에 넣는 게 왜 스택 재사용에 유리한지 설명.
- streaming decoder가 latency에 어떻게 기여하는지 설명.

## 논문 2개 (Notion에서 리뷰)

| # | 논문 | arXiv | 왜 |
|---|---|---|---|
| ① | **GLM-4-Voice** | 2412.02612 | 저 frame-rate(12.5Hz) **native SpeechLM**. **BayLing-Duplex의 백본** = 도착점의 몸통 |
| ② | **LLaMA-Omni** | 2409.06666 | **modular·저지연(226ms)** 대표. **도착점 그룹(ICT/CAS)의 시작점** |

두 논문으로 native(GLM-4-Voice) vs modular(LLaMA-Omni) 트레이드오프를 대비.
배경 필독: **AudioLM**(2209.03143, semantic+acoustic 계층 생성, SpeechLM 사고의 기틀).
그 외: SpeechGPT, Freeze-Omni, Mini-Omni, Qwen2.5-Omni(Thinker-Talker).

## 실습 코드 — `Mini-Omni 로컬 S2S + cascade 대조`

**설계** (`s2s_vs_cascade.py`로 구현 예정):
1. **로컬 실행**: 경량 **Mini-Omni**(2408.16725)로 speech-to-speech 흐름 체험.
2. **cascade 대조군**: Stage 2·3 재활용 → Whisper + (텍스트 LLM) + CosyVoice.
3. 같은 질문에 두 방식으로 응답 → **latency · 자연스러움 · paralinguistic** 비교.
4. **관찰**: cascade는 텍스트 병목에서 감정·톤이 손실됨. end-to-end는 일부 보존.

**셋업 메모** (이 스테이지 전용 venv, 결과는 `stage4_speechlm/outputs/`):
```bash
cd stage4_speechlm
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# Mini-Omni는 공식 repo 클론 권장: git clone https://github.com/gpt-omni/mini-omni
```

**8GB 메모**:
- **로컬 가능**: Mini-Omni(~0.5B급 백본).
- **데모 관찰**: GLM-4-Voice(9B)·Qwen2.5-Omni(7B)는 full inference 빠듯 → 공식 데모로.
  단, GLM-4-Voice는 **Stage 5 도착점(BayLing) 백본**이므로 데모라도 꼭 체험.

## 자가 점검
- [ ] native와 modular SpeechLM의 트레이드오프를 설명할 수 있나
- [ ] 음성 token을 LLM vocab에 넣는 게 왜 스택 재사용에 유리한지 말할 수 있나
- [ ] streaming decoder가 latency에 어떻게 기여하는지 답할 수 있나

## 다음
→ [Stage 5 — Full-duplex & 도착점](../stage5_fullduplex/).
