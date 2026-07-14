# Stage 5 — Full-duplex & 도착점 두 논문

> 듣기와 말하기가 **동시에**. "cascaded → 통합 end-to-end" 테제의 종착.

## 목표
- full-duplex 인터랙션(overlap·backchannel·barge-in)을 이해.
- 시간·동시성을 시퀀스로 인코딩하는 3가지 접근을 구분.
- 도착점 두 논문을 정독하고, 자체 voice agent의 **duplex 전환 RFC**를 쓸 근거 확보.

## 배경 필독 (도착점으로 가는 계보)
- **dGSLM**(2203.16502): dual-channel 대화 모델링 개척.
- **Moshi**(2410.00037): parallel RVQ + Inner Monologue. full-duplex 대표작 — **반드시 정독**. 실습 코드로도 사용.
- 배경: SyncLLM(clock token), OmniFlatten(channel interleaving), Full-Duplex-Bench(2503.04721, 평가).

시간 인코딩 3접근: **parallel stream**(Moshi) · **explicit clock token**(SyncLLM) ·
**channel interleaving**(OmniFlatten, BayLing-Duplex).

## 도착점 논문 2개 (Notion에서 정독 리뷰)

### ① SpeakerLM — 이해 축 (arXiv:2508.06372, AAAI 2026)
*End-to-End Versatile Speaker Diarization and Recognition with Multimodal LLMs*
- SD + ASR을 하나의 multimodal LLM으로 통합 → "누가·언제·무엇을"을 end-to-end로.
- Stage 2에서 체감한 cascade의 error propagation을 통합으로 해소. **이해 축에서 테제 증명.**
- 유연한 speaker registration, 대규모 실데이터 multi-stage 학습, cascade SOTA 상회.

### ② BayLing-Duplex — 대화 축 (arXiv:2506.14528, ICT/CAS 2026.06)
*Native Full-Duplex Speech Dialogue with a Single Autoregressive LLM*
- GLM-4-Voice(9B) 백본에 **dialogue-state token 4개**만 추가:
  `[SILENCE]`(침묵) · `[ASSISTANT]`(응답 시작) · `[PAD]`(텍스트 완료·음성 생성중) · `[EPAD]`(모두 완료)
- **3채널 block interleave**: user speech / assistant text / assistant speech (N:M=10:5, block ∆t=0.8s)
- → turn-taking·끼어들기 결정이 전부 **평범한 next-token prediction**으로 환원 (별도 state machine 불필요)
- 학습: 400K full-duplex SFT + 경량 DPO. 결과(vs Moshi): turn-taking 71.9→92.0%,
  끼어들기 81.9→100%, S2S 2.17→3.39.
- **왜 도착점인가**: 밑바닥부터 안 만들어도 됨. 강한 turn-based SpeechLM(GLM-4-Voice)에서
  시작해 **소량 데이터 + 토큰 몇 개 + DPO**로 full-duplex 전환. 팀의 현실적 도입 경로 그 자체.

## 실습 코드 — `Moshi full-duplex vs GLM-4-Voice turn-based 대조`

**설계** (`duplex_demo.md` + 스크립트로 진행 예정):
1. **Moshi**(Kyutai) 공개 데모/모델로 full-duplex 실시간 대화 — 끼어들기·backchannel 체험.
2. **GLM-4-Voice** 데모로 turn-based 대조 (BayLing 백본이므로 필수 체험).
3. **관찰**: "끼어들기가 되고 안 되고"의 UX 차이. VAD 의존 vs 모델 내재화.

**셋업 메모** (이 스테이지 전용 venv, 결과는 `stage5_fullduplex/outputs/`):
```bash
cd stage5_fullduplex
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements.txt        # moshi 포함
```

**8GB 메모**: Moshi(7B)는 q8로 로컬 시도 가능하나 빠듯 → **공식 hosted 데모 병행** 권장.
GLM-4-Voice(9B)도 데모 위주.

## 자가 점검
- [ ] full-duplex의 "시간·동시성" 3가지 인코딩 접근을 구분할 수 있나
- [ ] BayLing이 4개 token만으로 turn-taking을 next-token prediction으로 만든 방식을 설명할 수 있나
- [ ] SpeakerLM·BayLing이 공유하는 테제를 자기 팀 cascade에 어떻게 적용할지 말할 수 있나

## 최종 산출
두 논문까지 읽으면 → 메인 페이지 의사결정 표 + BayLing 레시피 기반으로
**자체 voice agent의 half/full-duplex 전환 RFC** 작성.
