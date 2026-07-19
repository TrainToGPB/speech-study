---
title: Tacotron: end-to-end seq2seq TTS — 텍스트 → mel → Griffin-Lim 파형
topic: speech-generation
paper: tacotron1
status: wip
env_tested: [work]
created: 2026-07-19
updated: 2026-07-19
links:
  paper: https://arxiv.org/abs/1703.10135
  notion: https://app.notion.com/p/39f74b55ec7c81158834c380a37bf247
---

## 🎯 목표
Tacotron1이 문자열을 어떻게 스펙트로그램으로 바꾸는지, 아키텍처를 손으로 재조립하며 따라간다.
논문 시그니처 세 가지를 판다: (1) **CBHG 모듈**(1-D conv bank K개 filter width + highway net +
BiGRU), (2) **reduction factor r** — 디코더가 스텝당 r개 mel 프레임을 방출하는 autoregressive
content-based(Bahdanau) attention decoder, (3) neural vocoder 없이 **Griffin-Lim**으로 linear
스펙트로그램 → 파형 복원. LLM 엔지니어 관점에서 인코더-디코더 attention은 NMT와 그대로 같고,
차이는 디코더 출력이 discrete token softmax가 아니라 **continuous mel 프레임 회귀**(L1/L2)라는 점.

계획(확정 — conformer식 step 스크립트 + from-scratch 손조립 + 사전학습 합성):
- `modules.py`: CBHG(conv bank·highway·BiGRU)·Bahdanau attention·decoder(attn RNN + decoder RNN, r-frame)·post-net을 from-scratch `nn.Module`로. 스텝 스크립트가 import.
- `common.py`: `get_device()`·seed·`rule()`·한글폰트·`savefig()`·`save_wav()`, toy char 인코더, 논문 하이퍼파라미터 dataclass(embed 256 / K_enc 16 / K_post 8 / mel 80 / r 2 / GRU 256), 실제 오디오 로더(conformer의 librispeech-dummy 재사용).
- `01_char_encoder_cbhg.py`: char embed → prenet → CBHG. conv bank width 1..16·max-pool·conv proj·highway·BiGRU shape 단계별 추적 + 논문값 assert.
- `02_attention_decoder.py`: content-based(Bahdanau) attention + 디코더. 스텝당 r=2 프레임 방출, 디코더 스텝=⌈T_mel/r⌉·attention 행합=1 검증, alignment shape.
- `03_postnet_and_loss.py`: post-net CBHG로 mel(80)→linear spectrogram, L1 손실(mel+linear).
- `04_griffinlim_demo.py`: 실제 오디오 → linear spectrogram → `torchaudio` GriffinLim 복원. `outputs/original.wav`·`griffinlim.wav` + 수렴(10/30/60 iter) + 스펙트로그램 비교 png. (유일하게 '진짜 소리' 나는 코어 스텝)
- `05_pretrained_synth.py`: Coqui TTS로 text→음성 end-to-end. tacotron-DCA 우선, 없으면 tacotron2-DDC 폴백. `outputs/tts_*.wav`. work 우선, dep은 `download.py`로 분리, lazy import·실패 시 로그.
- `run.py`: ctx 1회 빌드 → 01~04 체이닝, 05는 `--pretrained` 또는 Coqui import 가능 시.
- 검증: HF bit-exact 레퍼런스가 없으므로 01~03은 shape·수식·불변식 assert(✅/❌), 04는 복원 SNR·가청 wav, 05는 실제 합성 음성.

## ⚙️ 실행 방법
```bash
cd speech-generation/tacotron1
python <스킬경로>/scripts/setup_env.py home   # 또는 work
python run.py
python download_ttao.py                        # 실제 v1(ttaoREtw) 격리 설치 + 체크포인트(.venv-ttao)
.venv-ttao/bin/python run.py --real            # STEP 6~7: 실제 학습 v1 해부 + alignment 대비
```

## 📊 결과 / 관찰
- 2026-07-19: 01~05 전 스텝 동작 확인(work/MPS). `run.py`로 01~04 관통 OK.
- 01 CBHG 인코더: char embed (1,39,256) → prenet (1,39,128) → conv bank 16×128=2048 → enc_out (1,39,256). shape assert 전부 ✅.
- 02 attention 디코더: toy T_mel=50, r=2 → 디코더 스텝 25=⌈50/2⌉, mels (1,50,80), aligns (1,25,39), attention 행합=1 ✅. random init이라 alignment는 diffuse(`outputs/alignment_randinit.png`).
- 03 post-net: mel(80) → linear (1,50,513), L1 finite ✅.
- 04 Griffin-Lim: 실제 오디오(5.86s@16k) magnitude 복원. spectral convergence 10/30/60 iter = `0.135/0.060/0.039`로 수렴. `outputs/original.wav`·`griffinlim.wav`·`griffinlim_spec.png`.
- 05 사전학습: `tacotron-DCA`(Tacotron1 계열)는 현재 Coqui zoo에 없음(KeyError) → `tacotron2-DDC` 폴백 합성 성공. `outputs/tts_tacotron2-DDC.wav` (4.24s@22k, 실제 음성).
- 2026-07-19(실제 v1 해부 추가): "차원만 보는 건 의미 없다" → 실제 학습된 Tacotron**1**을 로드해 진짜 동작 확인. Coqui엔 v1이 없어(05는 v2 폴백) 웹에서 `ttaoREtw/Tacotron-pytorch`(MIT, LJSpeech, step 138k) 체크포인트 확보. 텐서 구조가 우리 `modules.py`와 동일(CBHG·BahdanauAttn·GRU decoder residual·post-net CBHG→linear·Griffin-Lim) 확인 → 우리 손조립 코드 구조 검증도 됨.
- 06 실제 해부: alignment (1,61,55) **peak 0.71·monotonic 0.97~0.98**(대각선), mel (1,~305,80)·linear (1,~305,1025), `real_v1.wav` 3.79s@22050(Griffin-Lim). `outputs/real_{mel,linear,alignment}.png`. (prenet dropout 유지 추론이라 run마다 값 미세 변동.)
- 07 대비: 같은 문장 alignment — 우리 random **peak 0.019**(diffuse, ≈1/T_enc) vs 학습됨 **peak 0.71**(대각선). "같은 content-based attention, 학습 전 vs 후"가 `outputs/alignment_compare.png`에 한눈에.

## 🧱 막힌 점 / TODO
- `torchaudio.save`는 2.11에서 torchcodec을 요구 → `soundfile`로 저장(`common.save_wav`).
- 05 Coqui 의존성 hell: coqui-tts가 torch 자동설치 안 함 + transformers 5.x(심볼 제거)·torchcodec 충돌. 격리 venv(`.venv-tts`)에 `torch`+`transformers>=4.55,<5`+`torchcodec` 고정으로 해결(`download.py`). ffmpeg 바이너리 없이도 합성·저장됨.
- Tacotron1(v1) 사전학습 체크포인트가 Coqui zoo에 없어 v2(`tacotron2-DDC`)로 대체. 진짜 v1 가중치 필요하면 keithito(구 TF)/다른 소스 탐색.
- home(8GB CUDA)에서 05 미검증(work만). 01~04는 환경 무관.
- (선택) 01~03에 toy alignment overfit을 더해 attention이 monotonic으로 학습되는 걸 눈으로 보는 스텝 추가 여지.
- 실제 v1 로드용 `ttao_ref/` vendoring(MIT, commit `6b0f615`): 추론 파일만 복사(dataset/solver 제외), `utils.py` 2곳 하드 API 패치(`np.complex`→`np.complex128`, `librosa.filters.mel` keyword). 상세 `ttao_ref/PROVENANCE.md`.
- `.venv-ttao` 설치: uv가 옛 `numba`(0.53, py<3.10 전용)를 집어 `llvmlite` 소스빌드 실패 → `numba>=0.60`+`numpy<2.1` 핀으로 프리빌트 휠 유도(해결). 최종 torch 2.13·librosa 0.11·numba 0.66·numpy 2.0.2.
- 이 체크포인트는 v1 구조지만 계층 이름·차원(r=5·linear 1025·vocab 250)이 우리 모듈과 달라 `load_state_dict` 직접 불가 → vendored 코드로 실행(우리 `modules.py` 이식은 fragile 복제라 비채택).

## 📝 메모
- 주의: `torchaudio`/HF 공식 파이프라인엔 Tacotron**2**만 있고 Tacotron1은 없음 → from-scratch
  조립 또는 커뮤니티 구현(keithito/tacotron, Coqui TTS) 참고. conformer/wav2vec2처럼 HF forward와
  bit-exact 비교할 레퍼런스가 없으므로, 검증 기준은 "shape·수식이 논문과 맞는가".
- 계보: Tacotron 2가 이 Griffin-Lim 단계를 WaveNet 뉴럴 보코더로 교체. Griffin-Lim(DSP, 학습 없음)
  ↔ 뉴럴 보코더/코덱 대비로 이해.
- 설계 결정(2026-07-19): 04 Griffin-Lim은 예시 오디오의 native sr(16kHz) 그대로 사용(논문 24kHz로
  리샘플 안 함 — GL 시연엔 무관). random-init 모듈이라 01~03 forward는 acoustic 의미가 없어 shape·불변식만 검증.
- 05 Coqui 분리: `coqui-tts`가 torch 핀을 끌어 core venv를 깨뜨릴 수 있어 `requirements.txt`가 아니라
  `download.py`로 분리 설치. `tacotron-DCA` 미존재 시 `tacotron2-DDC` 폴백(v2지만 seq2seq+attention 동일), 사용 모델·vocoder를 로그.
- 관련: [[speech-representation/wav2vec2]] (표현학습) ↔ 이건 생성(TTS). paper: arXiv 1703.10135.
