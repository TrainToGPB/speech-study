# Conformer 블록 텐서 단계별 추적 실험 — 설계

- 작성일: 2026-07-18
- 상태: 승인·구현 완료 (work에서 STEP 3·5 bit-exact `max|Δ|=0`, CTC 전사 정답 일치)
- 실험 경로: `speech-recognition/conformer/`
- 논문: [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) (Gulati et al., 2020)

## 배경 & 목표

Conformer 블록이 오디오 feature 시퀀스를 어떻게 처리하는지, 예시 오디오 하나를 잡고
**블록 내부 서브모듈을 순서대로 실제 텐서로 따라간다.** 형제 실험 `speech-recognition/whisper`,
`speech-representation/wav2vec2`의 "step-by-step 텐서 추적" 골격을 그대로 따른다.

핵심 학습 대상 = **Conformer 블록의 macaron 구조**:

```
½·FFN → MHSA(상대위치) → Conv module → ½·FFN → LayerNorm
```

특히 **상대위치(relative-position) multi-head self-attention** 이 이 논문의 시그니처이므로
가장 깊게 판다.

## 확정된 결정

1. **레퍼런스 구현 = HF `Wav2Vec2ConformerForCTC`** (`facebook/wav2vec2-conformer-rel-pos-large-960h-ft`).
   - 이유: 형제 실험들처럼 **HF 모델 1개를 bit-exact로 추적**하는 방식을 그대로 재사용.
     상대위치 MHSA까지 논문에 충실하고, 사전학습 가중치라 **실제 LibriSpeech 전사**가 capstone으로 붙음.
   - 반려한 대안: `torchaudio.models.Conformer`(설치돼 있고 무료·양쪽 환경 동일하지만 vanilla
     `nn.MultiheadAttention` — 상대위치 없음 + 랜덤 가중치라 전사 불가), 하이브리드(코드량↑·모델 2개).
2. **범위 = 블록 1개(`encoder.layers[0]`)를 깊게.** STEP 2~5가 서브모듈 단위로 뜯고 bit-exact 검증,
   STEP 6에서 24층 전체 + 전사로 줌아웃. (전 층 얕게 훑는 대안은 서브모듈 디테일이 사라져 반려.)
3. **타깃 환경 = work**(MacBook Pro M5 32GB · MPS, 이미 셋업 완료). home 8GB는 caveat로 기록.
4. **시각화 2종** 모두 포함 (whisper의 viz 2개 카운트에 맞춤).

## 대상 모델 구조 (transformers 4.46.3 소스에서 확인)

- 로드: `Wav2Vec2ConformerForCTC.from_pretrained(MODEL_ID, attn_implementation="eager")`
  - conformer 인코더: `model.wav2vec2_conformer.encoder`
  - 프런트엔드: `model.wav2vec2_conformer.feature_extractor`(CNN) + `.feature_projection`
  - CTC 헤드: `model.lm_head` + `Wav2Vec2Processor`로 디코딩 → 전사
- config는 하드코딩하지 않고 런타임에 읽음 (≈ hidden 1024 · head 16 · 24층 · FFN(intermediate) 4096 ·
  depthwise kernel — 코드가 `model.config`에서 읽어 출력).
- `encoder.embed_positions` = `Wav2Vec2ConformerRelPositionalEmbedding` (position_embeddings_type="relative").
  인코더가 rel-pos 임베딩을 **한 번** 계산해 **모든 layer의 `self_attn`에 전달**. 각 층 뒤 `encoder.layer_norm`.

### `Wav2Vec2ConformerEncoderLayer` 서브모듈
`ffn1_layer_norm`, `ffn1`, `self_attn_layer_norm`, `self_attn_dropout`, `self_attn`,
`conv_module`, `ffn2_layer_norm`, `ffn2`, `final_layer_norm`.

### 블록 forward (소스에서 확인한 정확한 순서)
```
residual = x;  x = ffn1_layer_norm(x); x = ffn1(x);          x = x * 0.5 + residual   # ½ FFN
residual = x;  x = self_attn_layer_norm(x); x, attn = self_attn(x, rel_pos_emb)
               x = self_attn_dropout(x);                     x = x + residual         # MHSA
residual = x;  x = conv_module(x);                           x = residual + x         # Conv (내부 LN 있음)
residual = x;  x = ffn2_layer_norm(x); x = ffn2(x);          x = x * 0.5 + residual   # ½ FFN
               x = final_layer_norm(x)
```

### 서브모듈 내부
- **`Wav2Vec2ConformerFeedForward`**: `intermediate_dense`(1024→4096) → act(`hidden_act`=swish/SiLU)
  → dropout → `output_dense`(4096→1024) → dropout.
- **`Wav2Vec2ConformerConvolutionModule`**: `layer_norm` → transpose → `pointwise_conv1`(1024→2048)
  → `glu`(dim=1, 채널 절반) → `depthwise_conv`(groups=1024, kernel k) → `batch_norm` → `activation`(SiLU)
  → `pointwise_conv2`(→1024) → dropout → transpose.
- **`Wav2Vec2ConformerSelfAttention`** (Transformer-XL 상대위치): `linear_q/k/v/out`, `linear_pos`(위치 임베딩용),
  학습 파라미터 `pos_bias_u`·`pos_bias_v`(content bias vs position bias). `matrix_ac`(content) + `matrix_bd`(position,
  rel-shift 트릭 적용) → attention score.

## 스텝 구성 (6스텝, whisper식 선형 체인)

| 스텝 | 파일 | 추적 |
|---|---|---|
| 1 | `01_frontend.py` | `build_context`가 만든 front-end 출력 `(1, T, 1024)`와 rel-pos 임베딩 텐서를 **확인·설명**(waveform → feature_extractor + feature_projection). 실제 forward·rel-pos 계산은 `build_context`에서 1회 수행해 전 스텝이 공유. 프런트엔드 상세는 wav2vec2 담당이라 짧게 + 상호링크. |
| 2 | `02_ffn_macaron1.py` | **½ FFN #1**: LN → Linear(1024→4096) → SiLU → Linear(→1024), residual **×0.5**. macaron 반쪽 residual 관찰. |
| 3 | `03_mhsa_relpos.py` | **상대위치 MHSA ★**: q/k/v, `pos_bias_u`/`pos_bias_v`, `matrix_ac`+`matrix_bd`, rel-shift → attention. residual. |
| 4 | `04_conv_module.py` | **Conv module**: LN → PWConv(→2048) → GLU → depthwise(groups=1024) → BatchNorm → SiLU → PWConv → residual. GLU 채널 절반 + depthwise 국소 수용영역. |
| 5 | `05_ffn2_and_verify.py` | **½ FFN #2 → final LN** → 블록 출력. 그다음 **`torch.allclose(manual, layer(x, rel_pos), atol=1e-4)`** → `일치 ✅ / 불일치 ❌`. |
| 6 | `06_stack_and_transcribe.py` | Capstone: 24층 인코더 전체 + CTC → **실제 전사**. 층 지날수록 residual ‖h‖ 증가 관찰. |

각 스텝은 자신이 소비하는 중간 텐서를 `ctx`에 다시 저장하고, 필요한 선행 결과가 없으면
whisper식 backward-import(`importlib.import_module("0N_prev").main(ctx)`)로 채운다.

## 파일 구성 & 컨벤션 (whisper 관용구 복사)

- **`common.py`**: `MODEL_ID`, `SEED=0`, `seed_everything`(np+torch), `load_sample`(LibriSpeech dummy,
  `decode=False`+soundfile, 사인파 폴백), `load_model`(`Wav2Vec2ConformerForCTC`, `attn_implementation="eager"`,
  `.eval()`), `build_context`(프런트엔드 → `(1,T,1024)` + rel-pos emb + `layer=encoder.layers[0]` + `LAYER_IDX=0`),
  `rule()`(`"="*72`), `savefig`, `use_korean_font`(import 시 1회). 레포 루트 shim으로 `shared.env.get_device` 임포트.
- **스텝 스켈레톤**: `def main(ctx=None)->dict` / 첫 줄 `ctx = ctx or build_context()` / 모든 forward `torch.no_grad()` /
  한글 라벨 정렬 print(`tuple(x.shape)`, `{:.2e}` 등) / matplotlib은 try/except 안 lazy import + `matplotlib.use("Agg")` /
  끝에 `if __name__=="__main__": main()`.
- **`run.py`**: `build_context()` 1회 → `STEPS` 리스트를 `importlib`로 관통하며 `ctx` 스레딩. 배너에 device·오디오 길이·전사·모델명.
- **`download.py`**: `load_sample()` + `Wav2Vec2Processor`·`Wav2Vec2ConformerForCTC` `.from_pretrained(MODEL_ID)`로
  HF 캐시 워밍(약 600M 파라미터 다운로드). 실제 파일은 캐시에만, git 제외.
- **`requirements.txt`**: whisper와 동일 —
  ```
  transformers==4.46.3   # home(CUDA cu121+torch2.5) 안전 핀. .bin 체크포인트 로드 가드 회피.
  datasets               # 예시 오디오(LibriSpeech dummy)
  soundfile              # flac 디코딩
  matplotlib             # 단계별 시각화 → outputs/*.png
  ```

## 시각화 (2종)

- **`viz_relpos_attention.py`** (STEP 3 심화, whisper `viz_attention.py` 대응):
  head별 상대위치 attention + **content(`matrix_ac`) vs position(`matrix_bd`) 분해** 비교 →
  절대위치 대비 상대위치가 더하는 것을 가시화. CLI 인자 `[layer_idx=0] [gamma=0.4]`, `PowerNorm`, 발화구간 crop.
  산출: `outputs/viz_relpos_attention_L{n}_*.png`.
- **`viz_conv_module.py`** (STEP 4 심화): GLU gate 값 분포 + depthwise conv 수용영역(kernel 폭) 시각화.
  conv module이 국소 시간 구조를 어떻게 잡는지. 산출: `outputs/viz_conv_module.png`.

viz 파일은 `main()->None`, 자체 `build_context()`, matplotlib top-level import(try/except 없이), `rule("VIZ · ...")` 접두.

## 검증

- STEP 5에서 손으로 재조립한 블록 출력 vs `encoder.layers[0].forward(x, relative_position_embeddings=rel_pos)`를
  `torch.allclose(atol=1e-4)`로 대조, `max|Δ|`를 `{:.2e}`로 출력. whisper STEP 5와 동일한 pass/fail 표기.
- STEP 6에서 CTC 전사가 정답(LibriSpeech dummy 첫 샘플)과 대소문자·문장부호 빼고 일치하는지 확인.

## 환경

- **work**(타깃): torch 2.13.0 · torchaudio 2.11.0 · transformers 4.46.3 · Python 3.12 · MPS. 이미 셋업 완료.
- **home 8GB**: ~600M 모델(fp32 ≈ 2.4GB) → 짧은 클립 forward는 돌 것으로 예상하나 무거움.
  실패 시 fp16 로드나 CPU 폴백 필요. 관례대로 `LOG.md`의 🧱 막힌 점에 기록하고, 확인 전엔 단정하지 않음.

## 산출물

- `speech-recognition/conformer/` 아래: `common.py`, `01_frontend.py` ~ `06_stack_and_transcribe.py`,
  `viz_relpos_attention.py`, `viz_conv_module.py`, `run.py`, `download.py`(갱신), `requirements.txt`(갱신), `LOG.md`(본문 채움).
- `run.py` 완주 시 `outputs/*.png` 다수 + STEP별 텐서 로그. `LOG.md` `status: wip → done` 후보, `env_tested: [work]`.

## 미해결 / 리스크

- **모델 크기**: rel-pos conformer는 large만 존재(base 없음). home 8GB에서의 실행 가능 여부는 실측 필요.
- **체크포인트 포맷(.bin vs safetensors)**: 다운로드 후 확인. `.bin`이면 `transformers==4.46.3` 핀이 home(torch<2.6)에서 필수.
- **`-960h-ft` 가용성**: HF hub에 없으면 SSL 베이스 `facebook/wav2vec2-conformer-rel-pos-large`로 폴백(전사 capstone은 생략, 블록 추적은 동일).
- **rel-shift 트릭 재현**: STEP 3에서 HF `_apply_relative_embeddings`의 shift 로직을 손으로 재현할 때 인덱싱 실수 위험 → STEP 5 allclose가 이를 잡아줌.
