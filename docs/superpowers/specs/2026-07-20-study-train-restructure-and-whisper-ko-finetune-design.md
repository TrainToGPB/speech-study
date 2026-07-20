# study/train 재구조화 + Whisper 한국어 미니 파인튜닝 — 설계

**날짜**: 2026-07-20
**브랜치**: `bard-channel/curriculum-restructure-papers`
**범위**: (A) 레포 재구조화, (B) 첫 `train/` 실험(Whisper 한국어 full fine-tune)

---

## 왜

지금까지 레포는 "논문 1개 = 실험 폴더 1개"로 **메커니즘 해부**만 담았다(tacotron1·voxcpm·conformer·whisper·wav2vec2). 이제 **학습/전처리 실무 실습**(모델을 실제로 파인튜닝, 데이터 파이프라인 구성)이라는 성격이 다른 작업이 들어온다. 두 성격을 최상위에서 분리한다:

- `study/` — 논문·메커니즘 해부 (기존 것 이동)
- `train/` — 학습·전처리 실무 실습 (신규)

동시에 `study/` 하위 경로에서 `speech-` prefix를 떼어 가독성을 높인다.

첫 `train/` 실험은 **회사 M5(32GB, MPS) 환경에서 GPU 서버 없이** 굴러가는 미니 재현이어야 한다: 기존 multilingual Whisper를 한국어에 조금 더 맞게 파인튜닝하고, **CER이 실제로 떨어지는 것**을 눈으로 확인한다.

---

## 결정된 구조

모든 실험은 **`<section>/<topic>/<name>` 3-depth로 통일**한다(study·train 대칭 → 툴링 한 경로).

```
sanddollar/
├── shared/env.py                         # 변경 없음
├── INDEX.md                              # 재생성 (구분 컬럼 추가)
├── README.md                            # 재작성 (study/·train/ 반영)
├── study/
│   ├── generation/{tacotron1, voxcpm}    # was speech-generation
│   ├── recognition/{conformer, whisper}  # was speech-recognition
│   └── representation/{wav2vec2}         # was speech-representation
└── train/
    └── recognition/whisper-finetune-ko/  # Part B (신규)
```

깊이 규칙: `root/study/recognition/whisper/common.py` → `parents[3] == repo root`. 모든 실험이 동일하므로 `parents[2]`(기존, 2-depth 가정) → **`parents[3]`**.

---

## Part A — 재구조화

### A1. 디렉터리 이동 (git 히스토리 보존)

- `speech-generation` → `study/generation`, `speech-recognition` → `study/recognition`, `speech-representation` → `study/representation`.
- 방법: 파일시스템 `mkdir study && git mv <old>/<topic> study/<topic>` 로 **tracked 파일만** 이동(git rename 감지로 히스토리 보존). `outputs/`(git 제외 PNG)는 물리적으로 함께 옮겨 보존한다.
- **venv 처리**: 4개 `.venv/`(tacotron1·voxcpm·conformer·whisper)는 절대경로가 박혀 있어 이동하면 어차피 깨지고, git 제외·비엄격 재현 대상이다 → **삭제**. 다음에 해당 실험을 열 때 `setup_env.py work`로 재생성. (wav2vec2엔 venv 없음.)

### A2. import 깊이 수정 `parents[2]` → `parents[3]`

대상 5개(이동 후 경로):
- `study/generation/tacotron1/common.py`
- `study/generation/voxcpm/run.py`
- `study/recognition/conformer/common.py`
- `study/recognition/whisper/common.py`
- `study/representation/wav2vec2/common.py`

번호 스텝 스크립트(`01_*.py` 등)는 같은 폴더의 `common`을 import하므로 `parents[]`를 직접 쓰지 않음(수정 불필요). `shared/env.py` 도크스트링 예시의 `parents[2]`도 정확성을 위해 `parents[3]`로 갱신.

### A3. 스킬 스크립트 재작성

**`build_index.py`** — 현재 `root → topic → exp`(2-depth) 스캔. `root → section → topic → exp`(3-depth)로 변경:
- `section` ∈ 최상위 디렉터리(IGNORE 제외)의 하위 = topic, 그 하위 = exp. `LOG.md` 있으면 row.
- row에 `section` 필드 추가. `topic`은 frontmatter `topic:` 우선, 없으면 경로의 topic 요소.
- **유형(kind)**: `section == "train"` → `🛠 실습`; 아니면 기존 로직(`idea` in fm → `💡 아이디어`, else `📄 논문`).
- INDEX 렌더에 **`구분`(study/train) 컬럼** 추가. 정렬 키 `(section, topic, name)`.
- `rel` 경로는 `root` 기준(`study/recognition/whisper/LOG.md`)이라 자동으로 맞음.

**`new_experiment.py`** — 3-part 경로로 변경:
- `path`를 `<section>/<topic>/<name>`로 파싱, `len(parts) != 3` 오류.
- `RUN_STUB`의 `parents[2]` → `parents[3]`.
- "다음 단계" 안내·`cd` 경로를 full path로.

**`assets/LOG_template.md`** — `cd {{TOPIC}}/{{NAME}}` → `cd {{PATH}}`(= `<section>/<topic>/<name>`). `new_experiment.py`가 `{{PATH}}`·`{{SECTION}}`을 치환하도록 추가. `topic: {{TOPIC}}` 유지.

### A4. 문서·텍스트 갱신

- **`SKILL.md`**: 컨벤션 트리(study/·train/), 명령 예시(`new_experiment.py study/recognition/whisper ...`), 경로 arity(3-part), sandbox 언급을 `<section>` 체계로.
- **`README.md`**: 구조 트리·예시 커맨드 재작성.
- **5개 LOG.md**: frontmatter `topic: speech-generation` → `topic: generation`(prefix 제거, recognition/representation 동일); 본문 `cd speech-generation/tacotron1` → `cd study/generation/tacotron1`; wiki-link `[[speech-representation/wav2vec2]]` → `[[study/representation/wav2vec2]]`(모든 cross-ref, 미존재 `hubert` 포함).
- **`study/recognition/conformer/01_frontend.py`**: 도크스트링 wiki-link 갱신.
- **INDEX.md**: `build_index.py` 재실행으로 재생성(직접 편집 금지).

### A5. 범위 밖 (건드리지 않음)

- `docs/superpowers/plans/`·`specs/`의 기존 dated 문서: 작성 시점 기록이므로 옛 경로 유지.
- `.gitignore`: `**/.venv/`·`**/outputs/` glob이 깊이 무관 → 수정 불필요.
- `setup_env.py`: venv를 cwd에서 찾고 root는 `git rev-parse`로 탐지 → 수정 불필요.

### A6. 검증

1. 이동 후 각 실험에서 `python -c "import sys,pathlib; sys.path.append(str(pathlib.Path('study/recognition/whisper/common.py').resolve().parents[3])); import shared.env"` 성격의 import 성공.
2. `python .claude/skills/speech-study/scripts/build_index.py` → INDEX에 **6개 실험**(기존 5 + 신규 1), 구분·topic 정확.
3. `git status`에 rename으로 잡히고 히스토리 보존(`git log --follow`).

---

## Part B — `train/recognition/whisper-finetune-ko`

### B1. 무엇을 / 규모

- **모델**: `openai/whisper-base` (multilingual, 이미 한국어 약간 앎 → 개선폭이 극적으로 보임, 기존 whisper 실험과 같은 체크포인트).
- **데이터**: **Zeroth-Korean** (`Bingsu/zeroth-korean`, FLAC, CC-BY-4.0 — Stage-0 노션 한국어 목록에 있음). train **소량 subset**(수백~1천 발화), eval 별도 held-out(수십~수백).
- **방식**: full fine-tune. **지표: CER**(주), WER(참고). 한국어는 띄어쓰기 때문에 WER이 노이즈 큼 → CER이 표준.
- **런타임 목표**: M5/MPS에서 20~40분. `max_steps` 소량으로 고정.

### B2. 파일 (기존 step-by-step·개별 실행 컨벤션 준수)

- **`common.py`** — `import` 시 `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"`; `get_device()`(parents[3]); `MODEL_ID`, `LANG="korean"`, `TASK="transcribe"`, `SEED`; `load_processor()`, `load_model()`; `load_splits(n_train, n_eval)`(Zeroth subset, 16k 보장); `compute_cer(refs,hyps)`(evaluate/jiwer); `DataCollatorSpeechSeq2SeqWithPadding`(feature pad + label pad, `-100` 마스크, BOS 트림); `OUT`, `CKPT_DIR = outputs/whisper-base-ko`.
- **`download.py`** — whisper-base + Zeroth subset + CER metric warm-up(HF 캐시).
- **`01_data.py`** — 한 샘플 end-to-end: raw(sr·길이·waveform) → 16k → `feature_extractor` → log-Mel `(1,80,3000)`; `text` → `tokenizer` label ids → decode 복원. **CER vs WER(한국어)** 설명 출력. LLM tokenization과의 대응 코멘트.
- **`02_baseline.py`** — 사전학습 base로 eval subset 전사(한국어 강제) → **baseline CER**. (정답, 예측) 몇 쌍 출력해 "얼마나 틀리는지" 확인.
- **`03_train.py`** — collator + `Seq2SeqTrainingArguments`(MPS-safe: `fp16=False`, `bf16=False`, `dataloader_pin_memory=False`, `dataloader_num_workers=0`, `remove_unused_columns=False`, `predict_with_generate=True`, `generation_max_length`, `per_device_train_batch_size=8`, `gradient_accumulation_steps`, `learning_rate=1e-5`, `warmup_steps`, `max_steps`(소량), `eval_strategy="steps"`, `logging_steps`, `report_to="none"`, `output_dir=CKPT_DIR`) + `Seq2SeqTrainer`; 학습 후 model·processor 저장. 모델 `generation_config.language/task` 설정, `forced_decoder_ids=None`, `suppress_tokens=[]`.
- **`04_evaluate.py`** — `CKPT_DIR`에서 파인튜닝 모델 로드 → **CER after** + before/after Δ + `(정답 / base / finetuned)` triple 출력 (capstone).
- **`run.py`** — 01→02→03→04 순차 오케스트레이션; 각 스텝은 단독 실행도 가능(02·04는 캐시/CKPT에서 로드).
- **`LOG.md`** — frontmatter `topic: recognition`, kind=실습, `env_tested: []`(실행 후 `[work]`), `status: planned`→실행 후 `wip`. 목표·실행법 채우고 결과는 실행 후 기록.

### B3. 의존성 (`requirements.txt`)

`transformers==4.46.3` · `datasets>=3.0,<4`(Zeroth FLAC를 soundfile로 디코딩 → torchcodec 회피, whisper LOG의 함정 반영) · `evaluate` · `jiwer` · `accelerate`(Trainer 필수) · `soundfile`. torch/torchaudio는 `setup_env.py work`가 설치.

### B4. MPS 리스크 & 대응

- fp16 미지원 → fp32 학습(느리지만 안전). bf16도 off(안전 우선).
- 일부 op MPS 미지원 가능 → `PYTORCH_ENABLE_MPS_FALLBACK=1`로 CPU 폴백.
- Mac 멀티프로세싱 + HF Audio 디코딩 hang 방지 → `num_workers=0`, `pin_memory=False`.
- 첫 실행 OOM/속도 이슈 시 `batch_size`·`max_steps` 조정, 안 되면 `whisper-tiny` 폴백(LOG 막힌 점에 기록).

### B5. 성공 기준

`04_evaluate.py`에서 **fine-tuned CER < baseline CER**(개선폭이 눈에 보임), 전 과정이 M5/MPS에서 완주, 예시 triple에서 한국어 전사가 실제로 나아진 게 보임.

---

## 구현 순서(상위)

1. Part A 이동 + 깊이 수정 + 스킬 스크립트 재작성 + 문서/LOG 갱신 → `build_index` → 검증 → 커밋.
2. Part B 스캐폴드(`new_experiment.py train/recognition/whisper-finetune-ko`) → 파일 작성 → `setup_env.py work` → `download.py` → 01~04 실행 → LOG 결과 기록 → 커밋.

세부 단계는 writing-plans로 별도 계획서에 전개한다.
