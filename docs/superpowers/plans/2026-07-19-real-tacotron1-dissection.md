# 실제 Tacotron1 해부 스텝 구현 Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. 스텝은 `- [ ]` 체크박스로 추적.

**Goal:** 실제 학습된 Tacotron1(ttaoREtw, MIT)을 로드해 진짜 alignment·mel·오디오를 뽑고, 우리 from-scratch 모듈의 random 출력과 대비하는 실습 스텝(06·07)을 추가한다.

**Architecture:** ttaoREtw 추론 코드(`module.py`·`symbols.py`·`utils.py`·`text/`)를 `tacotron1/ttao_ref/` 패키지로 vendoring. 격리 venv `.venv-ttao`에 dep 설치 + gdown 체크포인트. 06이 실제 합성·해부, 07이 우리 모듈 대비. 01~05 보존.

**Tech Stack:** PyTorch, librosa(Griffin-Lim), gdown, uv. device는 CPU/MPS(경량).

## Global Constraints
- 대상 디렉터리: `speech-generation/tacotron1/`. 레포 루트: `speech-study/sanddollar/`.
- vendored 코드는 MIT 유지 — `LICENSE` + `PROVENANCE.md`(source·commit `6b0f615`·패치내역) 동봉. 원본 최대 보존, 하드 API 파손만 수정.
- 격리: core `.venv`(01~04 검증됨)를 절대 건드리지 않는다. 모든 신규 dep은 `.venv-ttao`로.
- 대용량(체크포인트 74MB·wav·png)은 커밋 금지(gitignore). vendored 코드·스크립트만 커밋.
- `python` 아님 `.venv-*/bin/python` 사용(Mac). uv로 venv·설치.
- 체크포인트 GDrive id: `1q8xLo9zyyclIDgYk3V2mczofnQwqT6pk`. config는 vendored `ttao_ref/config.yaml`.

---

### Task 1: ttao_ref/ vendoring

**Files:**
- Create: `tacotron1/ttao_ref/{__init__.py, module.py, symbols.py, utils.py, config.yaml, LICENSE, PROVENANCE.md}`
- Create: `tacotron1/ttao_ref/text/{__init__.py, cleaners.py, numbers.py}`
- 소스: scratchpad clone `Tacotron-pytorch/`(commit 6b0f615)

- [ ] `module.py`·`symbols.py`·`text/cleaners.py`·`text/numbers.py`·`config.yaml`·`LICENSE`를 원본 그대로 복사.
- [ ] `ttao_ref/__init__.py`·`ttao_ref/text/__init__.py` 빈 파일 생성(패키지화 — 상대 import `from .text.cleaners` 유지).
- [ ] `utils.py` 복사 후 2곳 패치:
  - `S_complex = np.abs(S).astype(np.complex)` → `.astype(np.complex128)`
  - `librosa.filters.mel(self.sr, self.n_fft, n_mels=self.n_mels)` → `librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels)`
- [ ] `PROVENANCE.md` 작성: 출처 URL·commit·"추론 파일만 vendoring(dataset/solver 제외)"·utils.py 패치 2건 명시.
- [ ] 검증: `.venv` 없이도 `python3 -c "import ast; [ast.parse(open(f).read()) for f in (...)]"`로 문법 파싱 통과(런타임은 Task 3에서).

### Task 2: .gitignore + LOG 자리

**Files:** Modify `.gitignore`(레포 루트), `tacotron1/LOG.md`

- [ ] `.gitignore`에 `.venv-ttao/`·`**/.venv-ttao/` 추가(기존 `.venv-tts` 블록 옆).
- [ ] LOG.md `updated` 2026-07-19 유지, 본문에 이 작업 착수 한 줄(결과는 Task 6에서 채움).

### Task 3: .venv-ttao + download_ttao.py

**Files:** Create `tacotron1/download_ttao.py`

- [ ] `download_ttao.py` 작성: `uv venv --python 3.12 .venv-ttao` → `uv pip install`로 `torch torchaudio numpy librosa scipy pyyaml inflect unidecode soundfile matplotlib gdown` → gdown으로 체크포인트 → `outputs/ttao_tacotron.pth`. 실패 시 graceful 로그(`download.py` 미러).
- [ ] 실행: `python3 download_ttao.py`.
- [ ] 검증: `.venv-ttao/bin/python -c "import torch,librosa,yaml,inflect,unidecode,soundfile"` OK, `outputs/ttao_tacotron.pth` 존재(74MB).
  - (이미 scratchpad에 받아둔 체크포인트가 있으면 gdown 대신 복사로 시간 단축 가능 — 스크립트는 gdown 경로 유지.)

### Task 4: 06_real_v1_synth.py

**Files:** Create `tacotron1/06_real_v1_synth.py`

**Interfaces:**
- Consumes: `ttao_ref.module.Tacotron`, `ttao_ref.symbols.txt2seq`, `ttao_ref.utils.AudioProcessor`, `ttao_ref/config.yaml`, `outputs/ttao_tacotron.pth`.
- Produces: `main(ctx=None)` returns ctx with `real_attn`(np, dec×enc); `outputs/real_mel.png`·`real_linear.png`·`real_alignment.png`·`real_v1.wav`.

- [ ] `sys.path`에 `HERE` 추가, `import yaml`로 `ttao_ref/config.yaml` 로드. `Tacotron(**cfg['model']['tacotron'])`, `torch.load(ckpt, map_location='cpu', weights_only=False)`, `load_state_dict(ckpt['state_dict'])`. eval: `model.encoder.eval(); model.postnet.eval()`(mel_decoder는 원저자대로 dropout 유지).
- [ ] `TEXT="Tacotron turns characters directly into a spectrogram."`, `seq = torch.from_numpy(np.asarray(txt2seq(TEXT))).unsqueeze(0)`, `with torch.no_grad(): mel, spec, attn = model(seq)`.
- [ ] `AudioProcessor(**cfg['audio'])`로 `wav = ap.inv_spectrogram(spec[0].numpy().T)`; `ap.save_wav(wav, outputs/real_v1.wav)`.
- [ ] 저장: mel/linear는 imshow png, `attn[0]`은 대각선 heatmap(`real_alignment.png`). print shape + assert: `spec.shape[-1]==1025`, alignment argmax의 단조 증가 비율 계산(대각선 지표) 출력, wav finite·비영.
- [ ] 실행: `.venv-ttao/bin/python 06_real_v1_synth.py`. 검증: 대각선 alignment·가청 wav·shape 통과.

### Task 5: 07_alignment_compare.py

**Files:** Create `tacotron1/07_alignment_compare.py`

**Interfaces:** Consumes `modules.py`(Encoder/CBHG·BahdanauAttention·Decoder — torch-only), 06의 real alignment(재계산 또는 ctx). Produces `outputs/alignment_compare.png`.

- [ ] `common.py`를 import하지 않고(→ datasets 의존 회피), 로컬 `VOCAB`·char→id로 우리 `modules.py`를 same TEXT에 random-init 돌려 diffuse alignment 획득. (필요 최소 모듈만 `from modules import ...`.)
- [ ] 06을 함수로 재호출하거나 ctx로 real alignment 확보 → `matplotlib` 1×2 subplot: 좌 "ours (random init)" diffuse, 우 "ttaoREtw (trained)" 대각선. `outputs/alignment_compare.png` 저장.
- [ ] 실행: `.venv-ttao/bin/python 07_alignment_compare.py`. 검증: png 생성, 좌 diffuse·우 대각선 육안 확인.

### Task 6: run.py·LOG 갱신 + 전체 검증

**Files:** Modify `tacotron1/run.py`, `tacotron1/LOG.md`

- [ ] `run.py`: `--real` 플래그 → 06·07 순차 실행(같은 `.venv-ttao`에서). 기본 01~04는 기존대로. 도움말에 "`--real`은 `.venv-ttao`로 실행" 명시.
- [ ] core 01~04를 `.venv/bin/python run.py`로 재실행 → 여전히 통과 확인(격리 무결성).
- [ ] `LOG.md` 📊 결과에 06/07 관찰(대각선 alignment·real_v1.wav·대비 그림), 🧱 막힌점에 utils.py 패치 2건·`.venv-ttao` 격리·GDrive 링크 생존 기록. `env_tested`는 work 유지.
- [ ] `python3 <스킬>/scripts/build_index.py`로 INDEX 갱신(필요 시).

### Task 7: 마무리(finishing-a-development-branch)

- [ ] 스코프 확인: `git add -n`으로 `ttao_ref/`·`download_ttao.py`·`06_*`·`07_*`·`run.py`·`LOG.md`·`.gitignore`·spec·plan만(체크포인트/wav/png/venv 제외, voxcpm/whisper 무관 확인).
- [ ] finishing-a-development-branch 스킬로 옵션 제시(merge/PR/keep/discard) — push는 사용자 확인 후.
