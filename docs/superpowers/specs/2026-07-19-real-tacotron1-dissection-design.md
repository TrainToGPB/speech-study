# 실제 Tacotron1 해부 스텝 설계 (ttaoREtw 로드)

**날짜**: 2026-07-19
**대상 실험**: `speech-generation/tacotron1`
**상태**: 승인됨(사용자) → plan으로 이행

## 배경 / 문제

`tacotron1`의 01~03 스텝은 from-scratch `modules.py`(CBHG·content-attention·post-net)를
**random-init**으로 forward만 태워 shape·불변식만 검증한다. "차원만 보는 건 의미가 없다"는
지적대로, 실제 학습된 가중치로 **진짜 동작**(대각선 alignment·음성 같은 mel·가청 오디오)을 봐야
학습 가치가 크다.

Coqui zoo에는 Tacotron**1** 체크포인트가 없다(v2뿐). 웹 탐색 결과
**ttaoREtw/Tacotron-pytorch**(MIT, Tao Tu 2018)가 "원조 Tacotron에 충실"하고, 텐서 레벨에서
우리 `modules.py`와 **같은 아키텍처**(CBHG conv bank·`BahdanauAttn` content-based·GRU decoder
residual·post-net CBHG→linear·Griffin-Lim)임을 확인했다. 2019년 GDrive 체크포인트(74MB)도
현재 다운로드 가능(검증 완료).

다만 계층 이름·차원(r 5↔2, linear 1025↔513, vocab 250↔32)이 달라 이 가중치를 우리 모듈에
그대로 `load_state_dict`할 수는 없다. 따라서 **ttaoREtw 코드 + 체크포인트로 실제 합성**하고,
그 실제 출력을 **우리 from-scratch 모듈의 random 출력과 나란히** 놓는다("같은 구조, 학습 전 vs 후").

## 목표

실제 학습된 Tacotron1을 로드해 end-to-end로 해부하는 실습 스텝을 추가한다. 산출:
진짜 대각선 alignment, 진짜 mel/linear 스펙트로그램, Griffin-Lim 오디오, 그리고 우리 모듈의
diffuse alignment와의 side-by-side 대비 그림.

**비목표**: 우리 `modules.py`에 가중치를 이식(fragile 복제 대공사) — 대안으로만 언급, 이번 범위 아님.
01~05는 보존(구조 학습 트랙 유지). 학습 루프는 없음(사전학습 로드만).

## 접근

### Vendoring (MIT, attribution 유지)
ttaoREtw(commit `6b0f615`)에서 추론에 필요한 파일만 `tacotron1/ttao_ref/`에 패키지로 복사:
`module.py`(torch-only, 원본), `symbols.py`(원본, `from .text.cleaners`), `utils.py`(2곳 패치),
`text/cleaners.py`·`text/numbers.py`(원본), `config.yaml`(원본, 체크포인트와 일치), `LICENSE`,
`PROVENANCE.md`(출처·commit·패치 내역). `__init__.py`로 패키지화해 상대 import 유지.

**utils.py 하드 API 패치 2곳** (모던 numpy/librosa 대응, PROVENANCE에 명시):
- `np.complex` → `np.complex128` (numpy 1.24+ 제거됨)
- `librosa.filters.mel(sr, n_fft, n_mels=…)` → keyword `mel(sr=…, n_fft=…, n_mels=…)` (librosa 0.10+ keyword-only)

### venv 격리
`.venv-tts`(coqui)와 같은 패턴으로 `.venv-ttao` 분리 — librosa·numba·inflect·unidecode 등이
검증 끝난 core `.venv`(01~04)를 오염시키지 않게. `.gitignore`에 `.venv-ttao/` 추가.
의존성: torch·numpy·librosa·scipy·pyyaml·inflect·unidecode·soundfile·matplotlib.

### 스텝 (01~05 보존, chained 2개 추가)
- `download_ttao.py` — `.venv-ttao` 생성·dep 설치·gdown(id `1q8xLo9zyyclIDgYk3V2mczofnQwqT6pk`)으로
  체크포인트 → `outputs/ttao_tacotron.pth`(gitignore). `download.py` 미러, 실패 시 graceful.
- `06_real_v1_synth.py` — `ttao_ref` 로드, `Tacotron(**cfg)` + `load_state_dict(ckpt['state_dict'])`,
  eval 세팅(encoder·postnet만, mel_decoder는 dropout 유지 — 원저자 추론 방식), `txt2seq(text)` →
  `mel, spec, attn = model(seq)`. `outputs/`에 real mel·linear png, `real_alignment.png`(대각선),
  `real_v1.wav`(AudioProcessor Griffin-Lim). shape + monotonic(대각선) assert.
- `07_alignment_compare.py` — 우리 `modules.py`(torch-only 임포트, `common.py`의 datasets 임포트 회피)를
  같은 텍스트로 random-init 돌려 diffuse alignment 생성 → 06의 real 대각선과 side-by-side
  `outputs/alignment_compare.png`.
- `run.py`에 `--real` 경로(06~07), `LOG.md`·`.gitignore` 갱신.

## 검증
- 06: alignment argmax가 대체로 monotonic(대각선), `real_v1.wav` 가청·finite, mel/linear shape 일치.
- 07: 대비 그림에서 우리 것은 diffuse, ttaoREtw는 대각선.
- 격리이므로 core 01~04는 06 이후에도 그대로 통과.
- HF bit-exact 레퍼런스는 없음 — 검증은 "구조 일치 + 실제 대각선·가청음".

## 레포 변경 요약
신규(커밋): `ttao_ref/`(vendored) · `download_ttao.py` · `06_*.py` · `07_*.py` · spec · plan.
수정: `run.py` · `LOG.md` · `.gitignore`. gitignore: `.venv-ttao/`·체크포인트·wav·png는 제외.
보존: 01~05 그대로.
