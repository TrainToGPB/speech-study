# ttao_ref — vendored 출처

이 디렉터리는 실제 학습된 Tacotron1 체크포인트를 로드·추론하기 위해 외부 구현을
**vendoring**한 것이다. 우리 `../modules.py`(from-scratch)와 별개이며, 06/07 스텝에서만 쓴다.

- **출처**: https://github.com/ttaoREtw/Tacotron-pytorch
- **commit**: `6b0f615cafb0530370631a880aac5736fe9a2c64`
- **라이선스**: MIT (Copyright (c) 2018 Tao Tu) — `LICENSE` 원본 동봉.
- **체크포인트**: 원저자 GDrive `1q8xLo9zyyclIDgYk3V2mczofnQwqT6pk` (LJSpeech, 74MB).
  코드에 URL/로더만 두고 실제 파일은 `../download_ttao.py`로 받아 `../outputs/`(gitignore)에 둔다.

## 포함(추론에 필요한 것만)
- `module.py` — `Tacotron`(CBHG 인코더·`BahdanauAttn` content-based attention·GRU decoder·post-net CBHG→linear). torch만 의존.
- `symbols.py` — `txt2seq`(문자→id). `from .text.cleaners import english_cleaners`.
- `text/cleaners.py`, `text/numbers.py` — 영어 텍스트 정규화(`unidecode`·`inflect` 의존).
- `utils.py` — `AudioProcessor`(Griffin-Lim `inv_spectrogram`, `save_wav`).
- `config.yaml` — 체크포인트와 일치하는 하이퍼파라미터(r=5, linear_size=1025, sr=22050 등).

**제외**: `dataset.py`·`solver.py`(학습 전용).

## 원본 대비 수정
패키지화를 위해 `__init__.py`(이 디렉터리 + `text/`)를 추가했고, `utils.py`에 모던
numpy/librosa 대응 하드 API 패치 **2건**만 적용(그 외 코드는 원본 유지):

1. `_griffin_lim`: `np.abs(S).astype(np.complex)` → `np.abs(S).astype(np.complex128)`
   (numpy 1.24+에서 `np.complex` 별칭 제거됨)
2. `AudioProcessor.__init__`: `librosa.filters.mel(self.sr, self.n_fft, n_mels=...)` →
   `librosa.filters.mel(sr=..., n_fft=..., n_mels=...)` (librosa 0.10+ keyword-only 시그니처)
