"""Tacotron1 step-by-step 실습 공통 유틸.

conformer 실습과 같은 골격: build_context()로 device·config·toy 텍스트·실제 오디오를
1회 만들고, 각 스텝(0N_*.py)이 ctx로 공유한다. 단 Tacotron1은 공식 HF 모델이 없어
서브모듈을 from-scratch(modules.py)로 조립하고, 검증은 shape·수식·불변식 assert로 한다.
가중치가 random init이라 01~03 forward는 acoustic 의미가 없고 shape/불변식만 본다.
실제 소리는 04(Griffin-Lim)·05(Coqui 사전학습)에서 난다.
"""
from __future__ import annotations

import io
import os
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

# 레포 루트를 경로에 추가해 shared/env.py를 임포트
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
from shared.env import get_device  # noqa: E402

SEED = 0
OUT = pathlib.Path(__file__).resolve().parent / "outputs"

# 문자 vocab (idx 0 = space, pad 겸용)
VOCAB = list(" abcdefghijklmnopqrstuvwxyz.,!?'-")
TOY_TEXT = "hello world, this is tacotron speaking."


@dataclass
class HP:
    """논문 Table 1/2 하이퍼파라미터."""
    embed: int = 256
    prenet: tuple = (256, 128)
    k_enc: int = 16          # 인코더 CBHG conv bank width 1..16
    k_post: int = 8          # post-net CBHG conv bank width 1..8
    conv_ch: int = 128
    highway: int = 4
    enc_gru: int = 128       # BiGRU → 인코더 출력 dim = 2*128 = 256
    mel: int = 80
    r: int = 2               # reduction factor
    dec_gru: int = 256
    attn_dim: int = 128
    # 신호(04 Griffin-Lim). 예시 오디오 native sr(16k) 사용 — 논문 24k로 리샘플 안 함.
    n_fft: int = 1024
    hop: int = 256
    win: int = 1024
    gl_iter: int = 60
    gl_power: float = 1.2

    @property
    def enc_out(self) -> int:
        return 2 * self.enc_gru

    @property
    def linear_bins(self) -> int:
        return self.n_fft // 2 + 1


def seed_everything() -> None:
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)


def char_encode(text: str):
    """text → LongTensor (1, L). vocab 밖 문자는 space(0)로."""
    import torch

    idx = {c: i for i, c in enumerate(VOCAB)}
    ids = [idx.get(c, 0) for c in text.lower()]
    return torch.tensor([ids], dtype=torch.long)


def load_sample():
    """(audio np.float32 1D, sr, text). conformer/whisper와 동일 로더(librispeech dummy).

    최신 datasets의 torchcodec 의존을 피하려 decode=False로 받아 soundfile로 직접 디코딩.
    """
    try:
        import soundfile as sf
        from datasets import Audio, load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(decode=False))
        a = ds[0]["audio"]
        if a.get("path") and os.path.exists(a["path"]):
            audio, sr = sf.read(a["path"])
        else:
            audio, sr = sf.read(io.BytesIO(a["bytes"]))
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr), ds[0].get("text", "")
    except Exception as e:  # 오프라인 등
        print(f"[warn] 데이터셋 로드 실패({e}); 합성 사인파로 대체.", file=sys.stderr)
        sr = 16000
        t = np.linspace(0, 3, sr * 3, endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32), sr, "(synthetic sine)"


def save_wav(wav, sr: int, name: str) -> None:
    """outputs/ 에 wav 저장. soundfile 사용(torchaudio.save는 2.11에서 torchcodec 필요).

    wav: (N,) mono 또는 (C, N)/(N, C) tensor/ndarray → mono 1D로 저장.
    """
    import soundfile as sf

    if hasattr(wav, "detach"):  # torch.Tensor
        wav = wav.detach().cpu().float().numpy()
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    OUT.mkdir(exist_ok=True)
    sf.write(str(OUT / name), wav, sr)
    print(f"[저장] {OUT / name}")


def build_context() -> dict:
    """스텝이 공유하는 컨텍스트. run.py가 1회 만들어 관통시킨다."""
    seed_everything()
    audio, sr, ref_text = load_sample()
    OUT.mkdir(exist_ok=True)
    return {
        "device": get_device(),
        "hp": HP(),
        "text": TOY_TEXT,
        "char_ids": char_encode(TOY_TEXT),
        "audio": audio,
        "sr": sr,
        "ref_text": ref_text,
    }


def rule(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def use_korean_font() -> None:
    """그림 한글 제목이 안 깨지게 한글 폰트를 등록(있으면). 없으면 조용히 넘어감."""
    try:
        import matplotlib
        from matplotlib import font_manager

        avail = {f.name for f in font_manager.fontManager.ttflist}
        for name in ("AppleGothic", "Apple SD Gothic Neo", "AppleSDGothicNeo",
                     "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"):
            if name in avail:
                matplotlib.rcParams["font.family"] = name
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:  # noqa: BLE001
        pass


def savefig(fig, name: str) -> None:
    try:
        OUT.mkdir(exist_ok=True)
        fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
        print(f"[저장] {OUT / name}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 그림 저장 실패({name}): {e}")


use_korean_font()  # common import 시 1회 적용
