"""Stage 1 실습 공통 헬퍼."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "shared" / "sample_audio"   # 샘플 오디오는 스테이지 공통
OUT_DIR = STAGE_DIR / "outputs"                  # 결과물은 스테이지별로 분리
OUT_DIR.mkdir(exist_ok=True)


def sample_path(name: str = "speech_en.wav") -> Path:
    """공통 샘플 오디오 경로. 없으면 안내 후 종료."""
    p = SAMPLE_DIR / name
    if not p.exists():
        raise SystemExit(
            f"[!] 샘플 오디오가 없습니다: {p}\n"
            f"    먼저 실행: python shared/sample_audio/download_samples.py"
        )
    return p


def load_wav(path: Path, target_sr: int = 16000):
    """모노 · target_sr 로 로드해 (waveform[np.float32], sr) 반환."""
    import librosa

    wav, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return wav.astype(np.float32), sr


def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")
