"""wav2vec 2.0 step-by-step 실습 공통 유틸.

노션 정리("wav2vec 2.0")의 7단계 메커니즘을 실제 텐서로 따라가기 위한 공유 로더.
모델은 사전학습 SSL 체크포인트 `facebook/wav2vec2-base`(quantizer 포함)를 쓴다.
"""
import pathlib
import sys

import numpy as np

# 레포 루트를 경로에 추가해 shared/env.py를 임포트
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.env import get_device  # noqa: E402

MODEL_ID = "facebook/wav2vec2-base"  # 사전학습(SSL) 체크포인트 — quantizer 가중치 포함
SEED = 0
OUT = pathlib.Path(__file__).resolve().parent / "outputs"

# 노션/논문 하이퍼파라미터
MASK_PROB = 0.065   # 각 frame이 span 시작점이 될 확률 p
MASK_LEN = 10       # span 길이 M


def seed_everything() -> None:
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_sample():
    """(audio: np.float32 1D, sr: int, text: str). 실패 시 합성 사인파로 대체.

    최신 datasets는 Audio 디코딩에 torchcodec을 요구한다. 그 의존을 피하려고
    decode=False로 파일 경로/바이트만 받아 soundfile로 직접 디코딩한다.
    """
    try:
        import io
        import os

        import soundfile as sf
        from datasets import Audio, load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(decode=False))
        item = ds[0]
        a = item["audio"]
        if a.get("path") and os.path.exists(a["path"]):
            audio, sr = sf.read(a["path"])
        else:
            audio, sr = sf.read(io.BytesIO(a["bytes"]))
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:  # 스테레오 → 모노
            audio = audio.mean(axis=1)
        return audio, int(sr), item.get("text", "")
    except Exception as e:  # 오프라인 등
        print(f"[warn] 데이터셋 로드 실패({e}); 합성 사인파로 대체.", file=sys.stderr)
        sr = 16000
        t = np.linspace(0, 3, sr * 3, endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32), sr, "(synthetic sine)"


def load_model():
    from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining

    device = get_device()
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForPreTraining.from_pretrained(MODEL_ID).to(device).eval()
    return fe, model, device


def build_context() -> dict:
    """단계 스크립트가 공유하는 컨텍스트. run.py는 한 번만 만들어 재사용한다."""
    seed_everything()
    fe, model, device = load_model()
    audio, sr, text = load_sample()
    input_values = fe(audio, sampling_rate=sr, return_tensors="pt").input_values.to(device)
    OUT.mkdir(exist_ok=True)
    return {
        "fe": fe, "model": model, "device": device,
        "audio": audio, "sr": sr, "text": text, "input_values": input_values,
    }


def compute_mask(ctx: dict):
    """context 경로에서 가릴 span 마스크를 만든다(모든 단계가 같은 마스크를 공유).

    반환: (mask_bool[T tensor], seq_len). ctx에 mask_bool/mask_long/mask_np/seq_len 저장.
    """
    import torch
    from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

    if "mask_bool" in ctx:
        return ctx["mask_bool"], ctx["seq_len"]

    model, iv = ctx["model"], ctx["input_values"]
    seq_len = int(model._get_feat_extract_output_lengths(iv.shape[1]).item())
    np.random.seed(SEED)
    mask_np = _compute_mask_indices((iv.shape[0], seq_len), MASK_PROB, MASK_LEN, min_masks=2)
    ctx["mask_np"] = mask_np
    ctx["mask_bool"] = torch.tensor(mask_np, dtype=torch.bool, device=iv.device)
    ctx["mask_long"] = torch.tensor(mask_np, dtype=torch.long, device=iv.device)
    ctx["seq_len"] = seq_len
    return ctx["mask_bool"], seq_len


def rule(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
