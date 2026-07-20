"""Whisper 인코더 step-by-step 실습 공통 유틸.

목표: Whisper 인코더가 raw waveform을 어떻게 (1500 x d_model) 시퀀스로 바꾸는지
단계별 텐서로 따라간다. 모델은 `openai/whisper-base`(enc 6층, d_model 512, head 8,
mel 80) — 메커니즘이 또렷하고 Mac/MPS에서 가볍다. 더 큰 모델을 봐도 구조는 동일.

인코더 forward(HF)의 순서를 그대로 재현한다:
    gelu(conv1(mel)) → gelu(conv2(·)) → permute → + pos_embed → [encoder_layer]*N → layer_norm

attention_weights를 얻으려면 eager attention이 필요하므로 attn_implementation='eager'로 로드.
"""
import io
import os
import pathlib
import sys

import numpy as np

# 레포 루트를 경로에 추가해 shared/env.py를 임포트
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
from shared.env import get_device  # noqa: E402

MODEL_ID = "openai/whisper-base"  # enc 6층 · d_model 512 · head 8 · mel 80
SEED = 0
OUT = pathlib.Path(__file__).resolve().parent / "outputs"


def seed_everything() -> None:
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_sample():
    """(audio: np.float32 1D, sr: int, text: str). 실패 시 합성 사인파로 대체.

    최신 datasets는 Audio 디코딩에 torchcodec을 요구한다. 그 의존을 피하려고
    decode=False로 파일 경로/바이트만 받아 soundfile로 직접 디코딩한다.
    (wav2vec2 실험과 동일한 예시 오디오·로딩 방식을 재사용)
    """
    try:
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
        t = np.linspace(0, 4, sr * 4, endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32), sr, "(synthetic sine)"


def load_model():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = get_device()
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    # eager attention이라야 output_attentions로 attn_weights를 돌려준다(sdpa는 None).
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, attn_implementation="eager"
    ).to(device).eval()
    return processor, model, device


def build_context() -> dict:
    """단계 스크립트가 공유하는 컨텍스트. run.py는 한 번만 만들어 재사용한다.

    input_features = log-Mel spectrogram (1, num_mel_bins, 3000). Whisper feature extractor가
    오디오를 30초로 pad/trim 하므로 frame 수는 항상 3000(=100Hz)로 고정된다.
    """
    import torch

    seed_everything()
    processor, model, device = load_model()
    audio, sr, text = load_sample()

    feat = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    mel = feat.input_features.to(device)  # (1, 80, 3000)

    enc = model.model.encoder  # WhisperEncoder (conv1/conv2/embed_positions/layers/layer_norm)
    cfg = model.config
    OUT.mkdir(exist_ok=True)
    with torch.no_grad():
        pass
    return {
        "processor": processor, "model": model, "encoder": enc, "config": cfg,
        "device": device, "audio": audio, "sr": sr, "text": text, "mel": mel,
    }


def rule(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def use_korean_font() -> None:
    """그림의 한글 제목이 깨지지 않게 한글 지원 폰트를 등록한다(있으면).

    없으면 조용히 넘어간다(값·shape엔 영향 없음). common을 import하는 순간 적용되므로
    각 단계에서 따로 부를 필요 없다. matplotlib.use('Agg')보다 먼저여도 backend는 안 잠근다.
    """
    try:
        import matplotlib
        from matplotlib import font_manager

        avail = {f.name for f in font_manager.fontManager.ttflist}
        for name in ("AppleGothic", "Apple SD Gothic Neo", "AppleSDGothicNeo",
                     "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"):
            if name in avail:
                matplotlib.rcParams["font.family"] = name
                break
        matplotlib.rcParams["axes.unicode_minus"] = False  # 'ー' 대신 정상 마이너스
    except Exception:  # noqa: BLE001 — 폰트 설정 실패는 치명적이지 않음
        pass


def savefig(fig, name: str) -> None:
    """outputs/ 에 그림 저장(헤드리스 안전)."""
    try:
        fig.savefig(OUT / name, dpi=110)
        print(f"[저장] {OUT / name}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 그림 저장 실패({name}): {e}")


use_korean_font()  # common import 시 1회 적용
