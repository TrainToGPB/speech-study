"""Conformer 블록 step-by-step 실습 공통 유틸.

목표: Conformer 블록이 오디오 feature 시퀀스를 어떻게 처리하는지 서브모듈 단위로
텐서를 따라간다. 모델은 `facebook/wav2vec2-conformer-rel-pos-large-960h-ft`
(hidden 1024 · head 16 · 24층 · rel-pos MHSA) — 상대위치 attention이 논문 시그니처.
CTC 헤드가 붙어 있어 실제 LibriSpeech 전사까지 capstone으로 확인한다.

블록 forward(HF Wav2Vec2ConformerEncoderLayer)의 macaron 순서를 그대로 재현한다:
    x = ffn1(LN(x))*0.5 + x            # ½ FFN
    x = dropout(self_attn(LN(x), rel_pos)) + x   # rel-pos MHSA
    x = conv_module(x) + x             # Conv module (내부 LN)
    x = ffn2(LN(x))*0.5 + x            # ½ FFN
    x = final_LN(x)

front-end(feature_extractor+feature_projection)와 rel-pos 임베딩은 여기서 1회 계산해
전 스텝이 ctx로 공유한다. self_attn은 probs를 항상 돌려주므로 eager 지정이 필요없다.
"""
import io
import os
import pathlib
import sys

import numpy as np

# 레포 루트를 경로에 추가해 shared/env.py를 임포트
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.env import get_device  # noqa: E402

MODEL_ID = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"  # rel-pos · CTC(960h finetune)
SEED = 0
OUT = pathlib.Path(__file__).resolve().parent / "outputs"
LAYER_IDX = 0  # 깊게 뜯어볼 블록 인덱스 (0 = 첫 conformer 층)


def seed_everything() -> None:
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_sample():
    """(audio: np.float32 1D, sr: int, text: str). 실패 시 합성 사인파로 대체.

    최신 datasets는 Audio 디코딩에 torchcodec을 요구한다. 그 의존을 피하려고
    decode=False로 파일 경로/바이트만 받아 soundfile로 직접 디코딩한다.
    (whisper·wav2vec2 실험과 동일한 예시 오디오·로딩 방식을 재사용)
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
    from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

    device = get_device()
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    # self_attn이 output_attentions와 무관하게 probs를 반환하므로 eager 지정 불필요.
    model = Wav2Vec2ConformerForCTC.from_pretrained(MODEL_ID).to(device).eval()
    return processor, model, device


def build_context() -> dict:
    """단계 스크립트가 공유하는 컨텍스트. run.py는 한 번만 만들어 재사용한다.

    front-end를 실제로 돌려 conformer 입력 hidden (1, T, 1024)과 rel-pos 임베딩
    (1, 2T-1, 1024)을 만든다. eval 모드라 encoder 앞단의 dropout/spec-augment가 no-op이므로
    이 hidden이 곧 encoder.layers[0]가 받는 입력과 bit-exact로 같다(STEP 5에서 검증).
    """
    import torch

    seed_everything()
    processor, model, device = load_model()
    audio, sr, text = load_sample()

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)  # (1, N) zero-mean/unit-var 정규화됨

    conformer = model.wav2vec2_conformer  # Wav2Vec2ConformerModel
    encoder = conformer.encoder
    cfg = model.config
    OUT.mkdir(exist_ok=True)

    with torch.no_grad():
        extract = conformer.feature_extractor(input_values).transpose(1, 2)  # (1, T, 512)
        conformer_in, _ = conformer.feature_projection(extract)              # (1, T, 1024)
        rel_pos = encoder.embed_positions(conformer_in)                      # (1, 2T-1, 1024)

    return {
        "processor": processor, "model": model, "conformer": conformer,
        "encoder": encoder, "layer": encoder.layers[LAYER_IDX], "config": cfg,
        "device": device, "audio": audio, "sr": sr, "text": text,
        "input_values": input_values, "extract": extract,
        "conformer_in": conformer_in, "rel_pos": rel_pos,
    }


def rule(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def use_korean_font() -> None:
    """그림의 한글 제목이 깨지지 않게 한글 지원 폰트를 등록한다(있으면).

    없으면 조용히 넘어간다(값·shape엔 영향 없음). common을 import하는 순간 적용된다.
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
        matplotlib.rcParams["axes.unicode_minus"] = False
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
