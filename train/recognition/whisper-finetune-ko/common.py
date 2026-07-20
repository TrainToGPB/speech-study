"""Whisper 한국어 파인튜닝 공통 유틸 (M5·MPS 안전 설정 포함).

모델 openai/whisper-base 를 Zeroth-Korean 소량에 full fine-tune 한다.
MPS에서 안전하도록: fp32, MPS fallback 허용, tokenizers 병렬 off.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # 미지원 op는 CPU 폴백
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
from shared.env import get_device  # noqa: E402

MODEL_ID = "openai/whisper-base"     # multilingual (한국어 약함 → 개선폭 큼)
DATASET_ID = "Bingsu/zeroth-korean"  # FLAC, CC-BY-4.0
LANG = "korean"
TASK = "transcribe"
SEED = 0

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "outputs"
CKPT_DIR = OUT / "whisper-base-ko"


def seed_everything() -> None:
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_processor():
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(MODEL_ID, language=LANG, task=TASK)


def load_model(from_dir: str | None = None):
    """(model, device). from_dir 지정 시 파인튜닝 체크포인트에서 로드."""
    from transformers import WhisperForConditionalGeneration

    device = get_device()
    src = from_dir or MODEL_ID
    model = WhisperForConditionalGeneration.from_pretrained(src).to(device)
    model.generation_config.language = LANG
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    return model, device


def load_splits(n_train: int = 1000, n_eval: int = 200):
    """Zeroth train/test 를 소량 subset 으로. 16kHz mono 보장."""
    from datasets import Audio, load_dataset

    ds_train = load_dataset(DATASET_ID, split="train")
    ds_eval = load_dataset(DATASET_ID, split="test")
    ds_train = ds_train.shuffle(seed=SEED).select(range(min(n_train, len(ds_train))))
    ds_eval = ds_eval.select(range(min(n_eval, len(ds_eval))))
    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
    ds_eval = ds_eval.cast_column("audio", Audio(sampling_rate=16000))
    return ds_train, ds_eval


def make_prepare(processor):
    """example → {input_features, labels}. datasets.map 용."""

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    return prepare


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """input_features 와 labels 를 각각 패딩. label 패딩은 -100 마스크."""

    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # collator가 앞에 BOS를 다시 붙이므로 있으면 하나 제거
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_cer():
    import evaluate

    return evaluate.load("cer")


def transcribe(model, processor, device, example) -> str:
    """단일 example → 예측 문자열 (한국어 강제)."""
    import torch

    audio = example["audio"]
    feats = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(feats, language=LANG, task=TASK, max_new_tokens=225)
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def evaluate_cer(model, processor, device, ds):
    """(cer, pairs) — pairs는 (정답, 예측) 리스트. 02/04가 공유한다."""
    cer = load_cer()
    refs, hyps = [], []
    for ex in ds:
        refs.append(ex["text"])
        hyps.append(transcribe(model, processor, device, ex))
    return cer.compute(predictions=hyps, references=refs), list(zip(refs, hyps))


def rule(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
