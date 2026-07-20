"""whisper-base 가중치 + Zeroth-Korean subset + CER metric 을 HF 캐시에 미리 받는다.

실제 파일은 ~/.cache/huggingface(공유 캐시)에 저장된다.
"""
from common import DATASET_ID, MODEL_ID, load_cer


def main() -> None:
    print(f"1) 모델 캐시: {MODEL_ID}")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    WhisperProcessor.from_pretrained(MODEL_ID)
    WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    print(f"2) 데이터 캐시: {DATASET_ID} (train/test)")
    from datasets import load_dataset

    dtr = load_dataset(DATASET_ID, split="train")
    dte = load_dataset(DATASET_ID, split="test")
    print(f"   train {len(dtr)} · test {len(dte)} · columns {dtr.column_names}")
    print(f"   예시 전사: {dte[0].get('text', dte[0])!r}"[:100])

    print("3) CER metric 캐시")
    load_cer()
    print("완료. 캐시 위치: ~/.cache/huggingface")


if __name__ == "__main__":
    main()
