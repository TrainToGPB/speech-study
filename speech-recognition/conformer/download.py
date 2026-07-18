"""HF 캐시 워밍업 — 예시 오디오 + conformer 모델을 미리 받아 ~/.cache/huggingface 에 채운다.

실제 가중치·오디오는 HF 캐시에만 저장되고 git에서는 제외된다. run.py 전에 한 번 돌리면
어느 머신에서든 자산을 복원할 수 있다. (모델 ~600M — 최초 1회 다운로드에 시간이 걸림)
"""
from common import MODEL_ID, load_sample


def main() -> None:
    print("1) 예시 오디오 다운로드 (LibriSpeech dummy)")
    audio, sr, text = load_sample()
    print(f"   길이 {len(audio) / sr:.2f}s · sr {sr} · 전사 {text[:60]!r}")

    print(f"2) conformer 모델 다운로드/캐시: {MODEL_ID}  (~600M, 시간 걸림)")
    from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

    Wav2Vec2Processor.from_pretrained(MODEL_ID)
    Wav2Vec2ConformerForCTC.from_pretrained(MODEL_ID)
    print("완료. 캐시 위치: ~/.cache/huggingface")


if __name__ == "__main__":
    main()
