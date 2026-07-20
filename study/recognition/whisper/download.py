"""예시 오디오 + Whisper 가중치를 미리 받아 HF 캐시에 warm-up 한다.

실제 파일은 ~/.cache/huggingface(공유 캐시)에 저장된다. 어느 머신에서든 이 스크립트만
돌리면 자산이 복원된다. outputs/ 의 그림은 run.py가 생성한다.
"""
from common import MODEL_ID, load_sample


def main() -> None:
    print("1) 예시 오디오 다운로드 (LibriSpeech dummy)")
    audio, sr, text = load_sample()
    print(f"   길이 {len(audio) / sr:.2f}s @ {sr}Hz | 전사: {text[:60]!r}")

    print(f"2) Whisper 모델 다운로드/캐시: {MODEL_ID}")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    WhisperProcessor.from_pretrained(MODEL_ID)
    WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    print("   완료. 캐시 위치: ~/.cache/huggingface")


if __name__ == "__main__":
    main()
