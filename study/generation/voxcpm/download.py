"""VoxCPM-0.5B 가중치 + 참조 오디오 예시 다운로드.

실제 파일은 .gitignore로 git에서 제외됩니다. 여기엔 로더만 둡니다.
가중치는 공유 HF 캐시(~/.cache/huggingface)에 받아 두므로, 다른 실험과
캐시를 공유하고 run.py의 from_pretrained가 그대로 읽습니다.

    python download.py
"""
from huggingface_hub import snapshot_download

# arxiv 2509.24650 논문 원본 모델. VoxCPM2(v2)가 아니라 0.5B를 명시적으로 받는다.
MODEL_ID = "openbmb/VoxCPM-0.5B"


def main() -> None:
    path = snapshot_download(MODEL_ID)
    print(f"downloaded {MODEL_ID} -> {path}")

    # 참조 오디오(voice cloning prompt) 예시 — whisper/wav2vec2 실험과 동일 데이터셋.
    # 미리 캐시해 두면 오프라인에서도 run.py가 돈다. 실패해도 치명적이지 않다.
    try:
        from datasets import load_dataset

        load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        print("prewarmed reference audio: hf-internal-testing/librispeech_asr_dummy")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 참조 오디오 프리워밍 실패(런타임에 재시도): {e}")


if __name__ == "__main__":
    main()
