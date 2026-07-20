"""VoxCPM-0.5B 가중치 다운로드.

실제 파일은 .gitignore로 git에서 제외됩니다. 여기엔 로더만 둡니다.
가중치는 공유 HF 캐시(~/.cache/huggingface)에 받아 두므로, 다른 실험과
캐시를 공유하고 run.py의 from_pretrained가 그대로 읽습니다.

    python download.py
"""
from huggingface_hub import snapshot_download

# arxiv 2509.24650 논문 원본 모델. VoxCPM2(v2)가 아니라 0.5B를 명시적으로 받는다.
MODEL_ID = "openbmb/VoxCPM-0.5B"

if __name__ == "__main__":
    path = snapshot_download(MODEL_ID)
    print(f"downloaded {MODEL_ID} -> {path}")
