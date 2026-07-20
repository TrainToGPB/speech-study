"""05 사전학습 합성용 Coqui TTS를 격리 venv(.venv-tts)에 설치한다.

coqui-tts는 자체 torch/의존성 핀이 있어, 01~04 검증을 마친 core `.venv`를 깨지 않도록
별도 venv에 설치한다. 설치 후 05는 그 venv로 실행:

    python download.py
    .venv-tts/bin/python 05_pretrained_synth.py    # 또는: python run.py --pretrained (core에 없으면 skip)
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv-tts"


def main() -> None:
    use_uv = shutil.which("uv") is not None
    py = VENV / "bin" / "python"
    print(f"[download] Coqui TTS 격리 venv 설치 → {VENV}  (uv={use_uv})")
    try:
        # coqui-tts는 torch 자동 설치 안 함 → 함께. transformers 5.x는 coqui import 체인과
        # 안 맞아 4.x로 고정(isin_mps_friendly 등). torchcodec은 torch 2.9+ 오디오 IO용.
        pkgs = ["torch", "torchaudio", "coqui-tts", "transformers>=4.55,<5", "torchcodec", "soundfile"]
        if use_uv:
            subprocess.run(["uv", "venv", "--python", "3.12", str(VENV)], check=True)
            subprocess.run(["uv", "pip", "install", "--python", str(py), *pkgs], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
            subprocess.run([str(VENV / "bin" / "pip"), "install", *pkgs], check=True)
        print(f"[download] 완료. 실행:  {py} 05_pretrained_synth.py")
    except subprocess.CalledProcessError as e:
        print(f"[download] 설치 실패: {e}. 05는 건너뜁니다(LOG 막힌 점에 기록).")


if __name__ == "__main__":
    main()
