"""06~07 실제 v1 해부용 ttaoREtw 의존성을 격리 venv(.venv-ttao)에 설치하고 체크포인트를 받는다.

core `.venv`(01~04 검증됨)를 깨지 않도록 별도 venv에 설치한다. 설치 후 06/07은 그 venv로 실행:

    python3 download_ttao.py
    .venv-ttao/bin/python 06_real_v1_synth.py
    .venv-ttao/bin/python 07_alignment_compare.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv-ttao"
OUT = HERE / "outputs"
CKPT = OUT / "ttao_tacotron.pth"
GDRIVE_ID = "1q8xLo9zyyclIDgYk3V2mczofnQwqT6pk"  # ttaoREtw LJSpeech 체크포인트(74MB)
# module.py는 torch만, utils.py는 librosa/scipy/soundfile, text/는 inflect/unidecode 의존.
# numba/numpy 핀: 리졸버가 옛 numba(0.53, py<3.10 전용)를 집어 llvmlite 소스빌드가 깨지는 걸 막는다.
# numba>=0.60은 py3.12 프리빌트 휠 + numpy<2.1 요구 → numpy도 함께 핀.
PKGS = ["torch", "numpy<2.1", "numba>=0.60", "librosa>=0.10,<0.12", "scipy",
        "pyyaml", "inflect", "unidecode", "soundfile", "matplotlib", "gdown"]


def main() -> None:
    use_uv = shutil.which("uv") is not None
    py = VENV / "bin" / "python"
    OUT.mkdir(exist_ok=True)
    print(f"[download_ttao] 격리 venv 설치 → {VENV}  (uv={use_uv})")
    try:
        if use_uv:
            subprocess.run(["uv", "venv", "--python", "3.12", str(VENV)], check=True)
            subprocess.run(["uv", "pip", "install", "--python", str(py), *PKGS], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
            subprocess.run([str(VENV / "bin" / "pip"), "install", *PKGS], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[download_ttao] 설치 실패: {e}. 06/07은 건너뜁니다(LOG 막힌 점에 기록).")
        return

    if CKPT.exists() and CKPT.stat().st_size > 1_000_000:
        print(f"[download_ttao] 체크포인트 이미 있음(재다운로드 생략): {CKPT}")
    else:
        print(f"[download_ttao] 체크포인트 다운로드(GDrive {GDRIVE_ID}) → {CKPT}")
        try:
            subprocess.run(
                [str(py), "-c",
                 f"import gdown; gdown.download(id={GDRIVE_ID!r}, output={str(CKPT)!r}, quiet=False)"],
                check=True)
        except subprocess.CalledProcessError as e:
            print(f"[download_ttao] 체크포인트 다운로드 실패: {e}. 06/07은 건너뜁니다.")
            return
    print(f"[download_ttao] 완료. 실행:  {py} 06_real_v1_synth.py")


if __name__ == "__main__":
    main()
