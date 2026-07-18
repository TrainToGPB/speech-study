#!/usr/bin/env python3
"""현재 실험 폴더(cwd)에 머신별 venv를 만들고 의존성을 설치·검증한다.

반드시 대상 실험 폴더 안에서 실행하세요 (venv가 실험별로 분리됩니다).

  cd <topic>/<name>
  python <스킬경로>/scripts/setup_env.py [home|work] [--dry-run]

프로필:
  home  Windows(WSL2) · RTX 3060 Ti 8GB · CUDA cu121 (torch 2.5.1)
  work  MacBook Pro M5 32GB · MPS (torch 기본 빌드)

프로필 생략 시 OS로 추정(Linux→home, Darwin→work)합니다.
HF 캐시(~/.cache/huggingface)는 기본 위치를 그대로 써서 venv 간 공유됩니다.
"""
import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

TORCH = {
    # 프로필별 torch 설치 인자
    "home": ["torch==2.5.1", "torchaudio==2.5.1",
             "--index-url", "https://download.pytorch.org/whl/cu121"],
    "work": ["torch", "torchaudio"],
}


def guess_profile() -> str:
    return "work" if platform.system() == "Darwin" else "home"


def run(cmd: list[str], dry: bool) -> None:
    print("  $", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", nargs="?", choices=["home", "work"], default=None)
    ap.add_argument("--dry-run", action="store_true", help="명령만 출력하고 실행하지 않음")
    args = ap.parse_args()

    profile = args.profile or guess_profile()
    auto = args.profile is None
    cwd = Path.cwd()
    venv = cwd / ".venv"
    py = venv / "bin" / "python"
    use_uv = shutil.which("uv") is not None

    print(f"실험 폴더 : {cwd}")
    print(f"프로필    : {profile}" + (" (OS로 자동 추정 — 맞는지 확인)" if auto else ""))
    print(f"패키지매니저: {'uv' if use_uv else 'venv+pip'}")
    print(f"{'[dry-run] ' if args.dry_run else ''}셋업 시작\n")

    # 1) venv
    if use_uv:
        run(["uv", "venv", str(venv)], args.dry_run)
    else:
        run([sys.executable, "-m", "venv", str(venv)], args.dry_run)

    # 2) torch (프로필별)
    if use_uv:
        run(["uv", "pip", "install", "--python", str(py), *TORCH[profile]], args.dry_run)
    else:
        run([str(venv / "bin" / "pip"), "install", *TORCH[profile]], args.dry_run)

    # 3) 나머지 의존성
    req = cwd / "requirements.txt"
    if req.exists():
        if use_uv:
            run(["uv", "pip", "install", "--python", str(py), "-r", str(req)], args.dry_run)
        else:
            run([str(venv / "bin" / "pip"), "install", "-r", str(req)], args.dry_run)
    else:
        print("  (requirements.txt 없음 — 건너뜀)")

    # 4) 검증
    verify = (
        "import torch;"
        "print('torch', torch.__version__);"
        "print('cuda available', torch.cuda.is_available());"
        "print('mps available', getattr(torch.backends,'mps',None) is not None "
        "and torch.backends.mps.is_available())"
    )
    print("\n검증:")
    run([str(py), "-c", verify], args.dry_run)

    print("\n완료. 실행: python run.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
