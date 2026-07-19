import pathlib
import sys

# 레포 루트를 경로에 추가해 shared/env.py를 임포트
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.env import get_device

device = get_device()
print(f"device: {device}")

# TODO: 실험 코드
