"""환경 무관 공통 헬퍼.

두 실습 머신(home=Windows/CUDA, work=Mac/MPS)에서 동일한 코드가 돌도록
device를 자동 감지한다. 실험 코드에서는 항상 이 함수를 통해 device를 얻는다.

    import pathlib, sys
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
    from shared.env import get_device
    device = get_device()
"""
from __future__ import annotations


def get_device() -> str:
    """사용 가능한 최선의 device 문자열을 반환한다: 'cuda' → 'mps' → 'cpu'."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    print(get_device())
