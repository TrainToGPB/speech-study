"""VoxCPM-0.5B 최소 TTS 추론 (tokenizer-free TTS, arxiv 2509.24650).

텍스트 한 문장을 음성으로 합성해 outputs/에 저장한다. 두 환경 모두에서 돌게
device는 shared/env.py의 get_device()로 얻는다(work=mps / home=cuda).

    python download.py   # 최초 1회 가중치 받기
    python run.py
"""
import os
import pathlib
import sys

# MPS에서 아직 미지원인 연산(diffusion LocDiT 일부)이 있으면 CPU로 폴백.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# 레포 루트를 경로에 추가해 shared/env.py를 임포트
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.env import get_device  # noqa: E402

import soundfile as sf  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

MODEL_ID = "openbmb/VoxCPM-0.5B"
TEXT = "VoxCPM is a tokenizer-free text-to-speech model that generates highly expressive speech."
OUT = pathlib.Path(__file__).parent / "outputs" / "voxcpm_0.5b_demo.wav"


def main() -> None:
    device = get_device()
    print(f"device: {device}")

    # load_denoiser=False: 화자 복제용 참조오디오 향상기(zipenhancer)라 데모엔 불필요.
    # optimize=False: MPS 첫 실행 안정성 우선(컴파일/그래프 최적화 끄기).
    model = VoxCPM.from_pretrained(
        MODEL_ID, load_denoiser=False, optimize=False, device=device
    )

    # cfg_value: LM→LocDiT 가이던스 세기, inference_timesteps: LocDiT 확산 스텝.
    # denoise=False (denoiser 미로딩), normalize=True: 내장 텍스트 정규화(wetext).
    wav = model.generate(
        text=TEXT,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=True,
        denoise=False,
    )

    sr = getattr(getattr(model, "tts_model", None), "sample_rate", 16000)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT), wav, sr)
    dur = len(wav) / sr
    print(f"saved: {OUT}  ({dur:.2f}s @ {sr}Hz, {len(wav)} samples)")


if __name__ == "__main__":
    main()
