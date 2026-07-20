"""VoxCPM-0.5B step-by-step 실습 공통 유틸.

목표: tokenizer-free TTS(VoxCPM, arXiv 2509.24650)가 텍스트 한 문장을 어떻게
연속(continuous) speech latent으로 만들어 파형까지 가는지, 논문 아키텍처의 각
모듈을 **실제 가중치**로 하나씩 통과시키며 텐서로 따라간다.

모델은 공식 `openbmb/VoxCPM-0.5B`. wrapper(`from voxcpm import VoxCPM`)의 실체는
`model.tts_model = VoxCPMModel` 이고, 논문 모듈이 그대로 attribute로 붙어 있다:

    audio_vae    Causal Audio VAE  16kHz waveform <-> 25Hz x 64d continuous latent (640x)
    feat_encoder LocEnc            지금까지의 latent patch -> acoustic embedding
    base_lm      TSLM              MiniCPM-4-0.5B(24L, hidden 1024) semantic 골격
    fsq_layer    FSQ               semi-discrete skeleton: in_proj -> tanh -> round(*9)/9 -> out_proj
    residual_lm  RALM              6L, acoustic(화자·미세 prosody) 잔차
    feat_decoder LocDiT+flow-match 다음 latent patch를 noise에서 생성(Euler ODE, CFG)
    stop_head    Stop Predictor    continuous라 EOS 토큰이 없어 별도 정지 신호(2-class)
    lm_to_dit_proj + res_to_dit_proj  ->  h_final = skeleton + residual  (핵심 등식)

핵심 흐름(추론)은 VoxCPMModel._inference를 그대로 따라간다:
    text(BPE)+과거 latent(LocEnc) -> TSLM -> FSQ skeleton -> RALM residual
    -> h_final -> LocDiT diffusion으로 다음 latent patch -> ... -> AudioVAE.decode

device는 shared/env.py의 get_device()로 얻어 work(mps)/home(cuda) 모두에서 돈다.
"""
import io
import os
import pathlib
import sys

import numpy as np

# MPS에서 아직 미지원인 연산이 있으면 CPU로 폴백(일부 확산/attention 연산 대비).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# 레포 루트를 경로에 추가해 shared/env.py를 임포트
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
from shared.env import get_device  # noqa: E402

MODEL_ID = "openbmb/VoxCPM-0.5B"  # 논문 원본 0.5B (VoxCPM2 아님)
# 합성할 목표 문장. 짧게 잡아 work(MPS)에서도 빨리 돈다.
TARGET_TEXT = "The quick brown fox jumps over the lazy dog."
SEED = 0
OUT = pathlib.Path(__file__).resolve().parent / "outputs"


def seed_everything() -> None:
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_model():
    """(wrapper VoxCPM, tts_model VoxCPMModel, device) 반환. 모델은 1회만 로드한다.

    load_denoiser=False: 참조오디오 향상기(zipenhancer, modelscope)는 실습에 불필요.
    optimize=False: torch.compile/triton은 CUDA 전용이라 work(MPS)에선 꺼서 안정성 우선.
    """
    from voxcpm import VoxCPM

    device = get_device()
    model = VoxCPM.from_pretrained(
        MODEL_ID, load_denoiser=False, optimize=False, device=device
    )
    return model, model.tts_model, device


def model_dtype(tts):
    """모델 파라미터의 실제 dtype(예: MPS는 fp32, CUDA는 bf16)."""
    import torch

    return next(tts.base_lm.parameters()).dtype if True else torch.float32


def load_reference_samples(n: int = 1):
    """참조 오디오 예시를 n개 반환. 각 원소 dict(audio, sr, text, speaker).

    voice cloning(화자 복제)의 참조로 쓴다. whisper/wav2vec2 실험과 동일한
    librispeech dummy를 재사용하되, 최신 datasets의 torchcodec 의존을 피하려
    decode=False로 경로/바이트만 받아 soundfile로 직접 디코딩한다.
    실패 시 합성 사인파 1개로 대체(오프라인 안전).
    """
    try:
        import soundfile as sf
        from datasets import Audio, load_dataset

        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        ds = ds.cast_column("audio", Audio(decode=False))
        out = []
        for i in range(min(n, len(ds))):
            item = ds[i]
            a = item["audio"]
            if a.get("path") and os.path.exists(a["path"]):
                audio, sr = sf.read(a["path"])
            else:
                audio, sr = sf.read(io.BytesIO(a["bytes"]))
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:  # 스테레오 → 모노
                audio = audio.mean(axis=1)
            out.append(
                {
                    "audio": audio,
                    "sr": int(sr),
                    "text": item.get("text", ""),
                    "speaker": item.get("speaker_id", i),
                }
            )
        return out
    except Exception as e:  # noqa: BLE001 — 오프라인 등
        print(f"[warn] 데이터셋 로드 실패({e}); 합성 사인파로 대체.", file=sys.stderr)
        sr = 16000
        t = np.linspace(0, 4, sr * 4, endpoint=False)
        audio = (0.1 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return [{"audio": audio, "sr": sr, "text": "(synthetic sine)", "speaker": -1}][:n]


def save_wav(path, audio, sr: int) -> None:
    """1D float 파형을 wav로 저장(soundfile)."""
    import soundfile as sf

    OUT.mkdir(exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(str(path), audio, sr)


def prepare_reference_wav(sample, sample_rate: int):
    """참조 샘플을 모델 sample_rate(16k)로 맞춰 outputs/ref.wav로 저장하고 경로 반환."""
    import torch
    import torchaudio

    audio, sr = sample["audio"], sample["sr"]
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)  # (1, T)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    ref_path = OUT / "ref.wav"
    save_wav(ref_path, wav.squeeze(0).numpy(), sample_rate)
    return str(ref_path), wav.squeeze(0).numpy()


def build_context() -> dict:
    """단계 스크립트가 공유하는 컨텍스트. run.py는 한 번만 만들어 재사용한다.

    무거운 모델 로드를 1회로 끝내고, 참조 오디오(voice cloning용)와 목표 문장만
    준비한다. 실제 텐서(latent·hidden 등)는 각 STEP이 ctx에 채워 넣는다.
    """
    seed_everything()
    OUT.mkdir(exist_ok=True)
    model, tts, device = load_model()
    sample = load_reference_samples(1)[0]
    ref_path, ref_audio = prepare_reference_wav(sample, tts.sample_rate)
    return {
        "model": model,
        "tts": tts,
        "device": device,
        "sample_rate": tts.sample_rate,
        "ref_wav_path": ref_path,
        "ref_audio": ref_audio,        # np.float32 1D @ sample_rate
        "prompt_text": sample["text"],  # 참조 오디오의 전사(voice cloning prompt)
        "target_text": TARGET_TEXT,     # 합성할 목표 문장
        "dtype": model_dtype(tts),
    }


def rule(title: str) -> None:
    print("\n" + "=" * 76 + f"\n{title}\n" + "=" * 76)


def patch_torchaudio_load() -> None:
    """torchaudio 2.9+는 torchcodec 백엔드로 위임하는데, 이 백엔드는 시스템 ffmpeg
    (libavutil.56 등)을 요구한다. Mac에 ffmpeg가 없으면 `torchaudio.load`가 죽는다
    (`backend="soundfile"`로도 우회 안 됨 — 로더 전체가 torchcodec 기반).

    voxcpm 내부(build_prompt_cache/_generate)가 참조 오디오를 `torchaudio.load`로
    읽으므로, 이미 의존 중인 soundfile 기반 로더로 갈아끼워 torchcodec/ffmpeg를
    우회한다. functional.resample은 순수 torch라 그대로 둔다. home(CUDA)에서도 무해.
    """
    try:
        import soundfile as sf
        import torch
        import torchaudio
    except Exception:  # noqa: BLE001
        return

    def _sf_load(filepath, *args, **kwargs):
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)  # (T, C)
        return torch.from_numpy(data.T.copy()), sr  # (C, T), sr

    torchaudio.load = _sf_load


def use_korean_font() -> None:
    """그림의 한글 제목이 깨지지 않게 한글 지원 폰트를 등록한다(있으면)."""
    try:
        import matplotlib
        from matplotlib import font_manager

        avail = {f.name for f in font_manager.fontManager.ttflist}
        for name in ("AppleGothic", "Apple SD Gothic Neo", "AppleSDGothicNeo",
                     "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"):
            if name in avail:
                matplotlib.rcParams["font.family"] = name
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:  # noqa: BLE001 — 폰트 설정 실패는 치명적이지 않음
        pass


def savefig(fig, name: str) -> None:
    """outputs/ 에 그림 저장(헤드리스 안전)."""
    try:
        fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
        print(f"[저장] {OUT / name}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 그림 저장 실패({name}): {e}")


patch_torchaudio_load()  # common import 시 torchaudio.load → soundfile로 우회
use_korean_font()  # common import 시 1회 적용
