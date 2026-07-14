"""02 — EnCodec: waveform → discrete acoustic token → waveform 복원.

배우는 것 (논문: EnCodec, arXiv:2210.13438):
- neural codec은 오토인코더 + RVQ(residual vector quantization)로 파형을 discrete token으로.
- 코드북이 여러 개(계단식). 한 프레임이 여러 token(= codebook 개수)으로 표현된다.
  → 이 "여러 스트림" 구조가 Moshi(Stage 5)의 parallel-stream 설계 근거.
- acoustic token: 파형 복원이 목표 → 디코딩하면 원음이 (거의) 돌아온다. 눈으로 확인한다.

실행: python 02_encodec_tokens.py
출력: outputs/02_encodec_reconstructed.wav
"""
import numpy as np
import soundfile as sf
import torch

from _common import OUT_DIR, banner, load_wav, pick_device, sample_path


def main() -> None:
    banner("02 · EnCodec — acoustic token (encode → token → decode)")
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    device = pick_device()
    model = EncodecModel.encodec_model_24khz().to(device)
    # 목표 대역폭(kbps). 높을수록 codebook↑ = token↑ = 음질↑
    target_bw = 6.0
    model.set_target_bandwidth(target_bw)
    model.eval()

    wav16, _ = load_wav(sample_path(), target_sr=16000)
    wav = torch.tensor(wav16)[None, None]  # (1,1,T)
    wav = convert_audio(wav[0], 16000, model.sample_rate, model.channels)[None]
    wav = wav.to(device)

    with torch.no_grad():
        encoded = model.encode(wav)
    # encoded: list of (codes, scale). codes: (batch, n_codebooks, frames)
    codes = torch.cat([c for c, _ in encoded], dim=-1)
    n_q = codes.shape[1]
    frames = codes.shape[-1]
    dur = wav.shape[-1] / model.sample_rate
    frame_rate = frames / dur

    print(f"입력: {dur:.2f}s @ {model.sample_rate}Hz")
    print(f"codes shape: {tuple(codes.shape)}  = (batch, n_codebooks={n_q}, frames={frames})")
    print(f"frame rate ≈ {frame_rate:.1f}Hz, 대역폭 {target_bw}kbps")
    print(f"→ 한 시점을 {n_q}개 token(codebook)으로 표현. 총 token 수 = {n_q}×{frames} = {n_q*frames}")
    print(f"codebook 크기: {model.quantizer.bins} (각 token은 0~{model.quantizer.bins-1})")
    print(f"\ncodes 예시 (첫 4 codebook × 첫 8 frame):\n{codes[0, :4, :8].cpu().numpy()}")

    # 복원
    with torch.no_grad():
        recon = model.decode(encoded)[0, 0].cpu().numpy()

    ref = wav[0, 0].cpu().numpy()
    n = min(len(ref), len(recon))
    mse = float(np.mean((ref[:n] - recon[:n]) ** 2))
    print(f"\n복원 MSE = {mse:.6e}  (작을수록 원음에 가까움)")
    print("→ acoustic token은 '복원'이 목표라 디코딩하면 원음이 돌아온다. 이게 semantic과의 결정적 차이.")

    out = OUT_DIR / "02_encodec_reconstructed.wav"
    sf.write(out, recon, model.sample_rate)
    print(f"[saved] {out}  ← 원음과 들어보며 비교")


if __name__ == "__main__":
    main()
