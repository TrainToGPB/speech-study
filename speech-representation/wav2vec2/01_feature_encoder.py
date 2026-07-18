"""STEP 1 — Feature encoder: waveform X → latent sequence Z.

노션: 7층 temporal CNN이 16kHz raw waveform을 약 49Hz frame sequence로 압축한다.
stride의 곱이 5×2^6=320이므로 16,000 sample/s → 약 50 frame/s.
각 z_t는 약 25ms를 보는 '연속' 벡터일 뿐, 아직 ID가 정해진 discrete unit이 아니다.
"""
import numpy as np
import torch

from common import build_context, rule, OUT


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    model, iv = ctx["model"], ctx["input_values"]
    audio, sr = ctx["audio"], ctx["sr"]
    rule("STEP 1 · Feature encoder (waveform → latent Z)")

    with torch.no_grad():
        # feature_extractor = 7층 CNN. 출력은 (B, C=512, T)
        z = model.wav2vec2.feature_extractor(iv)  # (1, 512, T)
    z = z.transpose(1, 2)  # (1, T, 512)  frame이 시간축이 되도록
    ctx["Z"] = z
    T = z.shape[1]
    dur = len(audio) / sr

    print(f"입력 waveform : {tuple(iv.shape)}  ({dur:.2f}s @ {sr}Hz, sample {iv.shape[1]:,}개)")
    print(f"latent Z      : {tuple(z.shape)}  → frame {T}개, 채널 512")
    print(f"stride 곱     : 5×2^6 = 320  →  {sr}/320 = {sr/320:.1f} frame/s (이론)")
    print(f"실측 frame rate: {T/dur:.1f} Hz  (frame 간격 {1000*dur/T:.1f} ms)")
    print(f"z_0 앞부분    : {z[0, 0, :6].cpu().numpy().round(3)}  ← 연속값(ID 아님)")

    # 시각화: waveform + Z heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(11, 5), gridspec_kw={"height_ratios": [1, 2]})
        ax[0].plot(np.arange(len(audio)) / sr, audio, lw=0.4)
        ax[0].set(title="raw waveform X", xlabel="time (s)", xlim=(0, dur))
        im = ax[1].imshow(z[0].cpu().numpy().T, aspect="auto", origin="lower",
                          extent=(0, dur, 0, 512), cmap="magma")
        ax[1].set(title=f"latent Z = f(X)  ({T} frames x 512)", xlabel="time (s)", ylabel="channel")
        fig.colorbar(im, ax=ax[1], fraction=0.02)
        fig.tight_layout()
        fig.savefig(OUT / "01_feature_encoder.png", dpi=110)
        plt.close(fig)
        print(f"[저장] {OUT/'01_feature_encoder.png'}")
    except Exception as e:
        print(f"[warn] 시각화 생략: {e}")
    return ctx


if __name__ == "__main__":
    main()
