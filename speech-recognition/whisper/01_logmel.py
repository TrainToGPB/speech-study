"""STEP 1 — Log-Mel spectrogram: waveform → (80, 3000).

Whisper 인코더의 입력은 raw waveform이 아니라 **log-Mel spectrogram**이다.
feature extractor가 하는 일:
  - 오디오를 정확히 30초로 pad/trim (부족하면 0으로 채움)
  - 25ms window · 10ms hop 으로 STFT → power → 80개 Mel filterbank → log → 정규화
  - 30s / 10ms = 3000 frame, 즉 100 Hz 프레임레이트로 고정

그래서 shape는 발화 길이와 무관하게 항상 (1, 80, 3000). 실제 발화가 11초면 나머지
19초는 무음 패딩이라 mel 하한값으로 채워진다(그림에서 오른쪽이 밋밋한 이유).
"""
import numpy as np

from common import build_context, rule, savefig


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    audio, sr, mel, cfg = ctx["audio"], ctx["sr"], ctx["mel"], ctx["config"]
    rule("STEP 1 · Log-Mel spectrogram (waveform → 80 x 3000)")

    dur = len(audio) / sr
    n_mels, n_frames = mel.shape[1], mel.shape[2]
    frame_rate = n_frames / 30.0  # 30초 기준

    print(f"입력 waveform : ({len(audio):,},)  ({dur:.2f}s @ {sr}Hz)")
    print(f"log-Mel       : {tuple(mel.shape)}  → mel bin {n_mels}, frame {n_frames}")
    print(f"프레임레이트   : {frame_rate:.0f} Hz  (hop {1000 / frame_rate:.0f}ms) · 항상 30s로 pad→3000 frame")
    print(f"실제 발화 {dur:.1f}s → frame {int(round(dur * frame_rate))}개, 나머지 {int(3000 - dur * frame_rate)}개는 무음 패딩")
    print(f"값 범위       : min {mel.min():.2f} / max {mel.max():.2f}  (정규화된 log 스케일)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(11, 5), gridspec_kw={"height_ratios": [1, 2]})
        ax[0].plot(np.arange(len(audio)) / sr, audio, lw=0.4)
        ax[0].set(title="raw waveform", xlabel="time (s)", xlim=(0, 30))
        ax[0].axvline(dur, color="r", ls="--", lw=0.8)
        im = ax[1].imshow(mel[0].cpu().numpy(), aspect="auto", origin="lower",
                          extent=(0, 30, 0, n_mels), cmap="magma")
        ax[1].set(title=f"log-Mel spectrogram = 인코더 입력 ({n_mels} x {n_frames})",
                  xlabel="time (s) — 30s로 패딩됨", ylabel="mel bin")
        ax[1].axvline(dur, color="cyan", ls="--", lw=0.8)
        fig.colorbar(im, ax=ax[1], fraction=0.02)
        fig.tight_layout()
        savefig(fig, "01_logmel.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    return ctx


if __name__ == "__main__":
    main()
