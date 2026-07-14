"""01 — waveform · STFT · mel-spectrogram, 그리고 frame rate 감각.

배우는 것:
- 음성은 초당 16000개 샘플(16kHz)의 1차원 신호. 너무 길어 LLM에 직접 못 넣는다.
- STFT로 시간-주파수 표현을 만들고, mel 스케일로 사람 청각에 맞춰 압축한다.
- "frame rate"(초당 프레임 수)가 시퀀스 길이 = latency 예산을 결정한다.

실행: python 01_waveform_mel.py
출력: outputs/01_waveform_mel.png
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from _common import OUT_DIR, banner, load_wav, sample_path


def main() -> None:
    banner("01 · waveform / mel-spectrogram / frame rate")
    wav, sr = load_wav(sample_path())
    dur = len(wav) / sr
    print(f"waveform: {len(wav):,} samples, {sr}Hz, {dur:.2f}s")
    print(f"→ 이 길이를 그대로 토큰으로 쓰면 {len(wav):,} 스텝. 비현실적.")

    # STFT → mel-spectrogram
    n_fft = 400        # 25ms 윈도우 @16kHz
    hop = 160          # 10ms hop → frame rate = 100Hz
    n_mels = 80
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    frames = mel_db.shape[1]
    frame_rate = sr / hop
    print(f"\nmel-spectrogram: shape {mel_db.shape}  (n_mels={n_mels}, frames={frames})")
    print(f"hop={hop} samples → frame rate = {frame_rate:.0f}Hz "
          f"({1000*hop/sr:.0f}ms/frame)")
    print(f"→ {len(wav):,} 샘플이 {frames} 프레임으로 압축됨 (약 {len(wav)//frames}배)")

    print("\n[frame rate = latency 예산]")
    for fr in [100, 50, 25, 12.5]:
        n = int(dur * fr)
        print(f"  {fr:>5}Hz → {dur:.1f}s 발화가 {n:>4} 토큰  "
              f"({'전통 mel' if fr==100 else 'codec/SSL 계열'})")
    print("  ↑ Stage 4 GLM-4-Voice가 12.5Hz를 쓰는 이유: 토큰 수↓ = 컨텍스트·latency↓")

    # 시각화
    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    t = np.arange(len(wav)) / sr
    ax[0].plot(t, wav, lw=0.5)
    ax[0].set(title=f"Waveform ({sr}Hz, {dur:.2f}s)", xlabel="time (s)", ylabel="amp")
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop, x_axis="time", y_axis="mel", ax=ax[1]
    )
    ax[1].set(title=f"Mel-spectrogram (80 mels, frame rate {frame_rate:.0f}Hz)")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    fig.tight_layout()
    out = OUT_DIR / "01_waveform_mel.png"
    fig.savefig(out, dpi=110)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
