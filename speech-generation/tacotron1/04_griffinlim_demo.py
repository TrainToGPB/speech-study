"""STEP 4 — Griffin-Lim으로 실제 오디오 복원 (유일하게 '진짜 소리' 나는 코어 스텝).

linear-scale spectrogram은 크기(magnitude)만 있고 위상이 없다. Griffin-Lim은 STFT/ISTFT를
오가며 위상을 반복 추정해 파형을 복원한다. iter를 늘릴수록 spectral convergence가 낮아진다.
Tacotron1이 뉴럴 보코더 없이 파형을 얻는 바로 그 단계. (오디오 처리는 CPU — istft 안정성)
"""
import torch
import torchaudio

from common import HP, build_context, rule, save_wav, savefig


def spectral_convergence(mag_rec, mag_ref):
    return (torch.linalg.norm(mag_rec - mag_ref) / torch.linalg.norm(mag_ref)).item()


def main(ctx=None):
    ctx = ctx or build_context()
    hp: HP = ctx["hp"]
    rule("STEP 4 · Griffin-Lim 실제 오디오 복원")

    wav = torch.tensor(ctx["audio"], dtype=torch.float32)  # (N,) CPU
    sr = ctx["sr"]
    to_spec = torchaudio.transforms.Spectrogram(
        n_fft=hp.n_fft, win_length=hp.win, hop_length=hp.hop, power=1.0)
    mag = to_spec(wav)  # (freq, time) magnitude
    print(f"오디오         : {wav.numel() / sr:.2f}s @ {sr}Hz | magnitude spec {tuple(mag.shape)} "
          f"(freq={hp.linear_bins}, time)")

    recs = {}
    for n_iter in (10, 30, 60):
        gl = torchaudio.transforms.GriffinLim(
            n_fft=hp.n_fft, n_iter=n_iter, win_length=hp.win, hop_length=hp.hop, power=1.0)
        rec = gl(mag)
        sc = spectral_convergence(to_spec(rec), mag)
        recs[n_iter] = rec
        print(f"  GL {n_iter:>2d} iter : spectral convergence {sc:.4f}  (낮을수록 원본에 가까움)")

    save_wav(wav, sr, "original.wav")
    save_wav(recs[60] / recs[60].abs().max().clamp(min=1e-8), sr, "griffinlim.wav")
    print(f"위상 추정만으로 복원 — magnitude^{hp.gl_power} 트릭은 아티팩트 저감용, 여기선 생략")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        for ax, m, title in zip(axes, [mag, to_spec(recs[60])], ["원본 magnitude", "Griffin-Lim 60 iter"]):
            ax.imshow(torch.log1p(m).numpy(), aspect="auto", origin="lower")
            ax.set_title(title)
            ax.set_xlabel("frame")
            ax.set_ylabel("freq bin")
        savefig(fig, "griffinlim_spec.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] spec png 실패: {e}")

    ctx["gl_recons"] = recs
    return ctx


if __name__ == "__main__":
    main()
