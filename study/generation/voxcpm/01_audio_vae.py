"""STEP 1 — Causal Audio VAE: waveform <-> 연속 latent (tokenizer-free의 토대).

VoxCPM이 "tokenizer-free"인 이유가 여기서 드러난다. 기존 codec TTS(EnCodec 등)는
파형을 **이산 토큰**으로 바꿔 LLM처럼 예측하지만(양자화가 디테일을 버림),
VoxCPM은 Causal Audio VAE로 파형을 **연속(continuous) latent**으로만 압축한다.

  encode: 16kHz waveform → (64, T)  연속 latent      (downsample 640× → 25Hz)
  decode: (64, T) latent → 16kHz waveform            (거의 무손실 재구성)

이 latent엔 codebook도, argmax도 없다. 그냥 64차원 실수 벡터의 시퀀스다.
그 다음 patch_size=2로 두 프레임을 묶어 12.5Hz 토큰율로 만든다 — SpeechLM 기준
으로도 짧아 latency에 유리(Notion "patch-size 2 → 12.5Hz").

이 STEP은 참조 오디오를 latent로 encode하고(뒤 STEP들의 과거 latent 조건이 됨),
곧바로 decode해 왕복(round-trip)이 파형을 얼마나 잘 복원하는지 확인한다.
"""
import numpy as np
import torch

from common import build_context, rule, savefig, save_wav, OUT


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    tts, device, sr = ctx["tts"], ctx["device"], ctx["sample_rate"]
    rule("STEP 1 · Causal Audio VAE (waveform <-> 연속 latent, 640× / 25Hz)")

    vae = tts.audio_vae
    patch_len = tts.patch_size * vae.chunk_size  # 2 * 640 = 1280 (한 patch가 덮는 파형 샘플 수)

    # 참조 오디오를 patch_len의 배수로 좌측 패딩(실제 추론과 동일: 유효 오디오를 뒤에 둠)
    wav = torch.from_numpy(np.asarray(ctx["ref_audio"], dtype=np.float32)).unsqueeze(0)  # (1, T)
    orig_len = wav.shape[1]
    if orig_len % patch_len != 0:
        pad = patch_len - orig_len % patch_len
        wav = torch.nn.functional.pad(wav, (pad, 0))
    wav = wav.to(device)

    with torch.inference_mode():
        z = vae.encode(wav.to(torch.float32), sr)          # (1, 64, T_frame)
        rec = vae.decode(z.to(torch.float32)).squeeze(1)   # (1, T_samples)

    n_frame = z.shape[2]
    frame_rate = sr / vae.chunk_size                        # 16000 / 640 = 25 Hz
    # 실제 추론의 patchify: (D, T_frame) → (T_patch, patch_size, D)
    audio_feat = z.view(vae.latent_dim, -1, tts.patch_size).permute(1, 2, 0).contiguous()
    n_patch = audio_feat.shape[0]

    print(f"입력 waveform  : {tuple(wav.shape)}  ({wav.shape[1] / sr:.2f}s @ {sr}Hz, 좌패딩 {wav.shape[1]-orig_len} sample)")
    print(f"VAE latent(z)  : {tuple(z.shape)}  → 64차원 '연속' 벡터 × {n_frame} 프레임")
    print(f"압축률         : {sr}Hz → {frame_rate:.0f}Hz  (chunk {vae.chunk_size} = 640× 다운샘플)")
    print(f"patchify       : {tuple(audio_feat.shape)}  = (patch {n_patch}, patch_size {tts.patch_size}, dim {vae.latent_dim}) → 토큰율 {frame_rate/tts.patch_size:.1f}Hz")
    print(f"latent 값 범위  : min {z.min():.2f} / max {z.max():.2f} / std {z.std():.2f}  (이산 코드 아님 — 실수)")

    # round-trip 품질(패딩 구간 제외)
    rec_trim = rec[:, -orig_len:].float().cpu().numpy().reshape(-1)
    orig_trim = wav[:, -orig_len:].float().cpu().numpy().reshape(-1)
    corr = float(np.corrcoef(orig_trim, rec_trim)[0, 1])
    rmse = float(np.sqrt(np.mean((orig_trim - rec_trim) ** 2)))
    print(f"round-trip     : corr {corr:.4f} / rmse {rmse:.4f}  (연속 latent라 codec 양자화 손실이 없음)")
    save_wav(OUT / "01_vae_roundtrip.wav", rec_trim, sr)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.arange(orig_len) / sr
        fig, ax = plt.subplots(3, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 1, 2]})
        ax[0].plot(t, orig_trim, lw=0.4)
        ax[0].set(title="원본 waveform", xlim=(0, orig_len / sr))
        ax[1].plot(t, rec_trim, lw=0.4, color="tab:green")
        ax[1].set(title=f"VAE 재구성 (corr {corr:.3f})", xlabel="", xlim=(0, orig_len / sr))
        im = ax[2].imshow(z[0].float().cpu().numpy(), aspect="auto", origin="lower", cmap="magma",
                          extent=(0, n_frame / frame_rate, 0, vae.latent_dim))
        ax[2].set(title=f"연속 latent z = 64 × {n_frame} (25Hz) — 뒤 STEP들의 과거 오디오 조건",
                  xlabel="time (s)", ylabel="latent dim")
        fig.colorbar(im, ax=ax[2], fraction=0.02)
        fig.tight_layout()
        savefig(fig, "01_audio_vae.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    ctx["prompt_audio_feat"] = audio_feat.cpu()  # (T_patch, P, D) — 참조 오디오의 latent patch
    ctx["vae_latent"] = z
    return ctx


if __name__ == "__main__":
    main()
