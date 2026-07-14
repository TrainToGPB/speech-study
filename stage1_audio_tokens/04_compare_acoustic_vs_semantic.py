"""04 — 핵심 실습: acoustic(EnCodec) vs semantic(HuBERT) 토큰 비교.

Stage 1의 결론을 한 화면에 모은다. 같은 음성을 두 방식으로 토큰화해 성격 차이를 대비한다.

    | 축            | EnCodec (acoustic)        | HuBERT (semantic)              |
    | 목표          | 파형 복원 (음질·화자)      | 내용·발음 (이해)                |
    | 복원 가능?    | O (디코더로 원음 복원)     | X (내용만, 음색 소실)           |
    | 표현 형태     | 다중 codebook 정수 스트림  | 연속 벡터 → k-means로 unit화    |
    | 대표 소비처   | VALL-E, Moshi (Stage 3·5) | GLM-4-Voice tokenizer (St.4·5)  |

이 선택이 이후 모든 SpeechLM 설계의 첫 번째 갈림길이다.

실행: python 04_compare_acoustic_vs_semantic.py
출력: outputs/04_compare.png, outputs/04_encodec_recon.wav
"""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from _common import OUT_DIR, banner, load_wav, pick_device, sample_path


def encodec_tokens(wav16, device):
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    model = EncodecModel.encodec_model_24khz().to(device)
    model.set_target_bandwidth(6.0)
    model.eval()
    wav = torch.tensor(wav16)[None, None]
    wav = convert_audio(wav[0], 16000, model.sample_rate, model.channels)[None].to(device)
    with torch.no_grad():
        enc = model.encode(wav)
        recon = model.decode(enc)[0, 0].cpu().numpy()
    codes = torch.cat([c for c, _ in enc], dim=-1)[0].cpu().numpy()  # (n_q, frames)
    fr = codes.shape[-1] / (wav.shape[-1] / model.sample_rate)
    return codes, recon, model.sample_rate, fr


def hubert_units(wav16, sr, device, k=100, layer=9):
    from sklearn.cluster import KMeans
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    name = "facebook/hubert-base-ls960"
    fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
    model = HubertModel.from_pretrained(name).to(device).eval()
    inputs = fe(wav16, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        hs = model(**inputs, output_hidden_states=True).hidden_states
    feat = hs[layer][0].cpu().numpy()
    units = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(feat)
    fr = len(units) / (len(wav16) / sr)
    return units, fr


def main() -> None:
    banner("04 · acoustic(EnCodec) vs semantic(HuBERT) — Stage 1 종합")
    device = pick_device()
    wav16, sr = load_wav(sample_path(), target_sr=16000)
    dur = len(wav16) / sr
    print(f"입력 음성: {dur:.2f}s @ {sr}Hz, device={device}\n")

    print("[1/2] EnCodec (acoustic) ...")
    codes, recon, ac_sr, ac_fr = encodec_tokens(wav16, device)
    n_ref = min(len(wav16), len(librosa.resample(recon, orig_sr=ac_sr, target_sr=sr)))
    recon16 = librosa.resample(recon, orig_sr=ac_sr, target_sr=sr)[:n_ref]
    mse = float(np.mean((wav16[:n_ref] - recon16) ** 2))
    print(f"    codes {codes.shape} (codebooks×frames), frame rate ≈ {ac_fr:.1f}Hz")
    print(f"    복원 MSE = {mse:.3e} → 원음 복원 O")

    print("[2/2] HuBERT (semantic) ...")
    units, se_fr = hubert_units(wav16, sr, device)
    dedup = units[np.insert(np.diff(units) != 0, 0, True)]
    print(f"    units {units.shape}, frame rate ≈ {se_fr:.1f}Hz, 복원 X (내용만)")

    banner("요약")
    print(f"{'':14}{'EnCodec(acoustic)':<24}{'HuBERT(semantic)'}")
    print(f"{'표현':14}{f'{codes.shape[0]} codebook 정수':<24}{'연속벡터→k-means unit'}")
    print(f"{'frame rate':14}{f'{ac_fr:.0f}Hz':<24}{f'{se_fr:.0f}Hz'}")
    print(f"{'토큰/초':14}{f'{codes.shape[0]*ac_fr:.0f} (={codes.shape[0]}×{ac_fr:.0f})':<24}{f'{se_fr:.0f}'}")
    print(f"{'복원':14}{'O (MSE %.1e)' % mse:<24}{'X (음색·화자 소실)'}")
    print(f"{'쓰임':14}{'VALL-E, Moshi':<24}{'GLM-4-Voice tokenizer'}")

    # 시각화
    fig, ax = plt.subplots(3, 1, figsize=(11, 8))
    t = np.arange(len(wav16)) / sr
    ax[0].plot(t, wav16, lw=0.5)
    ax[0].set(title="원음 waveform", xlabel="s")
    ax[1].imshow(codes, aspect="auto", origin="lower", cmap="viridis",
                 interpolation="nearest")
    ax[1].set(title=f"EnCodec acoustic tokens ({codes.shape[0]} codebooks × {codes.shape[1]} frames)",
              ylabel="codebook")
    ax[2].step(np.arange(len(units)), units, where="mid", lw=0.7)
    ax[2].set(title=f"HuBERT semantic units (k-means, frame rate {se_fr:.0f}Hz)",
              xlabel="frame", ylabel="unit id")
    fig.tight_layout()
    out_png = OUT_DIR / "04_compare.png"
    fig.savefig(out_png, dpi=110)
    sf.write(OUT_DIR / "04_encodec_recon.wav", recon, ac_sr)
    print(f"\n[saved] {out_png}")
    print(f"[saved] {OUT_DIR / '04_encodec_recon.wav'}")
    print("\n▶ 자가 점검: ASR엔 어느 토큰? TTS 음질엔 어느 토큰? (README 참고)")


if __name__ == "__main__":
    main()
