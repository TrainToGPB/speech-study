"""STEP 6 — 실제 학습된 Tacotron1(ttaoREtw, MIT)을 로드해 end-to-end 해부.

01~03은 random-init from-scratch라 forward에 acoustic 의미가 없다(shape만 검증). 여기선
논문 계보가 같은 실제 학습 체크포인트(ttaoREtw/Tacotron-pytorch, LJSpeech)를 로드해
'진짜 동작'을 본다 — content-based attention의 **대각선 alignment**, 음성 같은 mel/linear,
그리고 Griffin-Lim 파형(우리 04와 같은 DSP 보코더).

이 체크포인트 구조는 우리 modules.py와 같은 계열(CBHG·BahdanauAttn·GRU decoder·post-net
CBHG→linear)이지만 이름·차원(r=5, linear=1025, vocab=250)이 달라 우리 모듈엔 직접 로드
못 한다. 그래서 vendored ttao_ref 코드로 그대로 돌린다(자세한 건 ttao_ref/PROVENANCE.md).

coqui(05)처럼 core .venv를 안 깨게 격리 venv(.venv-ttao)로 실행:
    python3 download_ttao.py
    .venv-ttao/bin/python 06_real_v1_synth.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "outputs"
CKPT = OUT / "ttao_tacotron.pth"
CFG = HERE / "ttao_ref" / "config.yaml"
TEXT = "Tacotron turns characters directly into a spectrogram."


def rule(msg):
    print("=" * 72)
    print(msg)
    print("=" * 72)


def monotonic_ratio(attn):
    """attn (T_dec, T_enc) → 각 디코더 스텝이 주목한 인코더 위치의 비감소 비율. 대각선이면 ~1."""
    import numpy as np
    pos = attn.argmax(axis=1)
    return float((np.diff(pos) >= 0).mean()) if len(pos) > 1 else 1.0


def main(ctx=None):
    rule("STEP 6 · 실제 학습된 Tacotron1(ttaoREtw) 로드 · end-to-end 해부")
    if not CKPT.exists():
        print(f"[skip] 체크포인트 없음({CKPT}). `python3 download_ttao.py` 먼저 실행.")
        return ctx
    try:
        import numpy as np
        import torch
        import yaml
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ttao_ref.module import Tacotron
        from ttao_ref.symbols import txt2seq
        from ttao_ref.utils import AudioProcessor
    except Exception as e:  # noqa: BLE001
        print(f"[skip] 의존성/vendored import 실패({type(e).__name__}: {e}). "
              f"`python3 download_ttao.py` 후 `.venv-ttao/bin/python 06_real_v1_synth.py`로 실행.")
        return ctx

    cfg = yaml.safe_load(open(CFG))
    model = Tacotron(**cfg["model"]["tacotron"])
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    # 원저자 추론 방식: encoder·postnet만 eval, mel_decoder는 prenet dropout 유지(Tacotron 관행).
    model.encoder.eval()
    model.postnet.eval()
    step = ck.get("global_step", "?")

    seq = np.asarray(txt2seq(TEXT))
    seq_t = torch.from_numpy(seq).unsqueeze(0)
    with torch.no_grad():
        mel, spec, attn = model(seq_t)
    a = attn[0].cpu().numpy()  # (T_dec, T_enc)

    lin = cfg["model"]["tacotron"]["linear_size"]
    print(f"학습 스텝    : {step}")
    print(f"입력 문자열  : {TEXT!r}  (T_enc={len(seq)})")
    print(f"mel          : {tuple(mel.shape)}  = (1, T_mel, 80)")
    print(f"linear       : {tuple(spec.shape)}  = (1, T_mel, {lin})")
    print(f"alignment    : {tuple(attn.shape)}  = (1, T_dec, T_enc)")

    mono = monotonic_ratio(a)
    peak = float(a.max(axis=1).mean())  # 스텝별 최대 attention의 평균 = 집중도(뾰족함)
    ok_shape = spec.shape[-1] == lin and mel.shape[-1] == 80
    print(f"  {'✅' if ok_shape else '❌'} shape: linear {lin} · mel 80")
    print(f"  {'✅' if peak > 0.1 else '⚠️ '} peak 집중도 = {peak:.3f} (뾰족할수록 학습됨; random은 ~1/T_enc≈{1/len(seq):.3f})")
    print(f"  {'✅' if mono > 0.8 else '⚠️ '} monotonic 비율 = {mono:.2f} (뾰족+대각선일 때만 의미; 대각선일수록 1)")

    # Griffin-Lim 파형 (vendored AudioProcessor)
    ap = AudioProcessor(**cfg["audio"])
    wav = ap.inv_spectrogram(spec[0].cpu().numpy().T)
    ap.save_wav(wav, str(OUT / "real_v1.wav"))
    sr = cfg["audio"]["sample_rate"]
    finite = bool(np.isfinite(wav).all()) and float(np.abs(wav).max()) > 0
    print(f"  {'✅' if finite else '❌'} real_v1.wav: {len(wav) / sr:.2f}s @ {sr}Hz, finite·비영")

    # 스펙트로그램·alignment 그림
    for arr, name, title in [
        (mel[0].cpu().numpy().T, "real_mel.png", "real mel (80-band)"),
        (spec[0].cpu().numpy().T, "real_linear.png", f"real linear ({lin}-bin)"),
    ]:
        plt.figure(figsize=(10, 4))
        plt.imshow(arr, aspect="auto", origin="lower")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(OUT / name, dpi=110)
        plt.close()
    plt.figure(figsize=(6, 5))
    plt.imshow(a.T, aspect="auto", origin="lower", interpolation="none")
    plt.xlabel("decoder step")
    plt.ylabel("encoder step (char)")
    plt.title(f"real alignment (peak={peak:.2f}, mono={mono:.2f})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUT / "real_alignment.png", dpi=110)
    plt.close()
    print("[저장] outputs/real_mel.png · real_linear.png · real_alignment.png · real_v1.wav")

    if ctx is not None:
        ctx["real_attn"] = a
        ctx["real_text"] = TEXT
    return ctx


if __name__ == "__main__":
    main()
