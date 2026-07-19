"""STEP 7 — alignment 대비: 우리 from-scratch(random init) vs ttaoREtw(trained).

같은 문장에 대해 (1) 우리 modules.py를 random-init로 돌린 alignment와 (2) 06에서 뽑은
실제 학습 모델의 alignment를 나란히 그린다. 같은 content-based attention 구조인데도
학습 전은 diffuse(번짐), 학습 후는 monotonic 대각선이 되는 걸 눈으로 본다.

modules.py만 임포트한다(torch-only) — common.py는 datasets를 끌어와 .venv-ttao에 없으므로
피하고, vocab·문자 인코딩을 여기서 최소로 정의한다.

    .venv-ttao/bin/python 07_alignment_compare.py     # 06을 내부 호출해 real alignment 확보
"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "outputs"
VOCAB = list(" abcdefghijklmnopqrstuvwxyz.,!?'-")
TEXT = "Tacotron turns characters directly into a spectrogram."


def rule(msg):
    print("=" * 72)
    print(msg)
    print("=" * 72)


def monotonic_ratio(attn):
    import numpy as np
    pos = attn.argmax(axis=1)
    return float((np.diff(pos) >= 0).mean()) if len(pos) > 1 else 1.0


def ours_random_alignment():
    """embed→prenet→CBHG(인코더)→Decoder를 random-init로 돌려 (n_steps, T_enc) alignment."""
    import torch
    import torch.nn as nn
    from modules import Prenet, CBHG, Decoder

    torch.manual_seed(0)
    ids = [VOCAB.index(c) for c in TEXT.lower() if c in VOCAB]
    x = torch.tensor(ids).unsqueeze(0)

    embed = nn.Embedding(len(VOCAB), 256)
    prenet = Prenet(256, (256, 128))
    cbhg = CBHG(128, K=16, proj=(128, 128), gru=128)  # enc_out = 2*128 = 256
    dec = Decoder(enc_dim=256, mel=80, r=2)
    for m in (embed, prenet, cbhg, dec):
        m.eval()  # B=1 BatchNorm은 running stat 사용

    with torch.no_grad():
        e = prenet(embed(x))       # (1, T, 128)
        enc = cbhg(e)              # (1, T, 256)
        _, aligns = dec(enc, n_steps=len(ids))
    return aligns[0].cpu().numpy()  # (n_steps, T_enc)


def real_alignment(ctx):
    """06에서 real alignment 확보 — ctx에 있으면 재사용, 없으면 06.main을 호출."""
    if ctx and "real_attn" in ctx:
        return ctx["real_attn"]
    mod06 = importlib.import_module("06_real_v1_synth")  # 숫자 프리픽스 → importlib
    c = {}
    mod06.main(c)
    return c.get("real_attn")


def main(ctx=None):
    rule("STEP 7 · alignment 대비: 우리(random init) vs ttaoREtw(trained)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ours = ours_random_alignment()
    real = real_alignment(ctx)
    if real is None:
        print("[skip] real alignment 없음 — 06(.venv-ttao)이 먼저 성공해야 함.")
        return ctx

    def peak(x):
        return float(x.max(axis=1).mean())  # 스텝별 최대 attention 평균 = 집중도

    op, rp = peak(ours), peak(real)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(ours.T, aspect="auto", origin="lower", interpolation="none")
    ax[0].set_title(f"ours from-scratch — random init\npeak={op:.3f} (diffuse, ~1/T_enc)")
    ax[1].imshow(real.T, aspect="auto", origin="lower", interpolation="none")
    ax[1].set_title(f"ttaoREtw — trained\npeak={rp:.2f}, mono={monotonic_ratio(real):.2f} (diagonal)")
    for a in ax:
        a.set_xlabel("decoder step")
        a.set_ylabel("encoder step (char)")
    plt.tight_layout()
    plt.savefig(OUT / "alignment_compare.png", dpi=110)
    plt.close()
    print(f"  peak 집중도 — 우리(random)={op:.3f} (diffuse)  |  ttaoREtw(trained)={rp:.2f} (뾰족·대각선)")
    print("[저장] outputs/alignment_compare.png (좌 diffuse · 우 대각선)")
    return ctx


if __name__ == "__main__":
    main()
