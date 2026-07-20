"""STEP 2 — content-based attention 디코더 + reduction factor r.

인코더 상태를 content-based(Bahdanau) attention으로 매 스텝 가중합하고, 디코더가
스텝당 r개 mel 프레임을 방출한다. 디코더 스텝 수 = ⌈T_mel/r⌉ 이라 r이 클수록 정렬할
스텝이 줄어 학습이 안정·가속된다. random init이라 alignment는 diffuse — 여기선
shape·불변식(행합=1, 스텝수)만 검증한다.
"""
import importlib
import math

import torch

from common import HP, build_context, rule, savefig
from modules import Decoder


def main(ctx=None):
    ctx = ctx or build_context()
    if "enc_out" not in ctx:
        ctx = importlib.import_module("01_char_encoder_cbhg").main(ctx)
    hp: HP = ctx["hp"]
    dev = ctx["device"]
    rule("STEP 2 · content-based attention decoder (r-frame 방출)")

    enc_out = ctx["enc_out"]        # (1, T_enc, 256)
    T_enc = enc_out.size(1)
    T_mel = 50                      # toy 타깃 길이(mel 프레임 수)
    n_steps = math.ceil(T_mel / hp.r)

    torch.manual_seed(0)
    dec = Decoder(enc_dim=hp.enc_out, mel=hp.mel, r=hp.r,
                  prenet=hp.prenet, attn_dim=hp.attn_dim, gru=hp.dec_gru).to(dev).eval()
    with torch.no_grad():
        mels, aligns = dec(enc_out, n_steps)   # (1, n*r, 80), (1, n, T_enc)

    row_sum = aligns.sum(-1)                    # (1, n) — 스텝별 attention 합
    ok = []

    def check(cond, msg):
        ok.append(bool(cond))
        print(f"  {'✅' if cond else '❌'} {msg}")

    print(f"T_enc(문자표현): {T_enc} | toy T_mel: {T_mel} | r: {hp.r}")
    print(f"mels           : {tuple(mels.shape)}  = (1, n_steps*r, mel)")
    print(f"aligns         : {tuple(aligns.shape)}  = (1, n_steps, T_enc)")
    check(tuple(mels.shape) == (1, n_steps * hp.r, hp.mel),
          f"mel 프레임 = n_steps*r = {n_steps}*{hp.r} = {n_steps * hp.r}")
    check(n_steps == math.ceil(T_mel / hp.r),
          f"디코더 스텝 = ⌈T_mel/r⌉ = ⌈{T_mel}/{hp.r}⌉ = {n_steps}")
    check(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4),
          f"attention 행합 = 1 (평균 {row_sum.mean():.4f})")
    print(f"검증           : {'모두 통과 ✅' if all(ok) else '실패 ❌'}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(aligns[0].float().cpu().numpy().T, aspect="auto", origin="lower")
        ax.set_xlabel("decoder step")
        ax.set_ylabel("encoder state")
        ax.set_title("attention alignment (random init → diffuse)")
        fig.colorbar(im, ax=ax)
        savefig(fig, "alignment_randinit.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] alignment png 실패: {e}")

    ctx["mels"], ctx["aligns"] = mels, aligns
    return ctx


if __name__ == "__main__":
    main()
