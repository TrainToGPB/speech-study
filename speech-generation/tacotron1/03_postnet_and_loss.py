"""STEP 3 — post-processing net(CBHG) + L1 손실.

디코더가 뱉은 mel 시퀀스를 두 번째 CBHG(K=8)로 다시 훑어 linear-scale spectrogram을
예측한다. 전체 시퀀스를 양방향으로 보므로 autoregressive 디코더가 놓친 부분을 보정한다.
손실은 디코더 mel + post-net linear 둘 다 단순 L1(동일 가중치).
"""
import importlib

import torch
import torch.nn.functional as F

from common import HP, build_context, rule
from modules import PostNet


def main(ctx=None):
    ctx = ctx or build_context()
    if "mels" not in ctx:
        ctx = importlib.import_module("02_attention_decoder").main(ctx)
    hp: HP = ctx["hp"]
    dev = ctx["device"]
    rule("STEP 3 · post-net(mel→linear) + L1 손실")

    mels = ctx["mels"]  # (1, T, 80)
    torch.manual_seed(0)
    postnet = PostNet(mel=hp.mel, K=hp.k_post, linear_bins=hp.linear_bins).to(dev).eval()
    with torch.no_grad():
        linear = postnet(mels)  # (1, T, 513)

    # toy 타깃(같은 shape)으로 L1 데모
    loss_mel = F.l1_loss(mels, torch.zeros_like(mels))
    loss_lin = F.l1_loss(linear, torch.zeros_like(linear))
    loss = loss_mel + loss_lin

    ok = []

    def check(cond, msg):
        ok.append(bool(cond))
        print(f"  {'✅' if cond else '❌'} {msg}")

    T = mels.size(1)
    print(f"mel(디코더)     : {tuple(mels.shape)}  (80-band)")
    print(f"linear(post-net): {tuple(linear.shape)}  (n_fft/2+1 = {hp.linear_bins} bins)")
    check(tuple(linear.shape) == (1, T, hp.linear_bins), f"linear = (1, {T}, {hp.linear_bins})")
    check(torch.isfinite(loss),
          f"L1 = mel {loss_mel:.4f} + linear {loss_lin:.4f} = {loss:.4f} (finite)")
    print(f"검증            : {'모두 통과 ✅' if all(ok) else '실패 ❌'}")

    ctx["linear"], ctx["loss"] = linear, loss
    return ctx


if __name__ == "__main__":
    main()
