"""VIZ · Conv module 심화 (STEP 4 보조, layer 0).

GLU 게이트가 채널을 어떻게 여닫는지, depthwise conv가 어떤 시간 국소성을 학습했는지 본다.
  outputs/viz_conv_module.png — GLU 게이트 히트맵/분포 + depthwise 커널(수용영역)

사용: python viz_conv_module.py
"""
import numpy as np
import torch

from common import build_context, rule, savefig

HOP_MS = 20  # frame당 밀리초


def main() -> None:
    rule("VIZ · conv module (GLU 게이트 + depthwise 커널 수용영역)")
    ctx = build_context()
    layer, rel_pos = ctx["layer"], ctx["rel_pos"]
    cm = layer.conv_module

    with torch.no_grad():
        x = ctx["conformer_in"]
        h = layer.ffn1(layer.ffn1_layer_norm(x)) * 0.5 + x
        a = layer.self_attn(layer.self_attn_layer_norm(h), relative_position_embeddings=rel_pos)[0]
        after_mhsa = layer.self_attn_dropout(a) + h
        t = cm.layer_norm(after_mhsa).transpose(1, 2)
        pw1 = cm.pointwise_conv1(t)
        _, b_half = pw1.chunk(2, dim=1)
        gate = torch.sigmoid(b_half)[0].cpu().numpy()  # (1024, T)

    kernel = cm.depthwise_conv.weight.detach()[:, 0, :].cpu().numpy()  # (1024, k)
    k = kernel.shape[1]
    taps = np.arange(k) - k // 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    im = axes[0].imshow(gate[:128], aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("GLU 게이트 sigmoid(b) (채널 0~127 × time)")
    axes[0].set_xlabel("time frame")
    axes[0].set_ylabel("channel")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].hist(gate.ravel(), bins=50, color="steelblue")
    axes[1].set_title(f"게이트 값 분포 (평균 {gate.mean():.3f})")
    axes[1].set_xlabel("gate (0=차단 ~ 1=통과)")
    axes[1].set_ylabel("count")

    axes[2].plot(taps, np.abs(kernel).mean(0), "k-", lw=2, label="mean |w| (1024ch)")
    for c in (0, 300, 600, 900):
        axes[2].plot(taps, kernel[c], alpha=0.5, lw=0.8, label=f"ch {c}")
    axes[2].axvline(0, color="gray", ls=":", lw=0.8)
    axes[2].set_title(f"depthwise 커널 (k={k}, 수용영역 ±{k // 2} frame)")
    axes[2].set_xlabel(f"tap (frame offset, {HOP_MS}ms/frame)")
    axes[2].set_ylabel("weight")
    axes[2].legend(fontsize=7)

    fig.suptitle("conv module — GLU 게이트 & depthwise 국소 수용영역", y=1.02)
    fig.tight_layout()
    savefig(fig, "viz_conv_module.png")
    plt.close(fig)

    print(f"게이트 평균 {gate.mean():.3f} · depthwise 커널 {kernel.shape} · "
          f"수용영역 ±{k // 2} frame (~±{k // 2 * HOP_MS}ms)")


if __name__ == "__main__":
    main()
