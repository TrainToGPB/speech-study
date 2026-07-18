"""VIZ · 상대위치 attention 심화 (STEP 3 보조, 기본 layer 0).

content(matrix_ac) vs position(matrix_bd)로 분해해 상대위치가 attention에 무엇을 더하는지 본다.
  outputs/viz_relpos_attention_L{n}_decomp.png  — content / position / 최종 probs (head 평균)
  outputs/viz_relpos_attention_L{n}_heads.png   — head별 attention (PowerNorm)

사용: python viz_relpos_attention.py [layer_idx=0] [gamma=0.4]
"""
import math
import sys

import numpy as np
import torch

from common import build_context, rule, savefig

HOP = 0.02  # conformer frame 간격(초) ~50Hz


def rel_shift(bd: torch.Tensor) -> torch.Tensor:
    b, h, t1, t2 = bd.size()
    zero_pad = torch.zeros((b, h, t1, 1), device=bd.device, dtype=bd.dtype)
    padded = torch.cat([zero_pad, bd], dim=-1).view(b, h, t2 + 1, t1)
    return padded[:, :, 1:].view(b, h, t1, t2)[:, :, :, : t2 // 2 + 1]


def attn_decomp(layer, h, rel_pos):
    """블록 앞단(½FFN)까지 통과시킨 뒤 self_attn의 content/position/probs를 돌려준다."""
    sa = layer.self_attn
    h1 = layer.ffn1(layer.ffn1_layer_norm(h)) * 0.5 + h
    xln = layer.self_attn_layer_norm(h1)
    B, T, _ = xln.shape
    nh, dk = sa.num_heads, sa.head_size
    q = sa.linear_q(xln).view(B, -1, nh, dk).transpose(1, 2)
    k = sa.linear_k(xln).view(B, -1, nh, dk).transpose(1, 2)
    p = sa.linear_pos(rel_pos).view(rel_pos.size(0), -1, nh, dk).transpose(1, 2).transpose(2, 3)
    q_t = q.transpose(1, 2)
    q_u = (q_t + sa.pos_bias_u).transpose(1, 2)
    q_v = (q_t + sa.pos_bias_v).transpose(1, 2)
    ac = torch.matmul(q_u, k.transpose(-2, -1))
    bd = rel_shift(torch.matmul(q_v, p))
    probs = torch.softmax((ac + bd) / math.sqrt(dk), dim=-1)
    return ac, bd, probs


def main() -> None:
    layer_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    rule(f"VIZ · layer {layer_idx} 상대위치 attention (content vs position 분해)")
    ctx = build_context()
    encoder, rel_pos = ctx["encoder"], ctx["rel_pos"]

    with torch.no_grad():
        h = ctx["conformer_in"]
        for i in range(layer_idx):
            h = encoder.layers[i](h, attention_mask=None, relative_position_embeddings=rel_pos)[0]
        ac, bd, probs = attn_decomp(encoder.layers[layer_idx], h, rel_pos)

    acm = ac.mean(1)[0].cpu().numpy()
    bdm = bd.mean(1)[0].cpu().numpy()
    pm = probs.mean(1)[0].cpu().numpy()
    nh = probs.shape[1]
    T = pm.shape[0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    # fig1: content / position / 최종 probs
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, arr, title, div in [
        (axes[0], acm, "content matrix_ac (head 평균)", True),
        (axes[1], bdm, "position matrix_bd (head 평균)", True),
        (axes[2], pm, "최종 attention probs (head 평균)", False),
    ]:
        if div:
            lim = float(np.abs(arr).max())
            im = ax.imshow(arr, aspect="auto", origin="upper", cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(arr, aspect="auto", origin="upper", cmap="magma", norm=PowerNorm(gamma))
        ax.set_title(title)
        ax.set_xlabel("key frame")
        ax.set_ylabel("query frame")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"layer {layer_idx} · 상대위치 attention 분해  (T={T}, ~50Hz)", y=1.02)
    fig.tight_layout()
    savefig(fig, f"viz_relpos_attention_L{layer_idx}_decomp.png")
    plt.close(fig)

    # fig2: head별 probs
    cols = 4
    rows = (nh + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    ph = probs[0].cpu().numpy()
    for hd in range(nh):
        ax = axes.flat[hd]
        ax.imshow(ph[hd], aspect="auto", origin="upper", cmap="magma", norm=PowerNorm(gamma))
        ax.set_title(f"head {hd}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(nh, rows * cols):
        axes.flat[j].axis("off")
    fig.suptitle(f"layer {layer_idx} · head별 attention (PowerNorm γ={gamma})", y=1.01)
    fig.tight_layout()
    savefig(fig, f"viz_relpos_attention_L{layer_idx}_heads.png")
    plt.close(fig)

    dist = np.average(np.abs(np.subtract.outer(np.arange(T), np.arange(T))), weights=pm)
    print(f"content |ac| 평균 {np.abs(acm).mean():.3f} · position |bd| 평균 {np.abs(bdm).mean():.3f}")
    print(f"probs 대각 집중도: 평균 |i-j| = {dist:.1f} frame (~{dist * HOP * 1000:.0f}ms)")


if __name__ == "__main__":
    main()
