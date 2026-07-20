"""STEP 4 보조 — self-attention 시각화 집중판 (기본 layer 5).

문제: 1500x1500 attention은 각 행이 softmax(합=1)라 평균 셀값이 ~1/1500 ≈ 0.0007.
선형 컬러스케일이면 대각/sink 소수 셀만 튀고 나머지가 전부 검게 깔려 구조가 안 보인다.
게다가 30초 중 실제 발화는 ~5.9s(50Hz→~293 frame)뿐이라 나머지 패딩 frame이 지도를 희석.

해법(합쳐서 쓴다):
  1) 실제 발화 구간으로 crop → padding 프레임 제거
  2) PowerNorm(gamma<1) → 저~중간 값을 끌어올려 dynamic range 압축
  3) 행별 max 정규화 → 각 query row의 패턴을 균등하게 가시화
  4) head 평균 대신 개별 head도 → 평균이 뭉개는 구조를 드러냄

산출:
  outputs/viz_attention_L{n}_norms.png  — head 평균에 정규화 3종 비교
  outputs/viz_attention_L{n}_heads.png  — 8개 head 각각 (PowerNorm)

사용: python viz_attention.py [layer_index=5] [gamma=0.4]
"""
import sys

import numpy as np
import torch

from common import build_context, rule, savefig

HOP = 0.02  # 인코더 frame 간격(초). 50Hz → 20ms


def main() -> None:
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    rule(f"VIZ · layer {layer} self-attention (crop + PowerNorm + 행정규화 + head별)")

    ctx = build_context()
    enc, mel, audio, sr = ctx["encoder"], ctx["mel"], ctx["audio"], ctx["sr"]
    with torch.no_grad():
        attentions = enc(mel, output_attentions=True).attentions  # tuple, 각 (1, H, 1500, 1500)
    A = attentions[layer][0].float().cpu()  # (H, 1500, 1500)

    dur = len(audio) / sr
    n = min(A.shape[-1], int(round(dur / HOP)) + 8)  # 실제 발화 frame + 여유
    A = A[:, :n, :n]                                  # padding 프레임 제거
    H = A.shape[0]
    sec = n * HOP
    print(f"layer {layer} · head {H} · 원본 1500x1500 → crop {n}x{n} (실제 발화 {dur:.2f}s)")
    print(f"셀값 통계(crop): mean {A.mean():.2e} · max {A.max():.3f} · 균등참조 1/{n}={1/n:.2e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    ext = (0, sec, sec, 0)  # x=key time, y=query time (초)
    mean = A.mean(0).numpy()                                   # head 평균
    vmax = float(np.percentile(mean, 99.5))                    # sink 셀이 스케일 독점하지 않게
    row = mean / mean.max(axis=1, keepdims=True)               # 행별 max 정규화

    # 그림 1 — 정규화 3종 비교 (head 평균)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    panels = [
        ("① 선형 (기존)", mean, dict(vmin=0, vmax=vmax)),
        (f"② PowerNorm γ={gamma}", mean, dict(norm=PowerNorm(gamma=gamma, vmin=0, vmax=vmax))),
        ("③ 행별 max 정규화", row, dict(vmin=0, vmax=1)),
    ]
    for a, (title, mat, kw) in zip(ax, panels):
        im = a.imshow(mat, origin="upper", cmap="magma", aspect="auto", extent=ext, **kw)
        a.set(title=title, xlabel="key time (s)", ylabel="query time (s)")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"layer {layer} head 평균 · 같은 데이터, 다른 스케일 (밝을수록 강한 attention)", y=1.02)
    fig.tight_layout()
    savefig(fig, f"viz_attention_L{layer}_norms.png")
    plt.close(fig)

    # 그림 2 — 개별 head 8종 (PowerNorm)
    cols = 4
    rows = (H + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    ax = np.atleast_1d(ax).ravel()
    for hd in range(H):
        m = A[hd].numpy()
        vm = float(np.percentile(m, 99.5))
        ax[hd].imshow(m, origin="upper", cmap="magma", aspect="auto", extent=ext,
                      norm=PowerNorm(gamma=gamma, vmin=0, vmax=vm))
        ax[hd].set_title(f"head {hd}", fontsize=9)
        ax[hd].tick_params(labelsize=7)
    for k in range(H, len(ax)):
        ax[k].axis("off")
    fig.suptitle(f"layer {layer} · 8 head 개별 (PowerNorm γ={gamma}) — 평균이 뭉개는 head별 구조", y=1.01)
    fig.tight_layout()
    savefig(fig, f"viz_attention_L{layer}_heads.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
