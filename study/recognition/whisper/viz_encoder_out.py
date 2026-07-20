"""STEP 5 보조 — encoder 출력 시각화 집중판 (실제 발화 구간 확대).

encoder 출력은 확률이 아니라 부호 있는 활성값 (1500 x 512), mean≈0/std≈1.46이고
소수 'outlier dim'(massive activation)이 스케일을 독점한다. 게다가 실제 발화는 ~293 frame
(5.86s)뿐이라 나머지 패딩이 그림을 희석. 그래서:

  ① 실제 스케일 : 대칭 diverging + |값| 99pct 클립 (outlier dim 억제)
  ② dim별 z-score: 채널마다 시간축으로 표준화 → baseline 제거, 시간 변화만 증폭
  ③ frame×frame 유사도: 시각 간 cosine → 안정 구간(대각 블록) vs 변화 지점

전부 실제 발화+여유(≈330 frame)로 crop. 이건 mechanical representation이라 의미론적 라벨은
없지만, 시각 간 '차이'의 구조를 눈으로 확대해 보는 용도.

사용: python viz_encoder_out.py
"""
import numpy as np
import torch

from common import build_context, rule, savefig

HOP = 0.02  # 50Hz, 20ms/frame


def main() -> None:
    rule("VIZ · encoder 출력 (실제 발화 구간 확대 + 스케일 조정)")
    ctx = build_context()
    enc, mel, audio, sr = ctx["encoder"], ctx["mel"], ctx["audio"], ctx["sr"]
    with torch.no_grad():
        eo = enc(mel).last_hidden_state[0].float().cpu().numpy()  # (1500, 512)

    dur = len(audio) / sr
    n_real = int(round(dur / HOP))
    n = min(eo.shape[0], n_real + 40)     # 발화 + 무음 전환부 조금
    X = eo[:n]                             # (n, 512)
    sec = n * HOP
    print(f"encoder 출력 (1500,512) → crop ({n},512) · 실제 발화 {dur:.2f}s(={n_real}f) + 여유")
    print(f"값 통계(crop): mean {X.mean():+.3f} · std {X.std():.3f} · |값| 99pct {np.percentile(np.abs(X),99):.2f} · max|·| {np.abs(X).max():.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext_t = (0, sec, 512, 0)  # x=time(s), y=dim
    v = float(np.percentile(np.abs(X), 99))           # outlier dim 억제용 대칭 클립
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)           # dim별 z-score (시간축 표준화)

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T                                     # (n, n) frame간 cosine 유사도

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    im0 = ax[0].imshow(X.T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=ext_t)
    ax[0].set(title=f"① 실제 스케일 (대칭 ±{v:.1f} 클립)", xlabel="time (s)", ylabel="d_model dim")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(Z.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, extent=ext_t)
    ax[1].set(title="② dim별 z-score (시간변화 증폭, ±3σ)", xlabel="time (s)", ylabel="d_model dim")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(S, aspect="auto", cmap="magma", origin="upper",
                       extent=(0, sec, sec, 0), vmin=float(np.percentile(S, 2)), vmax=1.0)
    ax[2].set(title="③ frame×frame 유사도 (cosine)", xlabel="time (s)", ylabel="time (s)")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)

    for a in (ax[0], ax[1]):        # 발화 끝(무음 전환) 표시
        a.axvline(dur, color="lime", ls="--", lw=0.8)
    ax[2].axvline(dur, color="cyan", ls="--", lw=0.6); ax[2].axhline(dur, color="cyan", ls="--", lw=0.6)

    fig.suptitle(f"encoder 최종 출력 · 실제 발화 {dur:.2f}s 확대 (점선=발화 끝)", y=1.02)
    fig.tight_layout()
    savefig(fig, "viz_encoder_out.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
