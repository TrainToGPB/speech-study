"""STEP 3 — Positional embedding: 고정 sinusoidal을 더한다.

Self-attention은 순서를 모른다(집합 연산). 그래서 각 frame에 "몇 번째 위치인지"를
알려주는 positional embedding을 더한다. Whisper 인코더는 이걸 **학습하지 않는다** —
sinusoid로 초기화한 뒤 requires_grad=False로 고정(embed_positions.weight).

  hidden_0 = inputs_embeds + embed_positions.weight     # (1, 1500, 512) + (1500, 512)

sinusoid는 위치마다 서로 다른 주파수의 sin/cos 패턴이라, 모델이 상대/절대 위치를
구분할 수 있는 유일무이한 지문을 준다. (1500 = config.max_source_positions.)
"""
import torch

from common import build_context, rule, savefig


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    enc = ctx["encoder"]
    if "inputs_embeds" not in ctx:
        import importlib
        ctx = importlib.import_module("02_conv_stem").main(ctx)
    inputs_embeds = ctx["inputs_embeds"]
    rule("STEP 3 · Positional embedding (고정 sinusoidal 더하기)")

    pos = enc.embed_positions.weight  # (1500, 512)
    with torch.no_grad():
        hidden0 = inputs_embeds + pos  # (1, 1500, 512)
    ctx["hidden0"] = hidden0

    learnable = pos.requires_grad
    print(f"positional emb : {tuple(pos.shape)}  (max_source_positions=1500)")
    print(f"학습 대상?     : requires_grad={learnable}  → {'학습됨' if learnable else '고정 sinusoid (학습 안 함)'}")
    print(f"conv 출력      : {tuple(inputs_embeds.shape)}")
    print(f"+ pos 결과     : {tuple(hidden0.shape)}   = 첫 encoder block의 입력 hidden state")
    print(f"pos 값 예시    : dim 0..5 @ pos 0 = {pos[0, :6].detach().cpu().numpy().round(3)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        p = pos.detach().cpu().numpy()
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), gridspec_kw={"height_ratios": [2, 1]})
        im = ax[0].imshow(p.T, aspect="auto", origin="lower", cmap="RdBu", vmin=-1, vmax=1)
        ax[0].set(title="positional embedding (512 dim x 1500 pos) — sinusoid 지문", ylabel="dim")
        fig.colorbar(im, ax=ax[0], fraction=0.02)
        for d in (0, 4, 20, 100):
            ax[1].plot(p[:200, d], lw=0.9, label=f"dim {d}")
        ax[1].set(title="일부 dim의 위치별 sin/cos (앞 200 pos)", xlabel="position", ylabel="value")
        ax[1].legend(fontsize=8, ncol=4)
        fig.tight_layout()
        savefig(fig, "03_positional.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    return ctx


if __name__ == "__main__":
    main()
