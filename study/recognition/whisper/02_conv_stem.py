"""STEP 2 — Conv stem: 2개의 Conv1d + GELU (80x3000 → 1500 x d_model).

인코더의 첫 연산은 transformer가 아니라 작은 CNN 2층이다. 여기서 (a) mel 채널(80)을
d_model(512)로 올리고, (b) 시간축을 절반으로 줄인다.

  conv1: Conv1d(80 → 512, kernel 3, stride 1, padding 1) → GELU     # 길이 3000 유지
  conv2: Conv1d(512 → 512, kernel 3, stride 2, padding 1) → GELU     # stride 2 → 3000→1500
  permute (B, C, T) → (B, T, C)                                       # transformer가 쓰는 (time, feat)

stride 2 한 번이 프레임레이트를 100Hz→50Hz(20ms/frame)로 낮춘다. 이후 인코더의 모든
attention은 이 1500개 토큰 위에서 일어난다. (HF forward와 동일하게 F.gelu 사용.)
"""
import torch
import torch.nn.functional as F

from common import build_context, rule, savefig


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    enc, mel = ctx["encoder"], ctx["mel"]
    rule("STEP 2 · Conv stem (2x Conv1d + GELU, 시간축 절반)")

    with torch.no_grad():
        after_c1 = F.gelu(enc.conv1(mel))   # (1, 512, 3000)
        after_c2 = F.gelu(enc.conv2(after_c1))  # (1, 512, 1500)
        inputs_embeds = after_c2.permute(0, 2, 1)  # (1, 1500, 512)
    ctx["inputs_embeds"] = inputs_embeds

    def desc(conv):
        return (f"in {conv.in_channels}→out {conv.out_channels}, "
                f"k{conv.kernel_size[0]} s{conv.stride[0]} p{conv.padding[0]}")

    print(f"입력 log-Mel   : {tuple(mel.shape)}")
    print(f"conv1 ({desc(enc.conv1)})")
    print(f"  → GELU 후    : {tuple(after_c1.shape)}   (채널 80→512, 길이 3000 유지)")
    print(f"conv2 ({desc(enc.conv2)})")
    print(f"  → GELU 후    : {tuple(after_c2.shape)}   (stride 2 → 3000→1500, 50Hz/20ms)")
    print(f"permute 후     : {tuple(inputs_embeds.shape)}   = (batch, time=1500, d_model=512) transformer 입력")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(11, 5))
        im0 = ax[0].imshow(after_c1[0].cpu().numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax[0].set(title="conv1 출력 (512 x 3000)", ylabel="channel")
        fig.colorbar(im0, ax=ax[0], fraction=0.02)
        im1 = ax[1].imshow(after_c2[0].cpu().numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax[1].set(title="conv2 출력 (512 x 1500) — 시간축 절반", xlabel="frame", ylabel="channel")
        fig.colorbar(im1, ax=ax[1], fraction=0.02)
        fig.tight_layout()
        savefig(fig, "02_conv_stem.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    return ctx


if __name__ == "__main__":
    main()
