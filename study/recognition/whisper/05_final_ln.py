"""STEP 5 — Final LayerNorm → encoder 출력 (그리고 HF와 일치 검증).

마지막 block 뒤에 LayerNorm 하나가 더 있다. 그 결과가 인코더의 최종 산출물:

  encoder_out = layer_norm(h)        # (1, 1500, 512)

이 (1500 x 512) 시퀀스가 인코더가 하는 일의 전부다. 각 행은 약 20ms(50Hz)의 오디오를
요약한 문맥 벡터이고, 디코더는 이걸 cross-attention으로 읽어 글자를 뽑는다(STEP 6).

STEP 2~5에서 우리가 손으로 재현한 파이프라인이 HF WhisperEncoder와 정확히 같은지
`enc(mel).last_hidden_state`와 allclose로 확인한다(재현이 맞다는 증거).
"""
import torch

from common import build_context, rule, savefig


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    enc, mel = ctx["encoder"], ctx["mel"]
    if "hidden_pre_ln" not in ctx:
        import importlib
        ctx = importlib.import_module("04_encoder_blocks").main(ctx)
    rule("STEP 5 · Final LayerNorm → encoder 출력 (HF 대조)")

    with torch.no_grad():
        manual = enc.layer_norm(ctx["hidden_pre_ln"])         # 우리가 손으로 만든 것
        ref = enc(mel).last_hidden_state                       # HF 공식 forward
    ctx["encoder_out"] = manual

    max_diff = (manual - ref).abs().max().item()
    ok = torch.allclose(manual, ref, atol=1e-4)
    print(f"encoder 출력   : {tuple(manual.shape)}  = (batch, time=1500, d_model=512)")
    print(f"HF 대조        : max|Δ| = {max_diff:.2e}  → {'일치 ✅ (재현 정확)' if ok else '불일치 ❌'}")
    print(f"프레임 의미    : 1500 frame x 20ms = 30s. 각 행 = 그 시각의 문맥 요약 벡터")
    print(f"값 통계        : mean {manual.mean():.3f} · std {manual.std():.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m = manual[0].cpu().numpy()
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(m.T, aspect="auto", origin="lower", cmap="magma")
        ax.set(title="encoder 최종 출력 (512 dim x 1500 frame) — 디코더가 읽는 표현",
               xlabel="frame (20ms each)", ylabel="d_model dim")
        fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout()
        savefig(fig, "05_encoder_out.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    return ctx


if __name__ == "__main__":
    main()
