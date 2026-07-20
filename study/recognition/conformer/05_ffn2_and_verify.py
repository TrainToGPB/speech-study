"""STEP 5 — ½ FFN #2 + final LN, 그리고 블록 전체를 HF와 bit-exact 대조.

macaron 뒤쪽 반쪽 + 블록 마무리 LN:
  residual = x
  x = ffn2_layer_norm(x); x = ffn2(x); x = x*0.5 + residual   # ½ FFN #2
  block_out = final_layer_norm(x)

그다음 STEP 2~5에서 손으로 이어붙인 블록 출력(우리가 만든 것) vs
HF encoder.layers[0].forward(conformer_in, rel_pos)(공식 forward)를 allclose로 검증한다.
whisper STEP 5와 동일한 재현 정확성 확인. rel_shift 인덱싱 같은 실수를 여기서 잡는다.
"""
import torch

from common import build_context, rule


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    if "after_conv" not in ctx:
        import importlib
        ctx = importlib.import_module("04_conv_module").main(ctx)

    layer = ctx["layer"]
    x = ctx["after_conv"]
    rule("STEP 5 · ½ FFN #2 + final LN → 블록 출력, HF 대조 (allclose)")

    with torch.no_grad():
        residual = x
        xln = layer.ffn2_layer_norm(x)
        ff = layer.ffn2(xln)
        merged = ff * 0.5 + residual
        block_out = layer.final_layer_norm(merged)     # 우리가 손으로 만든 블록 출력

        # HF 공식 forward (같은 입력·같은 rel_pos)
        ref = layer(
            ctx["conformer_in"], attention_mask=None,
            relative_position_embeddings=ctx["rel_pos"], output_attentions=False,
        )[0]

    ctx["block_out"] = block_out

    max_diff = (block_out - ref).abs().max().item()
    ok = torch.allclose(block_out, ref, atol=1e-4)
    print(f"½ FFN #2 후     : {tuple(merged.shape)}  ‖·‖ {residual.norm(dim=-1).mean():.2f} → {merged.norm(dim=-1).mean():.2f}")
    print(f"final LN 후     : {tuple(block_out.shape)}  std {block_out.std():.3f} (affine γ로 재정규화)")
    print(f"블록 출력       : {tuple(block_out.shape)}  = (batch, time, d_model)")
    print(f"HF 대조         : max|Δ| = {max_diff:.2e}  → {'일치 ✅ (블록 재현 정확)' if ok else '불일치 ❌'}")

    return ctx


if __name__ == "__main__":
    main()
