"""STEP 2 — ½ FFN #1 (macaron 앞쪽 반쪽).

Conformer 블록은 attention을 두 개의 half-step FFN이 감싸는 "macaron" 구조다.
그 첫 번째 반쪽:
  residual = x
  x = ffn1_layer_norm(x)                       # pre-LN
  x = intermediate_dense(x)  (1024 → 4096)     # 확장
  x = swish(x)                                 # = SiLU
  x = output_dense(x)        (4096 → 1024)     # 축소
  out = x * 0.5 + residual                     # ★ residual에 0.5 가중 (half-step)

vanilla Transformer의 FFN(1×)과 달리 계수 0.5로 두 번(앞·뒤) 나눠 넣는 게 macaron의 핵심.
dropout들은 eval에서 no-op이라 값에 영향 없다.
"""
import torch

from common import build_context, rule


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    layer = ctx["layer"]
    x = ctx["conformer_in"]  # (1, T, 1024) — layers[0]가 받는 입력과 동일
    rule("STEP 2 · ½ FFN #1 (macaron 앞쪽, residual ×0.5)")

    with torch.no_grad():
        residual = x
        xln = layer.ffn1_layer_norm(x)                     # (1, T, 1024)
        inter = layer.ffn1.intermediate_dense(xln)         # (1, T, 4096)
        act = layer.ffn1.intermediate_act_fn(inter)        # swish
        ff = layer.ffn1.output_dense(act)                  # (1, T, 1024)
        out = ff * 0.5 + residual                          # half-step residual
    ctx["after_ffn1"] = out

    print(f"입력 x         : {tuple(x.shape)}")
    print(f"LN 후          : {tuple(xln.shape)}  mean {xln.mean():.3f} std {xln.std():.3f}")
    print(f"intermediate   : {tuple(inter.shape)}  ← 1024→{inter.shape[-1]} 확장 후 swish")
    print(f"FFN 출력 ff     : {tuple(ff.shape)}  ‖ff‖ 평균 {ff.norm(dim=-1).mean():.2f}")
    print(f"out = ff·0.5+res: {tuple(out.shape)}  ‖·‖ {x.norm(dim=-1).mean():.2f} → {out.norm(dim=-1).mean():.2f}")
    print(f"기여도         : Δ = out-res, ‖Δ‖/‖res‖ 평균 {((out - residual).norm(dim=-1) / residual.norm(dim=-1)).mean():.3f}"
          f"  (0.5 계수라 attention 앞에서 표현을 '살짝' 밀어줌)")

    return ctx


if __name__ == "__main__":
    main()
