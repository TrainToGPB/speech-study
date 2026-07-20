"""STEP 3 — 상대위치 Multi-Head Self-Attention (★ Conformer 시그니처).

Transformer-XL 방식의 relative-position attention. 절대 위치 임베딩을 더하는 대신,
query에 두 개의 학습 bias를 더해 score를 content 항과 position 항으로 분해한다:

  q_u = q + pos_bias_u,  q_v = q + pos_bias_v
  matrix_ac = q_u · kᵀ                    # content: "무슨 내용인가" (위치 무관)
  matrix_bd = q_v · (Wp·rel_pos)ᵀ         # position: "얼마나 떨어졌나" (상대거리)
  matrix_bd = rel_shift(matrix_bd)        # (T, 2T-1) → (T, T) 대각 정렬
  scores    = (matrix_ac + matrix_bd) / sqrt(d_k)

rel_shift는 각 query 행을 한 칸씩 밀어 "상대거리 축"을 "절대 key 축"으로 바꾸는 트릭이다.
이 인덱싱을 손으로 재현하면 실수하기 쉬운데, STEP 5의 allclose가 정확성을 잡아준다.
블록 관점: residual = x; x = dropout(self_attn(LN(x), rel_pos)) + residual.
"""
import math

import torch

from common import build_context, rule


def rel_shift(bd: torch.Tensor) -> torch.Tensor:
    """(B, H, T, 2T-1) → (B, H, T, T). HF _apply_relative_embeddings의 5단계 shift와 동일."""
    b, h, t1, t2 = bd.size()  # t2 == 2*t1 - 1
    zero_pad = torch.zeros((b, h, t1, 1), device=bd.device, dtype=bd.dtype)
    padded = torch.cat([zero_pad, bd], dim=-1)      # (b, h, t1, 2t1)
    padded = padded.view(b, h, t2 + 1, t1)          # (b, h, 2t1, t1)
    shifted = padded[:, :, 1:].view(b, h, t1, t2)   # (b, h, t1, 2t1-1)
    return shifted[:, :, :, : t2 // 2 + 1]          # (b, h, t1, t1)


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    if "after_ffn1" not in ctx:
        import importlib
        ctx = importlib.import_module("02_ffn_macaron1").main(ctx)

    layer, rel_pos = ctx["layer"], ctx["rel_pos"]
    x = ctx["after_ffn1"]
    sa = layer.self_attn
    rule("STEP 3 · 상대위치 MHSA (content matrix_ac + position matrix_bd)")

    with torch.no_grad():
        residual = x
        xln = layer.self_attn_layer_norm(x)
        B, T, _ = xln.shape
        nh, dk = sa.num_heads, sa.head_size

        q = sa.linear_q(xln).view(B, -1, nh, dk).transpose(1, 2)  # (B, H, T, dk)
        k = sa.linear_k(xln).view(B, -1, nh, dk).transpose(1, 2)
        v = sa.linear_v(xln).view(B, -1, nh, dk).transpose(1, 2)

        # 위치 임베딩 투영 → (B, H, dk, 2T-1)
        p = sa.linear_pos(rel_pos).view(rel_pos.size(0), -1, nh, dk)
        p = p.transpose(1, 2).transpose(2, 3)

        q_t = q.transpose(1, 2)                       # (B, T, H, dk)
        q_u = (q_t + sa.pos_bias_u).transpose(1, 2)   # (B, H, T, dk)
        q_v = (q_t + sa.pos_bias_v).transpose(1, 2)

        ac = torch.matmul(q_u, k.transpose(-2, -1))   # (B, H, T, T)   content
        bd = torch.matmul(q_v, p)                     # (B, H, T, 2T-1) position(pre-shift)
        bd_shift = rel_shift(bd)                      # (B, H, T, T)
        scores = (ac + bd_shift) / math.sqrt(dk)

        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v)                 # (B, H, T, dk)
        attn = attn.transpose(1, 2).reshape(B, -1, nh * dk)
        attn = sa.linear_out(attn)                    # (B, T, 1024)
        out = layer.self_attn_dropout(attn) + residual

        # HF self_attn과 대조 (같은 서브모듈이므로 bit-exact이어야)
        ref_attn, _ = sa(xln, relative_position_embeddings=rel_pos)
        max_diff = (attn - ref_attn).abs().max().item()

    ctx["after_mhsa"] = out
    ctx["mhsa_ac"] = ac
    ctx["mhsa_bd"] = bd_shift
    ctx["mhsa_probs"] = probs

    print(f"q,k,v          : {tuple(q.shape)}  = (B, head={nh}, T={T}, d_k={dk})")
    print(f"pos_bias_u/v   : {tuple(sa.pos_bias_u.shape)}  ← query에 더하는 두 학습 bias (content/position)")
    print(f"matrix_ac      : {tuple(ac.shape)}  content  |·| 평균 {ac.abs().mean():.3f}")
    print(f"matrix_bd(pre) : {tuple(bd.shape)}  → rel_shift → {tuple(bd_shift.shape)}  position |·| 평균 {bd_shift.abs().mean():.3f}")
    print(f"scores→probs   : {tuple(probs.shape)}  행합 {probs.sum(-1).mean():.3f} (=1)  "
          f"엔트로피 {(-(probs * (probs + 1e-9).log()).sum(-1)).mean() / math.log(2):.2f} bit (max {math.log2(T):.2f})")
    print(f"attn 출력      : {tuple(out.shape)}  ‖·‖ {residual.norm(dim=-1).mean():.2f} → {out.norm(dim=-1).mean():.2f}")
    print(f"HF self_attn 대조: max|Δ| = {max_diff:.2e}  → {'일치 ✅' if max_diff < 1e-4 else '불일치 ❌'}")
    print(f"분해 비중      : content ‖ac‖ {ac.norm():.1f} vs position ‖bd‖ {bd_shift.norm():.1f}  "
          "(position 항이 상대거리 편향을 얼마나 싣는지)")

    return ctx


if __name__ == "__main__":
    main()
