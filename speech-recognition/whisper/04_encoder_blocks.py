"""STEP 4 — Transformer encoder blocks: self-attention x N (마스크 없음).

이제 1500개 토큰이 N개의 동일한 encoder block을 통과한다. 각 block은 pre-LayerNorm 구조:

  residual = h
  h = self_attn_layer_norm(h)
  h = self_attention(h)              # 양방향(전체 1500 frame 참조), causal mask 없음
  h = residual + h                   # 잔차 연결
  residual = h
  h = final_layer_norm(h)
  h = fc2(GELU(fc1(h)))              # position-wise MLP (d_model → 4*d_model → d_model)
  h = residual + h

인코더는 디코더와 달리 **causal mask가 없다** — 각 frame이 과거·미래 전체를 본다
(오프라인 전사라 전체 30초를 한 번에 보는 게 당연). 여기서는 HF forward를 그대로
손으로 재현하며 layer마다 shape·attention을 뽑는다.
"""
import torch

from common import build_context, rule, savefig


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    enc, cfg = ctx["encoder"], ctx["config"]
    if "hidden0" not in ctx:
        import importlib
        ctx = importlib.import_module("03_positional").main(ctx)
    rule("STEP 4 · Transformer encoder blocks (self-attention x N)")

    h = ctx["hidden0"]
    hidden_states = [h]
    attentions = []
    with torch.no_grad():
        for layer in enc.layers:
            # WhisperEncoderLayer.forward(hidden, attention_mask, layer_head_mask, output_attentions)
            out = layer(h, None, None, output_attentions=True)
            h, attn = out[0], out[1]  # attn: (1, heads, 1500, 1500)
            hidden_states.append(h)
            attentions.append(attn)
    ctx["hidden_pre_ln"] = h           # 마지막 layer_norm 전 (STEP 5에서 마무리)
    ctx["hidden_states_all"] = hidden_states
    ctx["attentions"] = attentions

    n_layers = len(enc.layers)
    n_heads = cfg.encoder_attention_heads
    print(f"block 수       : {n_layers}  · head {n_heads} · d_model {cfg.d_model} · MLP {cfg.encoder_ffn_dim}")
    print(f"토큰 shape     : {tuple(h.shape)} (모든 layer에서 불변)  · causal mask 없음(양방향)")
    print("\nlayer | out ‖h‖(frame평균) | attn 엔트로피(bits, 낮을수록 집중)")
    for i, (hs, at) in enumerate(zip(hidden_states[1:], attentions)):
        hnorm = hs[0].norm(dim=-1).mean().item()
        p = at[0].mean(0)  # head 평균 (1500,1500)
        ent = (-(p * (p.clamp_min(1e-9)).log2()).sum(-1)).mean().item()
        print(f"  L{i:<2} | {hnorm:16.2f} | {ent:6.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        a0 = attentions[0][0].mean(0).cpu().numpy()
        aL = attentions[-1][0].mean(0).cpu().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
        for a, mat, name in ((ax[0], a0, "layer 0"), (ax[1], aL, f"layer {n_layers - 1}")):
            im = a.imshow(mat, origin="upper", cmap="magma", aspect="auto")
            a.set(title=f"{name} self-attention (head 평균, 1500x1500)",
                  xlabel="key frame (attended-to)", ylabel="query frame")
            fig.colorbar(im, ax=a, fraction=0.046)
        fig.tight_layout()
        savefig(fig, "04_attention.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    return ctx


if __name__ == "__main__":
    main()
