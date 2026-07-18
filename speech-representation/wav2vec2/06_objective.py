"""STEP 6·7 — Diversity loss + 전체 objective.

노션:
- Diversity loss L_d: contrastive만 최적화하면 quantizer가 소수 entry만 쓰는 붕괴가 생김.
  batch·time 평균 선택확률의 negative entropy를 더해(=최소화) codebook을 고르게 쓰게 한다.
- 전체 objective: L = L_m + α·L_d (논문 α=0.1).
  L_m은 masked context가 진짜 target을 찾게, L_d는 target codebook이 붕괴하지 않게 한다.

주의: HF 구현의 diversity_loss는 노션의 엔트로피 식과 '동기는 같고 형태가 다른'
perplexity 기반 정규화((num_codevectors - perplexity)/num_codevectors)를 쓴다. 방향(다양성↑)은 동일.
"""
import torch

from common import build_context, compute_mask, rule, OUT


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    model = ctx["model"]
    rule("STEP 6·7 · Diversity loss & 전체 objective")

    if "out" not in ctx:  # STEP 5를 안 거쳤으면 여기서 forward
        import numpy as np
        from transformers.models.wav2vec2.modeling_wav2vec2 import _sample_negative_indices

        compute_mask(ctx)
        K = model.config.num_negatives
        neg_np = _sample_negative_indices((1, ctx["seq_len"]), K, mask_time_indices=ctx["mask_np"])
        neg_t = torch.tensor(neg_np, dtype=torch.long, device=ctx["device"])
        with torch.no_grad():
            ctx["out"] = model(ctx["input_values"], mask_time_indices=ctx["mask_long"],
                               sampled_negative_indices=neg_t)
    out = ctx["out"]

    alpha = model.config.diversity_loss_weight
    G, V = model.quantizer.num_groups, model.quantizer.num_vars
    l_m = out.contrastive_loss.item()
    l_d = out.diversity_loss.item()
    total = out.loss.item()

    print(f"contrastive loss  L_m      : {l_m:.3f}")
    print(f"diversity   loss  L_d      : {l_d:.3f}")
    print(f"가중치           α         : {alpha}")
    print(f"전체 objective  L = L_m + αL_d = {l_m:.3f} + {alpha}×{l_d:.3f} = {l_m + alpha*l_d:.3f}")
    print(f"HF out.loss(검증)          : {total:.3f}")
    print(f"codevector perplexity      : {out.codevector_perplexity.item():.1f} / {G*V}"
          f"  (클수록 codebook을 고르게 사용)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        parts = {"L_m\n(contrastive)": l_m, f"alpha*L_d\n(diversity, a={alpha})": alpha * l_d}
        ax.bar(parts.keys(), parts.values(), color=["#4c72b0", "#dd8452"])
        ax.axhline(total, color="k", ls="--", lw=1, label=f"total L = {total:.2f}")
        ax.set(title="pre-training objective breakdown", ylabel="loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "06_objective.png", dpi=110)
        plt.close(fig)
        print(f"[저장] {OUT/'06_objective.png'}")
    except Exception as e:
        print(f"[warn] 시각화 생략: {e}")
    return ctx


if __name__ == "__main__":
    main()
