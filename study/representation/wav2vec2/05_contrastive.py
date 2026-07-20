"""STEP 5 — Contrastive task: 정답 q_t를 K개 distractor와 구별한다.

노션: 마스킹된 각 위치 t에서 후보집합 Q_t = {q_t(정답), q̃_1..q̃_K(같은 발화 내 다른 masked
위치에서 뽑은 distractor)}. c_t와 각 후보의 cosine similarity를 temperature κ로 나눠 InfoNCE.
분자=positive 점수, 분모=모든 후보 점수 합. 논문 K=100, κ=0.1.
"""
import numpy as np
import torch
import torch.nn.functional as F

from common import build_context, compute_mask, rule, OUT


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    model, iv = ctx["model"], ctx["input_values"]
    device = ctx["device"]
    rule("STEP 5 · Contrastive task (InfoNCE: 진짜 q_t 골라내기)")

    mask_bool, seq_len = compute_mask(ctx)
    from transformers.models.wav2vec2.modeling_wav2vec2 import _sample_negative_indices

    K = model.config.num_negatives
    kappa = model.config.contrastive_logits_temperature
    neg_np = _sample_negative_indices((1, seq_len), K, mask_time_indices=ctx["mask_np"])
    neg_t = torch.tensor(neg_np, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(iv, mask_time_indices=ctx["mask_long"], sampled_negative_indices=neg_t)
    c = out.projected_states               # (1, T, 256)  문맥표현(projected)
    qp = out.projected_quantized_states    # (1, T, 256)  정답 target(projected)

    # positive similarity
    pos_sim = torch.cosine_similarity(c, qp, dim=-1)[0]  # (T,)

    # distractor 모으기 (HF 내부와 동일한 gather)
    hs = qp.shape[-1]
    negs = qp.view(-1, hs)[neg_t.view(-1)].view(1, seq_len, K, hs).permute(2, 0, 1, 3)  # (K,1,T,256)
    neg_sim = torch.cosine_similarity(c.unsqueeze(0), negs, dim=-1)[:, 0, :]  # (K, T)

    mask = mask_bool[0]
    mp = torch.where(mask)[0]
    t = int(mp[0])
    print(f"K(distractor)   : {K},  κ(temperature): {kappa}")
    print(f"projected c / q : {tuple(c.shape)} / {tuple(qp.shape)}")
    print(f"masked 위치 평균 cosine sim → positive {pos_sim[mask].mean():.3f}"
          f" vs distractor {neg_sim[:, mask].mean():.3f}")

    # 한 위치 t의 InfoNCE 재구성
    logits = torch.cat([pos_sim[t].view(1), neg_sim[:, t]]) / kappa  # (1+K,)
    ce = F.cross_entropy(logits.view(1, -1), torch.zeros(1, dtype=torch.long, device=device))
    rank = int((logits[0] < logits[1:]).sum())  # positive보다 높은 distractor 수
    print(f"위치 t={t}: positive logit {logits[0]:.2f}, distractor 평균 {logits[1:].mean():.2f}")
    print(f"  → positive 순위 {rank+1}/{K+1},  이 위치 L_m ≈ {ce.item():.3f}")
    print(f"HF 공식 contrastive_loss(전체 masked 합): {out.contrastive_loss.item():.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(neg_sim[:, t].cpu().numpy(), bins=25, alpha=0.7, label="distractors")
        ax.axvline(pos_sim[t].item(), color="crimson", lw=2, label="positive q_t")
        ax.set(title=f"cosine sim at masked t={t}: c_t vs candidates",
               xlabel="cosine similarity", ylabel="count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "05_contrastive_sim.png", dpi=110)
        plt.close(fig)
        print(f"[저장] {OUT/'05_contrastive_sim.png'}")
    except Exception as e:
        print(f"[warn] 시각화 생략: {e}")

    ctx["out"] = out
    return ctx


if __name__ == "__main__":
    main()
