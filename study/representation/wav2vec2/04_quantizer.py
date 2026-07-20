"""STEP 4 — Quantizer: 원본 z_t로 discrete target q_t를 만든다.

노션: 음성엔 미리 정해진 vocabulary가 없으므로 target도 학습한다. product quantization —
G개 codebook group마다 V개 entry, 각 group에서 하나씩 고른 codevector를 이어붙여 linear
projection해 q_t. forward는 group별 argmax(hard), backward는 Gumbel-softmax straight-through.
논문 설정 G=2, V=320 → entry 640개지만 조합은 최대 320^2=102,400.
"""
import numpy as np
import torch

from common import build_context, compute_mask, rule, OUT


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    model, iv = ctx["model"], ctx["input_values"]
    rule("STEP 4 · Quantizer (원본 Z → discrete target Q)")

    compute_mask(ctx)
    if "extract_norm" not in ctx:
        with torch.no_grad():
            z = model.wav2vec2.feature_extractor(iv).transpose(1, 2)
            _, ctx["extract_norm"] = model.wav2vec2.feature_projection(z)
    extract_norm = ctx["extract_norm"]  # (1, T, 512)  ← 마스킹 안 된 원본 경로

    q = model.quantizer
    G, V = q.num_groups, q.num_vars
    with torch.no_grad():
        quantized, perplexity = q(extract_norm)          # (1, T, codevector_dim=256)
        projected_q = model.project_q(quantized)          # (1, T, 256)  contrastive 공간

        # 각 frame이 고른 group별 entry id 복원 (forward = argmax)
        B, T, _ = extract_norm.shape
        logits = q.weight_proj(extract_norm).view(B * T * G, V)
        ids = logits.argmax(-1).view(B, T, G)             # (1, T, 2)
    ctx["projected_q"] = projected_q

    print(f"codebook 구성   : G={G} groups × V={V} entries → 조합 최대 {V**G:,}")
    print(f"quantized q_t   : {tuple(quantized.shape)} → project_q → {tuple(projected_q.shape)}")
    print(f"perplexity      : {perplexity.item():.1f} / {G*V} (전체 frame 기준 code 다양성)")
    print("  ※ loss에 쓰이는 건 masked 위치만의 perplexity → STEP 6 참고(값이 더 작음)")
    print(f"frame별 code id  (앞 8개, [g0 g1]):")
    print(f"  {ids[0, :8].cpu().numpy().tolist()}")
    uniq = {tuple(x) for x in ids[0].cpu().numpy().tolist()}
    print(f"이 발화에서 등장한 고유 codeword 조합: {len(uniq)}개 / frame {T}개")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ids_np = ids[0].cpu().numpy()  # (T, 2)
        fig, ax = plt.subplots(2, 1, figsize=(11, 3.2), sharex=True)
        for g in range(G):
            ax[g].plot(ids_np[:, g], drawstyle="steps-mid", lw=0.9)
            ax[g].set(ylabel=f"group {g}\nentry id", ylim=(-5, V + 5))
        ax[0].set_title(f"per-frame selected codebook entry (G={G}, V={V})")
        ax[-1].set_xlabel("frame index")
        fig.tight_layout()
        fig.savefig(OUT / "04_codebook_ids.png", dpi=110)
        plt.close(fig)
        print(f"[저장] {OUT/'04_codebook_ids.png'}")
    except Exception as e:
        print(f"[warn] 시각화 생략: {e}")
    return ctx


if __name__ == "__main__":
    main()
