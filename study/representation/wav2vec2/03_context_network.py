"""STEP 3 — Context network: 가려진 위치의 문맥 표현 c_t를 만든다.

노션: Z̃에 convolutional positional embedding을 더한 뒤 Transformer g에 넣어
contextual representation C=(c_1,...,c_T)를 얻는다. 마스킹된 위치의 입력은 원래 z_t가
아니라 동일한 MASK 벡터이므로, c_t를 잘 만들려면 앞뒤 speech context를 써야 한다.
c_t로 z_t를 직접 regression하지 않고, quantizer가 만든 discrete target을 '식별'하는 표현을 배운다.
"""
import torch

from common import build_context, compute_mask, rule


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    model, iv = ctx["model"], ctx["input_values"]
    rule("STEP 3 · Context network (Transformer → 문맥표현 C)")

    mask_bool, seq_len = compute_mask(ctx)

    with torch.no_grad():
        # feature_projection: (B,T,512) → hidden(B,T,768) + norm된 extract features(B,T,512)
        z = model.wav2vec2.feature_extractor(iv).transpose(1, 2)
        hidden, extract_norm = model.wav2vec2.feature_projection(z)
        ctx["extract_norm"] = extract_norm  # quantizer가 볼 '원본' 경로 (STEP 4에서 사용)

        # 마스킹 적용 → Transformer
        hidden_masked = model.wav2vec2._mask_hidden_states(hidden, mask_time_indices=mask_bool)
        enc = model.wav2vec2.encoder(hidden_masked)
        context = enc[0]  # (1, T, 768)
    ctx["context"] = context

    masked_pos = torch.where(mask_bool[0])[0]
    t = int(masked_pos[0]) if len(masked_pos) else 0
    print(f"projected hidden : {tuple(hidden.shape)}  (Transformer 입력 차원 768)")
    print(f"context C        : {tuple(context.shape)}")
    print(f"MASK 공유벡터    : masked_spec_embed shape {tuple(model.wav2vec2.masked_spec_embed.shape)}")
    print(f"예시 masked 위치 t={t}")
    print(f"  c_t 앞부분     : {context[0, t, :6].cpu().numpy().round(3)}")
    print("  이 위치의 Transformer 입력은 원본 z_t가 아니라 MASK 벡터였음 → 문맥으로 추론된 표현")
    return ctx


if __name__ == "__main__":
    main()
