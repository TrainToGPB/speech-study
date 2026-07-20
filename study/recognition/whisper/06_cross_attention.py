"""STEP 6 — 인코더 출력은 어디에 쓰이나: 디코더 cross-attention → 전사.

인코더가 만든 (1500 x 512) 표현은 그 자체로 끝이 아니라, 디코더가 글자를 하나씩
뽑을 때 **cross-attention**으로 참조하는 대상이다. 각 출력 토큰이 인코더의 어느
시각(frame)을 보는지를 그리면, 자연스럽게 "글자 ↔ 오디오 시간" 정렬이 드러난다.

절차:
  1) generate로 전사 토큰 시퀀스를 얻고
  2) 그 토큰들을 teacher-forcing으로 다시 넣어(encoder_outputs=우리 STEP5 출력)
     cross_attentions를 뽑아 (토큰 x 1500 frame) 정렬맵을 그린다.

이 단계는 인코더 본체는 아니지만, "인코더 출력이 왜 그런 모양이어야 하는지"를 닫아주는
capstone이다. (MPS에서 generate 이슈가 나면 CPU로 폴백.)
"""
import torch
from transformers.modeling_outputs import BaseModelOutput

from common import build_context, rule, savefig


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    if "encoder_out" not in ctx:
        import importlib
        ctx = importlib.import_module("05_final_ln").main(ctx)
    model, processor, mel = ctx["model"], ctx["processor"], ctx["mel"]
    rule("STEP 6 · 디코더 cross-attention (인코더 출력 → 전사)")

    def run(dev):
        m = model.to(dev)
        mel_d = mel.to(dev)
        enc_out = BaseModelOutput(last_hidden_state=ctx["encoder_out"].to(dev))
        with torch.no_grad():
            gen = m.generate(mel_d, language="en", task="transcribe", max_new_tokens=64)
            out = m(decoder_input_ids=gen, encoder_outputs=enc_out, output_attentions=True)
        return gen, out.cross_attentions

    try:
        gen, cross = run(model.device)
    except Exception as e:  # noqa: BLE001 — MPS generate 폴백
        print(f"[warn] {model.device}에서 실패({type(e).__name__}); CPU로 폴백.")
        gen, cross = run(torch.device("cpu"))

    text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    ref = ctx.get("text", "")
    tokens = processor.tokenizer.convert_ids_to_tokens(gen[0].tolist())
    n_real = int((len(ctx["audio"]) / ctx["sr"]) / 0.02) + 1  # 실제 발화 frame 수(50Hz)

    # cross: tuple(layers) of (1, heads, tgt_len, 1500) → 모든 (layer, head)를 쌓는다
    stack = torch.stack([c[0] for c in cross]).float().cpu()  # (L, H, tgt, 1500)
    # 깨끗한 시간정렬은 '모든 헤드 평균'이 아니라 Whisper가 지정한 alignment_heads로만 나온다.
    # 나머지 헤드 다수는 오디오 시작부(frame ~6)로 쏠리는 attention-sink라 평균하면 정렬을 덮는다.
    # alignment_heads = 단어 타임스탬프(DTW)에 쓰라고 논문/모델이 골라둔 (layer, head) 목록.
    ah = getattr(model.generation_config, "alignment_heads", None)
    if ah:
        heads = [(int(l), int(h)) for l, h in ah if int(l) < stack.shape[0] and int(h) < stack.shape[1]]
        align = torch.stack([stack[l, h] for l, h in heads]).mean(0)  # (tgt, 1500)
        head_desc = f"alignment_heads {len(heads)}개 지정 평균 (layer,head={heads})"
    else:
        lh = stack.reshape(-1, stack.shape[2], stack.shape[3])
        sink = lh[:, :, :10].sum(-1).mean(-1)
        keep = sink < 0.6
        if int(keep.sum()) < 4:
            keep = torch.ones_like(keep, dtype=torch.bool)
        align = lh[keep].mean(0)
        head_desc = f"sink 제외 {int(keep.sum())}/{lh.shape[0]}개 평균 (alignment_heads 미제공 폴백)"

    print(f"전사 결과      : {text!r}")
    print(f"정답(참고)     : {ref[:len(text) + 10]!r}")
    print(f"디코더 layer   : {len(cross)} · cross-attn shape/layer {tuple(cross[0].shape)}")
    print(f"정렬 헤드      : {head_desc}")
    print(f"정렬맵         : {tuple(align.shape)} = (생성 토큰 {align.shape[0]}, 인코더 frame 1500)")
    peak = align[:, :n_real].argmax(-1)  # 실제 발화 구간에서 각 토큰이 가장 많이 본 frame
    print("토큰 → 최다 참조 시각(초):")
    for tok, fr in list(zip(tokens, peak.tolist())):
        if tok.startswith("<|"):
            continue
        print(f"  {tok.replace('Ġ', '·'):14} → {fr * 0.02:5.2f}s")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        show = min(1500, int((len(ctx["audio"]) / ctx["sr"]) / 0.02) + 40)  # 실제 발화 구간만
        labels = [t.replace("Ġ", "·").replace("<|", "").replace("|>", "") for t in tokens]
        fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(tokens))))
        im = ax.imshow(align[:, :show].numpy(), aspect="auto", origin="upper", cmap="magma",
                       extent=(0, show * 0.02, len(tokens) - 0.5, -0.5))
        ax.set(title="디코더 cross-attention 정렬 (토큰 x 오디오 시간)", xlabel="audio time (s)")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.025)
        fig.tight_layout()
        savefig(fig, "06_cross_attention.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    ctx["transcription"] = text
    return ctx


if __name__ == "__main__":
    main()
