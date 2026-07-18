"""STEP 6 — capstone: 블록 24개 스택 → CTC 전사 (실제 ASR).

우리가 뜯어본 블록은 24층 conformer 인코더의 첫 층일 뿐이다. 전체를 돌려:
  1) 우리 STEP 5 블록 출력 == 모델의 layer 0 출력(hidden_states[1]) 재확인
  2) 층을 지날수록 residual stream ‖h‖가 어떻게 변하는지
  3) 마지막 CTC 헤드 → argmax → 실제 LibriSpeech 전사
를 확인한다. whisper STEP 6(전사 capstone)의 conformer 판.
"""
import torch

from common import build_context, rule, savefig


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    model, iv, proc = ctx["model"], ctx["input_values"], ctx["processor"]
    rule("STEP 6 · 24층 스택 + CTC 전사 (capstone)")

    with torch.no_grad():
        out = model(iv, output_hidden_states=True)
    logits = out.logits                          # (1, T, vocab=32)
    hs = out.hidden_states                        # tuple: 층별 입력 + 최종(len = 층수+1)
    pred_ids = logits.argmax(dim=-1)
    transcription = proc.batch_decode(pred_ids)[0]

    # 1) STEP 5 블록 출력과 모델의 layer0 출력(hs[1]) 대조
    if "block_out" in ctx and len(hs) > 1:
        d = (ctx["block_out"] - hs[1]).abs().max().item()
        print(f"STEP5 블록 출력 vs 모델 layer0 출력(hs[1]): max|Δ| = {d:.2e}  "
              f"→ {'일치 ✅' if d < 1e-4 else '불일치 ❌'}")

    # 2) 층별 ‖h‖ (frame별 norm 평균)
    norms = [h[0].norm(dim=-1).mean().item() for h in hs]
    print(f"층별 ‖h‖ (0=입력 … {len(norms) - 2}=마지막블록, {len(norms) - 1}=final LN 후):")
    print("   " + " ".join(f"{n:.1f}" for n in norms))

    # 3) 전사
    ref = ctx["text"]
    print(f"\nlogits         : {tuple(logits.shape)}  (vocab {logits.shape[-1]}, CTC)")
    print(f"전사(예측)      : {transcription!r}")
    print(f"정답(참조)      : {ref!r}")
    match = transcription.strip().upper().replace(" ", "") == ref.strip().upper().replace(" ", "")
    print(f"일치 여부       : {'✅ (공백·대소문자 무시 시 동일)' if match else '⚠️ 차이 있음 (아래 로그 비교)'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(range(len(norms)), norms, marker="o", ms=3)
        ax.set_title("STEP 6 · 층별 residual stream ‖h‖ 성장 (마지막은 final LN 재정규화)")
        ax.set_xlabel("hidden_states index (0=conformer_in)")
        ax.set_ylabel("frame별 ‖h‖ 평균")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        savefig(fig, "06_layer_norms.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    ctx["transcription"] = transcription
    return ctx


if __name__ == "__main__":
    main()
