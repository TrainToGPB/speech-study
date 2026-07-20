"""STEP 2 — 사전학습 whisper-base가 한국어 eval에서 얼마나 틀리는지(baseline CER)."""
from common import evaluate_cer, load_model, load_processor, load_splits, rule, seed_everything


def main() -> None:
    seed_everything()
    processor = load_processor()
    model, device = load_model()
    _, ds_eval = load_splits(n_train=1, n_eval=100)

    rule(f"STEP 2 · baseline CER (whisper-base, n={len(ds_eval)}, device={device})")
    score, pairs = evaluate_cer(model, processor, device, ds_eval)
    print(f"baseline CER = {score:.4f}")
    rule("예시 (정답 → 예측)")
    for ref, hyp in pairs[:5]:
        print(f"  정답: {ref}")
        print(f"  예측: {hyp}\n")


if __name__ == "__main__":
    main()
