"""STEP 4 — 파인튜닝 후 CER, 그리고 전/후 비교 (capstone).

같은 eval subset에서 base와 fine-tuned의 CER을 각각 재고, (정답/base/ft) triple을 본다.
"""
from common import (
    CKPT_DIR,
    evaluate_cer,
    load_model,
    load_processor,
    load_splits,
    rule,
    seed_everything,
)


def main() -> None:
    seed_everything()
    _, ds_eval = load_splits(n_train=1, n_eval=100)

    base_proc = load_processor()
    base_model, device = load_model()
    ft_proc = load_processor()
    ft_model, _ = load_model(from_dir=str(CKPT_DIR))

    rule(f"STEP 4 · CER 전/후 (n={len(ds_eval)}, device={device})")
    base_cer, base_pairs = evaluate_cer(base_model, base_proc, device, ds_eval)
    ft_cer, ft_pairs = evaluate_cer(ft_model, ft_proc, device, ds_eval)
    delta = base_cer - ft_cer
    print(f"baseline CER   = {base_cer:.4f}")
    print(f"fine-tuned CER = {ft_cer:.4f}")
    print(f"개선(Δ)        = {delta:+.4f}  ({'좋아짐' if delta > 0 else '나빠짐'})")

    rule("예시 (정답 / base / fine-tuned)")
    for (ref, b_hyp), (_, f_hyp) in list(zip(base_pairs, ft_pairs))[:5]:
        print(f"  정답      : {ref}")
        print(f"  base      : {b_hyp}")
        print(f"  finetuned : {f_hyp}\n")


if __name__ == "__main__":
    main()
