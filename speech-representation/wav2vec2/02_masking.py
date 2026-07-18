"""STEP 2 — Masking: context 경로에서만 latent span을 가린다.

노션: 마스킹은 waveform이 아니라 CNN 출력 Z 위에서 한다. 시작점을 확률 p=0.065로
뽑아 M=10 frame을 학습가능 공유벡터 e_mask로 치환. span이 겹칠 수 있어 실제로는
전체의 약 49%가 가려진다. 가장 중요한 비대칭: Transformer는 masked Z̃를 보지만
quantizer는 원본 Z를 본다(문제엔 답을 숨기고, 정답 생성엔 원본을 준다).
"""
import numpy as np

from common import build_context, compute_mask, rule, OUT


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    rule("STEP 2 · Masking (context 경로의 latent span 가리기)")

    mask_bool, seq_len = compute_mask(ctx)
    m = ctx["mask_np"][0]  # (T,) bool
    n_masked = int(m.sum())

    # 겹침을 합친 연속 구간 통계
    spans, cur = [], 0
    for v in m:
        if v:
            cur += 1
        elif cur:
            spans.append(cur); cur = 0
    if cur:
        spans.append(cur)

    print(f"frame 수 T        : {seq_len}")
    print(f"마스킹 파라미터   : p={0.065}, M=10 (span 시작확률·길이)")
    print(f"가려진 frame      : {n_masked}/{seq_len} = {100*n_masked/seq_len:.1f}%")
    print(f"연속 span 개수    : {len(spans)}개, 평균 길이 {np.mean(spans):.1f} frame"
          f" (≈ {np.mean(spans)*20:.0f} ms)" if spans else "연속 span 없음")
    print(f"마스킹된 위치 예시: {np.where(m)[0][:15].tolist()} ...")
    print("비대칭 → Transformer: Z̃(masked) 입력 / quantizer: Z(원본) 입력")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 1.6))
        ax.imshow(m[None, :], aspect="auto", cmap="Greys", interpolation="nearest",
                  extent=(0, seq_len, 0, 1))
        ax.set(title=f"mask pattern (black=masked, {n_masked}/{seq_len} = {100*n_masked/seq_len:.0f}%)",
               xlabel="frame index", yticks=[])
        fig.tight_layout()
        fig.savefig(OUT / "02_mask_pattern.png", dpi=110)
        plt.close(fig)
        print(f"[저장] {OUT/'02_mask_pattern.png'}")
    except Exception as e:
        print(f"[warn] 시각화 생략: {e}")
    return ctx


if __name__ == "__main__":
    main()
