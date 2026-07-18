"""STEP 1 — Front-end: waveform → conformer 입력 (1, T, 1024) + rel-pos 임베딩.

conformer 인코더 앞단은 wav2vec2와 동일하다:
  input_values (1, N) → feature_extractor(7-layer CNN, ~320x 다운샘플) → (1, 512, T)
  → transpose → (1, T, 512) → feature_projection(LN·Linear 512→1024) → (1, T, 1024)
그리고 encoder가 rel-pos 임베딩 (1, 2T-1, 1024)을 한 번 만들어 모든 층 self_attn에 넘긴다.
(conformer 인코더는 wav2vec2와 달리 pos_conv를 쓰지 않고 위치정보는 전적으로 상대위치로 준다.)

front-end 상세는 [[speech-representation/wav2vec2]]의 STEP 1(feature encoder)에서 깊게 다뤘다.
여기선 conformer 블록 입력의 shape·성질만 확인하고 STEP 2로 넘어간다. 실제 forward는
build_context에서 1회 수행돼 ctx로 공유된다.
"""
from common import build_context, rule, savefig


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    iv, extract = ctx["input_values"], ctx["extract"]
    x, rel_pos, cfg = ctx["conformer_in"], ctx["rel_pos"], ctx["config"]
    rule("STEP 1 · Front-end (waveform → conformer 입력 + rel-pos 임베딩)")

    T = x.shape[1]
    sec = len(ctx["audio"]) / ctx["sr"]
    print(f"input_values   : {tuple(iv.shape)}  ({sec:.2f}s · {ctx['sr']}Hz waveform, 정규화됨)")
    print(f"feature_extract: {tuple(extract.shape)}  → CNN 7층이 시간축 {iv.shape[1]}→{T} (~{T/sec:.0f}Hz, 20ms/frame)")
    print(f"conformer_in   : {tuple(x.shape)}  = (batch, time={T}, d_model={cfg.hidden_size})  ← feature_projection 후")
    print(f"rel_pos 임베딩  : {tuple(rel_pos.shape)}  = (1, 2T-1={2 * T - 1}, {cfg.hidden_size})  ← ±(T-1) 상대거리")
    print(f"모델 구성      : hidden {cfg.hidden_size} · head {cfg.num_attention_heads} · "
          f"{cfg.num_hidden_layers}층 · FFN {cfg.intermediate_size} · act {cfg.hidden_act} · "
          f"depthwise k{cfg.conv_depthwise_kernel_size}")
    print(f"conformer_in 통계: mean {x.mean():.4f} · std {x.std():.4f} · frame별 ‖·‖ 평균 {x.norm(dim=-1).mean():.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr = x[0].abs().cpu().numpy().T  # (d_model, T)
        fig, ax = plt.subplots(figsize=(11, 3.2))
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"STEP 1 · conformer 입력 |x|  (d_model={x.shape[-1]} × time={T})")
        ax.set_xlabel("time frame (~50Hz)")
        ax.set_ylabel("d_model")
        fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout()
        savefig(fig, "01_frontend.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    return ctx


if __name__ == "__main__":
    main()
