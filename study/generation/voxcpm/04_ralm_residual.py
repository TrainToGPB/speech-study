"""STEP 4 — RALM(acoustic 잔차) → h_final = semantic skeleton + acoustic residual.

VoxCPM 설계의 핵심. 안정성(skeleton)과 표현력(residual)을 두 계층으로 나눈다.

  RALM(residual_lm, 6층): (FSQ skeleton + 과거 acoustic embedding)을 받아
     speaker·미세 prosody 같은 **acoustic 잔차**를 추정한다.
  h_final = lm_to_dit_proj(skeleton) + res_to_dit_proj(residual)     ← LocDiT 조건

명시적 supervision 없이도 이 분해에서 semantic–acoustic 분리가 암묵적으로 생긴다
(Notion Fig 2 t-SNE: skeleton은 speaker-agnostic, residual은 화자별 군집 —
 이 증거는 viz_decoupling_tsne.py로 재현). RALM을 빼면 EN WER 4.34로 악화(Notion).

_inference의 prefill을 이어서(STEP 3의 enc_out을 입력) residual을 계산한다.
※ 실제 AR 루프는 '마지막 위치' hidden만 조건으로 쓰지만, 여기선 이해를 위해
   전체 위치의 skeleton/residual 기여도를 함께 본다(투영 가중치는 실제 가중치).
"""
import torch

from common import build_context, rule, savefig


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    if "enc_out" not in ctx:  # STEP 3의 skeleton이 필요
        import importlib
        ctx = importlib.import_module("03_tslm_fsq").main(ctx)

    tts, dtype = ctx["tts"], ctx["dtype"]
    rule("STEP 4 · RALM acoustic 잔차 → h_final = skeleton + residual")

    enc_out, feat_embed = ctx["enc_out"], ctx["feat_embed"]  # STEP 3
    audio_mask = ctx["audio_mask"]
    amask = audio_mask.unsqueeze(-1).to(dtype)

    with torch.inference_mode():
        # RALM 입력 = FSQ skeleton + 오디오 위치의 acoustic embedding
        residual_in = enc_out + amask * feat_embed
        res_raw, _ = tts.residual_lm(inputs_embeds=residual_in, is_causal=True)  # [B, T, 1024]

        # 두 축을 LocDiT 차원으로 투영해 더한 것이 최종 조건 h_final
        dit_from_skeleton = tts.lm_to_dit_proj(enc_out)     # semantic 기여
        dit_from_residual = tts.res_to_dit_proj(res_raw)    # acoustic 기여
        h_final = dit_from_skeleton + dit_from_residual     # [B, T, h_dit]

    ai = ctx["L_text"]  # 첫 오디오 위치
    skel_norm = dit_from_skeleton[0, ai:].float().norm(dim=-1).mean().item()
    res_norm = dit_from_residual[0, ai:].float().norm(dim=-1).mean().item()
    fin_norm = h_final[0, ai:].float().norm(dim=-1).mean().item()
    print(f"RALM 입력       : enc_out(skeleton) + audio·feat_embed  → {tuple(residual_in.shape)}")
    print(f"RALM 출력       : {tuple(res_raw.shape)}  (MiniCPM 6층 — acoustic 잔차)")
    print(f"h_final         : {tuple(h_final.shape)}  = skeleton투영 + residual투영 (LocDiT 조건)")
    print(f"기여 크기(오디오): |skeleton|={skel_norm:.2f}  |residual|={res_norm:.2f}  |h_final|={fin_norm:.2f}")
    print(f"  → skeleton이 semantic 골격, residual이 그 위에 speaker·음색을 더하는 구도")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = dit_from_skeleton[0, ai:].float().norm(dim=-1).cpu().numpy()
        r = dit_from_residual[0, ai:].float().norm(dim=-1).cpu().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        x = range(len(s))
        ax[0].plot(x, s, label="skeleton (semantic)", color="tab:blue")
        ax[0].plot(x, r, label="residual (acoustic)", color="tab:orange")
        ax[0].set(title="오디오 위치별 두 축 기여 크기(norm)", xlabel="오디오 patch", ylabel="‖·‖")
        ax[0].legend(fontsize=8)
        im = ax[1].imshow(h_final[0, ai:].float().cpu().numpy().T, aspect="auto", origin="lower", cmap="coolwarm")
        ax[1].set(title="h_final (LocDiT 조건) — 오디오 위치", xlabel="오디오 patch", ylabel="dit dim")
        fig.colorbar(im, ax=ax[1], fraction=0.02)
        fig.tight_layout()
        savefig(fig, "04_ralm_residual.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    ctx["res_raw"] = res_raw
    ctx["h_final"] = h_final
    return ctx


if __name__ == "__main__":
    main()
