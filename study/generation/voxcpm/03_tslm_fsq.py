"""STEP 3 — LocEnc → TSLM(MiniCPM semantic backbone) → FSQ semi-discrete skeleton.

여기가 LLM 엔지니어에게 가장 익숙한 지점이다. TSLM은 text LLM(MiniCPM-4-0.5B,
24층)을 그대로 speech backbone으로 쓴다. 흐름(VoxCPMModel._inference의 prefill):

  1) LocEnc(feat_encoder): 과거 latent patch [B,T,P,D] → acoustic embedding [B,T,1024]
     (patch마다 special token을 붙여 작은 bidirectional MiniCPM으로 CLS pooling)
  2) 같은 위치에서 text_embed 또는 feat_embed 하나만 남겨 combined 시퀀스를 만들고
  3) TSLM(base_lm): causal MiniCPM으로 semantic·prosody 골격을 계산
  4) FSQ(fsq_layer): TSLM 출력을 semi-discrete로 만든다 —
        h = tanh(in_proj(x))           # 256차원, (-1,1)로 clip 역할
        skeleton = round(h * 9) / 9    # dim당 {-9..9}/9 → 19레벨 (미분 불가라 학습땐 straight-through)
        out = out_proj(skeleton)       # 다시 1024차원

FSQ는 "맞혀야 할 타깃"이 아니라 continuous flow 한가운데 끼운 **미분가능 bottleneck**
이다. 이산화의 안정성만 빌려오고(안 넣으면 ZH Hard CER 8.87→24.92로 붕괴, Notion),
텍스트 위치엔 적용하지 않는다(오디오 위치에만). 결과 skeleton이 semantic 축을 담당.
"""
import torch

from common import build_context, rule, savefig


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    if "text_token" not in ctx:  # STEP 2가 조건 시퀀스를 채워야 한다
        import importlib
        ctx = importlib.import_module("02_text_and_condition").main(ctx)

    tts, dtype = ctx["tts"], ctx["dtype"]
    rule("STEP 3 · LocEnc → TSLM(MiniCPM) → FSQ semi-discrete skeleton")

    text_token, text_mask = ctx["text_token"], ctx["text_mask"]
    feat, audio_mask = ctx["audio_feat_seq"], ctx["audio_mask"]
    tmask = text_mask.unsqueeze(-1).to(dtype)
    amask = audio_mask.unsqueeze(-1).to(dtype)

    with torch.inference_mode():
        # 1) LocEnc: 과거 latent patch → acoustic embedding
        feat_embed = tts.feat_encoder(feat)             # [B, T, 1024]
        feat_embed = tts.enc_to_lm_proj(feat_embed)     # [B, T, 1024]

        # 2) text/audio 위치를 mask로 합쳐 하나의 입력 시퀀스로
        scale_emb = tts.config.lm_config.scale_emb if tts.config.lm_config.use_mup else 1.0
        text_embed = tts.base_lm.embed_tokens(text_token) * scale_emb
        combined = tmask * text_embed + amask * feat_embed

        # 3) TSLM(base_lm) causal prefill  (kv_cache는 건드리지 않는 순수 forward)
        enc_raw, _ = tts.base_lm(inputs_embeds=combined, is_causal=True)   # [B, T, 1024]

        # 4) FSQ: 오디오 위치에만 semi-discrete skeleton 적용, 텍스트 위치는 원본 유지
        skeleton_full = tts.fsq_layer(enc_raw)
        enc_out = skeleton_full * amask + enc_raw * tmask                  # [B, T, 1024]

        # FSQ 내부(256차원 quantized code)를 직접 재현해 '반이산' 성질을 확인
        fsq = tts.fsq_layer
        h_cont = torch.tanh(fsq.in_proj(enc_raw))       # (-1,1) 연속
        h_quant = torch.round(h_cont * fsq.scale) / fsq.scale  # {-9..9}/9

    ai = ctx["L_text"]  # 첫 오디오 위치 인덱스(= 텍스트 길이)
    print(f"LocEnc 출력     : {tuple(feat_embed.shape)}  (과거 latent patch → acoustic embedding)")
    print(f"combined 입력   : {tuple(combined.shape)}  (text_embed⊕feat_embed, mask로 택1)")
    print(f"TSLM 출력       : {tuple(enc_raw.shape)}  (MiniCPM 24층 causal)")
    print(f"FSQ 후 enc_out  : {tuple(enc_out.shape)}  (오디오 {int(audio_mask.sum())}위치만 quantize)")
    lev = torch.unique(torch.round(h_quant[0, ai:] * fsq.scale)).cpu().tolist()
    print(f"FSQ 256d code   : 오디오 위치 값들이 {len(lev)}개 정수레벨에만 존재 {sorted(int(x) for x in lev)}  → 반이산(semi-discrete)")
    d_before = (enc_raw[0, ai:] ).float().std().item()
    d_after = (enc_out[0, ai:]).float().std().item()
    print(f"양자화 효과     : 오디오 hidden std {d_before:.3f} → {d_after:.3f}  (skeleton으로 눌러 안정화)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hc = h_cont[0, ai:].float().cpu().numpy().reshape(-1)
        hq = h_quant[0, ai:].float().cpu().numpy().reshape(-1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(hc, bins=80, color="tab:gray", alpha=0.7, label="tanh(in_proj) 연속")
        ax[0].hist(hq, bins=80, color="tab:red", alpha=0.7, label="round(·×9)/9 (skeleton)")
        ax[0].set(title="FSQ: 연속 → 반이산 (오디오 위치, 256d)", xlabel="값", ylabel="빈도")
        ax[0].legend(fontsize=8)
        im = ax[1].imshow(enc_out[0].float().cpu().numpy().T, aspect="auto", origin="lower", cmap="viridis")
        ax[1].axvline(ai - 0.5, color="cyan", ls="--", lw=1)
        ax[1].set(title="enc_out (skeleton) — 좌:텍스트 / 우:오디오(FSQ 적용)", xlabel="시퀀스 위치", ylabel="hidden dim")
        fig.colorbar(im, ax=ax[1], fraction=0.02)
        fig.tight_layout()
        savefig(fig, "03_tslm_fsq.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    ctx["feat_embed"] = feat_embed
    ctx["enc_raw"] = enc_raw
    ctx["enc_out"] = enc_out      # FSQ skeleton (semantic 축)
    return ctx


if __name__ == "__main__":
    main()
