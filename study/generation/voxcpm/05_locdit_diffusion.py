"""STEP 5 — LocDiT: 다음 latent patch를 noise에서 flow-matching으로 생성.

softmax·argmax로 이산 토큰을 뽑는 LLM과 달리, VoxCPM은 조건 h_final을 받아
**연속 latent patch**를 diffusion(정확히는 rectified-flow / flow-matching)으로 낸다.

  UnifiedCFM.forward:
    z ~ N(0, I)                      # t=1, 순수 noise  shape (b, 64, patch_size)
    t: 1 → 0 을 n_timesteps로 Euler 적분          (직전 patch(cond)를 이어 그리는 outpainting)
    각 step에서 LocDiT estimator가 속도장 v를 예측 → x ← x - dt·v
    반환: t=0의 x = 다음 latent patch (b, 64, 2)

CFG(classifier-free guidance): 배치를 2배로 만들어 조건부(mu=h_final)와
무조건부(mu=0)를 함께 추론하고 cfg_value로 외삽한다(VoiceBox식). cfg↑ → 더 또렷/
과장. 이 STEP은 _inference의 '첫 생성 patch'를 그대로 재현하고, cfg·step 수의 효과를 본다.
"""
import torch

from common import build_context, rule, savefig


def _sample(tts, mu, cond, n_timesteps, cfg_value, seed=0):
    torch.manual_seed(seed)  # 내부 z=randn 재현용
    with torch.inference_mode():
        patch = tts.feat_decoder(
            mu=mu, n_timesteps=n_timesteps, patch_size=tts.patch_size,
            cond=cond, cfg_value=cfg_value,
        )  # [b, 64, patch_size]
    return patch


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    if "h_final" not in ctx:  # STEP 4의 조건이 필요
        import importlib
        ctx = importlib.import_module("04_ralm_residual").main(ctx)

    tts = ctx["tts"]
    rule("STEP 5 · LocDiT flow-matching (h_final → 다음 latent patch, CFG)")

    # _inference의 첫 생성 스텝과 동일한 조건: 마지막 위치 h_final + 직전(마지막) patch
    mu = ctx["h_final"][:, -1, :].contiguous()                        # [1, h_dit]
    cond = ctx["audio_feat_seq"][:, -1, ...].transpose(1, 2).contiguous()  # [1, 64, patch]
    print(f"조건 mu(h_final): {tuple(mu.shape)}  | 직전 patch cond: {tuple(cond.shape)}  | patch_size={tts.patch_size}")

    # 기준(reference): 많은 step으로 수렴시킨 patch
    ref = _sample(tts, mu, cond, n_timesteps=50, cfg_value=2.0, seed=0)
    print(f"\nstep 수렴(같은 noise seed, cfg=2.0 / 50-step 대비 L2 거리):")
    for nt in (2, 5, 10, 30):
        p = _sample(tts, mu, cond, n_timesteps=nt, cfg_value=2.0, seed=0)
        print(f"  n_timesteps={nt:>2} → patch {tuple(p.shape)}, |patch|={p.float().norm():.2f}, ref대비 L2 {(p-ref).float().norm():.3f}")

    print(f"\nCFG 세기 효과(같은 noise seed, 10-step):")
    patches_cfg = {}
    for cfg in (1.0, 2.0, 3.0):
        p = _sample(tts, mu, cond, n_timesteps=10, cfg_value=cfg, seed=0)
        patches_cfg[cfg] = p
        print(f"  cfg={cfg} → |patch|={p.float().norm():.2f}, std={p.float().std():.3f}  (cfg↑ → 조건 방향으로 더 외삽)")

    print(f"\nnoise seed 다양성(cfg=2.0, 10-step):")
    for sd in (0, 1, 2):
        p = _sample(tts, mu, cond, n_timesteps=10, cfg_value=2.0, seed=sd)
        print(f"  seed={sd} → |patch|={p.float().norm():.2f}  (연속 생성이라 seed마다 다른 실현값)")

    gen_patch = patches_cfg[2.0]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        for a, cfg in zip(ax, (1.0, 2.0, 3.0)):
            im = a.imshow(patches_cfg[cfg][0].float().cpu().numpy(), aspect="auto", origin="lower", cmap="magma")
            a.set(title=f"생성 patch (cfg={cfg})", xlabel="patch pos", ylabel="latent dim")
        fig.colorbar(im, ax=ax.tolist(), fraction=0.02)
        fig.suptitle("LocDiT가 noise에서 생성한 다음 latent patch (64 × patch_size)")
        savefig(fig, "05_locdit_diffusion.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")

    ctx["gen_patch"] = gen_patch
    return ctx


if __name__ == "__main__":
    main()
