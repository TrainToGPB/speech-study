"""STEP 6 — 전체 AR 루프 + Stop Predictor로 실제 합성 (capstone).

STEP 1~5를 한 patch에 대해 따라왔다. 실제 생성은 이 과정을 patch 단위로
autoregressive하게 반복한다:  p(Z|T) = ∏ p(z_i | T, Z_<i).

  각 스텝: h_final(=skeleton+residual) → LocDiT로 다음 latent patch 생성
           → 그 patch를 LocEnc로 다시 임베딩해 TSLM/RALM을 한 칸 전진(forward_step)
           → Stop Predictor(stop_head)가 "지금 멈출까?"를 2-class로 판단.
  continuous 생성이라 discrete EOS 토큰이 없어 별도 정지 신호(Stop Predictor)가 필수.
  마지막에 AudioVAE.decode로 latent 시퀀스 전체를 waveform으로 복원.

이 STEP은 공식 고수준 API(model.generate)로 참조 화자를 복제(voice cloning)해
합성하고, stop_head에 forward hook을 걸어 **정지 신호가 스텝마다 어떻게 커지는지**를
관찰한다(= 언제 발화를 끝낼지 학습된 지점).
"""
import time

import torch

from common import build_context, rule, savefig, OUT


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    tts, model = ctx["tts"], ctx["model"]
    sr = ctx["sample_rate"]
    rule("STEP 6 · AR 루프 + Stop Predictor로 voice cloning 합성 (capstone)")

    # stop_head 출력(스텝별 정지 logit)을 forward hook으로 수집
    stop_logits = []

    def hook(_m, _inp, out):
        stop_logits.append(out.detach().float().cpu())

    handle = tts.stop_head.register_forward_hook(hook)
    t0 = time.time()
    try:
        wav = model.generate(
            text=ctx["target_text"],
            prompt_wav_path=ctx["ref_wav_path"],   # 참조 화자
            prompt_text=ctx["prompt_text"],
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=True,
            denoise=False,
            retry_badcase=False,                   # 단일 실행(hook 스텝 수를 깔끔히)
        )
    finally:
        handle.remove()
    gen_s = time.time() - t0

    dur = len(wav) / sr
    n_steps = len(stop_logits)
    out_wav = OUT / "06_voice_clone.wav"
    import soundfile as sf
    sf.write(str(out_wav), wav, sr)

    print(f"목표 문장       : {ctx['target_text']!r}")
    print(f"참조 화자       : {ctx['prompt_text'][:50]!r}...")
    print(f"AR 스텝(=patch) : {n_steps}개  (patch당 {tts.patch_size}프레임 = {tts.patch_size / (sr / tts.audio_vae.chunk_size) * 1000:.0f}ms)")
    print(f"합성 파형       : {len(wav):,} sample = {dur:.2f}s @ {sr}Hz")
    print(f"속도            : {gen_s:.2f}s 소요 → RTF {gen_s / dur:.2f} (1보다 작으면 실시간보다 빠름)")
    print(f"저장            : {out_wav}")

    if n_steps:
        probs = torch.softmax(torch.cat(stop_logits, dim=0), dim=-1)[:, 1].numpy()  # P(stop)
        fired = [i for i, p in enumerate(probs) if p > 0.5]
        print(f"Stop Predictor  : P(stop) 최종 {probs[-1]:.2f}, 0.5 초과 첫 스텝 {fired[0] if fired else '없음(max_len 도달)'}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(11, 3.2))
            ax.plot(probs, color="tab:red")
            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
            ax.set(title="Stop Predictor: P(stop) over AR 스텝 (연속 생성의 EOS 대체)",
                   xlabel="AR 스텝(patch)", ylabel="P(stop)", ylim=(-0.02, 1.02))
            fig.tight_layout()
            savefig(fig, "06_stop_predictor.png")
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 시각화 생략: {e}")

    ctx["wav"] = wav
    return ctx


if __name__ == "__main__":
    main()
