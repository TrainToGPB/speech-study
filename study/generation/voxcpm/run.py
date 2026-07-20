"""VoxCPM-0.5B 동작을 예시 한 쌍(참조 오디오 + 목표 문장)으로 STEP 1~6 순서 실행.

각 단계는 개별 실행도 가능(python 01_audio_vae.py ...). 여기서는 컨텍스트(모델·
참조 오디오·조건)를 한 번만 만들어 순서대로 공유하며 돈다. 결과 그림/오디오는 outputs/.

STEP 1 Audio VAE(연속 latent) → 2 text BPE+조건조립 → 3 LocEnc+TSLM+FSQ skeleton
→ 4 RALM 잔차+h_final → 5 LocDiT diffusion(1 patch) → 6 AR 루프+Stop→합성.

tokenizer-free TTS의 전 파이프라인을 실제 가중치로 한 번에 따라간다.
(semantic–acoustic 분리 증거인 t-SNE는 별도: python viz_decoupling_tsne.py)
"""
import importlib

from common import build_context, rule

STEPS = [
    "01_audio_vae",
    "02_text_and_condition",
    "03_tslm_fsq",
    "04_ralm_residual",
    "05_locdit_diffusion",
    "06_generate_and_stop",
]


def main() -> None:
    rule("VoxCPM-0.5B — tokenizer-free TTS step-by-step (참조 1개 + 목표 문장 1개)")
    ctx = build_context()
    print(f"device={ctx['device']} | dtype={ctx['dtype']} | model={ctx['tts'].__class__.__name__}")
    print(f"참조 오디오={len(ctx['ref_audio']) / ctx['sample_rate']:.2f}s | 목표 문장 {ctx['target_text']!r}")
    for name in STEPS:
        mod = importlib.import_module(name)
        ctx = mod.main(ctx)
    rule("완료 — outputs/ 의 png·wav와 위 로그로 tokenizer-free TTS 파이프라인을 순서대로 확인하세요")


if __name__ == "__main__":
    main()
