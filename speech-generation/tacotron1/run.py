"""Tacotron1 step-by-step 러너. build_context를 1회 만들어 STEP 1~4에 관통시킨다.

    python run.py                        # STEP 1~4 (from-scratch 손조립 + Griffin-Lim)
    python run.py --pretrained           # + STEP 5 (Coqui 사전학습 합성, work 우선)
    .venv-ttao/bin/python run.py --real  # STEP 6~7 (실제 학습 v1 해부 + alignment 대비; .venv-ttao 필요)
    python 02_attention_decoder.py       # 단계 개별 실행(선행 스텝 자동 빌드)
"""
import importlib
import sys

STEPS = [
    "01_char_encoder_cbhg",
    "02_attention_decoder",
    "03_postnet_and_loss",
    "04_griffinlim_demo",
]
REAL_STEPS = [
    "06_real_v1_synth",
    "07_alignment_compare",
]


def run_real() -> None:
    """STEP 6~7: vendored ttao_ref(실제 학습 v1)로 해부. .venv-ttao 의존성 필요.

    예시 오디오·datasets를 쓰는 build_context는 호출하지 않는다(06/07은 텍스트만 씀).
    06이 채운 ctx['real_attn']를 07이 재사용해 06 재실행을 피한다.
    """
    print("=" * 72)
    print("Tacotron1 — 실제 학습 v1 해부 (ttaoREtw, .venv-ttao 필요)")
    print("=" * 72)
    ctx = {}
    for name in REAL_STEPS:
        ctx = importlib.import_module(name).main(ctx)
    print("완료 — outputs/real_*.png · real_v1.wav · alignment_compare.png 확인")


def main() -> None:
    if "--real" in sys.argv:
        run_real()
        return

    from common import build_context, rule  # noqa: E402 (core 트랙만 common 필요)
    rule("Tacotron1 — step-by-step (toy 텍스트 + 예시 오디오 1개)")
    ctx = build_context()
    print(f"device={ctx['device']} | 오디오 {len(ctx['audio']) / ctx['sr']:.2f}s "
          f"| toy text {ctx['text']!r}")
    for name in STEPS:
        ctx = importlib.import_module(name).main(ctx)
    if "--pretrained" in sys.argv:
        try:
            importlib.import_module("05_pretrained_synth").main(ctx)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] 05 사전학습 합성 실패: {e}")
    rule("완료 — outputs/ 의 png·wav와 노션 정리를 대조해보세요")


if __name__ == "__main__":
    main()
