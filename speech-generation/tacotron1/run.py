"""Tacotron1 step-by-step 러너. build_context를 1회 만들어 STEP 1~4에 관통시킨다.

    python run.py                     # STEP 1~4 (from-scratch 손조립 + Griffin-Lim)
    python run.py --pretrained        # + STEP 5 (Coqui 사전학습 합성, work 우선)
    python 02_attention_decoder.py    # 단계 개별 실행(선행 스텝 자동 빌드)
"""
import importlib
import sys

from common import build_context, rule

STEPS = [
    "01_char_encoder_cbhg",
    "02_attention_decoder",
    "03_postnet_and_loss",
    "04_griffinlim_demo",
]


def main() -> None:
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
