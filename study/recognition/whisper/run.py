"""Whisper 인코더 메커니즘을 예시 오디오 하나로 STEP 1~6 순서대로 실행한다.

각 단계는 개별 스크립트로도 실행 가능(python 01_logmel.py ...). 여기서는 컨텍스트
(모델·오디오·log-Mel)를 한 번만 로드해 순서대로 공유하며 돈다. 결과 그림은 outputs/.

STEP 1 log-Mel → 2 conv stem → 3 positional → 4 encoder blocks → 5 final LN(+HF 대조)
→ 6 cross-attention(인코더 출력이 전사로 쓰이는 모습).
"""
import importlib

from common import build_context, rule

STEPS = [
    "01_logmel",
    "02_conv_stem",
    "03_positional",
    "04_encoder_blocks",
    "05_final_ln",
    "06_cross_attention",
]


def main() -> None:
    rule("Whisper encoder — step-by-step (예시 오디오 1개)")
    ctx = build_context()
    print(f"device={ctx['device']} | model={ctx['config'].name_or_path} "
          f"| 오디오 {len(ctx['audio']) / ctx['sr']:.2f}s | 전사 {ctx['text'][:50]!r}")
    for name in STEPS:
        mod = importlib.import_module(name)
        ctx = mod.main(ctx)
    rule("완료 — outputs/ 의 png와 위 로그로 인코더 파이프라인을 순서대로 확인하세요")


if __name__ == "__main__":
    main()
