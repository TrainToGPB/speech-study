"""wav2vec 2.0 메커니즘을 예시 오디오 하나로 STEP 1~7 쭉 실행한다.

각 단계는 개별 스크립트로도 실행 가능(python 01_feature_encoder.py ...).
여기서는 컨텍스트(모델·오디오)를 한 번만 로드해 순서대로 공유하며 돈다.
결과 그림은 outputs/ 에 저장된다.
"""
import importlib

from common import build_context, rule

STEPS = [
    "01_feature_encoder",
    "02_masking",
    "03_context_network",
    "04_quantizer",
    "05_contrastive",
    "06_objective",
]


def main() -> None:
    rule("wav2vec 2.0 — step-by-step (예시 오디오 1개)")
    ctx = build_context()
    print(f"device={ctx['device']} | 오디오 {len(ctx['audio'])/ctx['sr']:.2f}s"
          f" | 전사 {ctx['text'][:50]!r}")
    for name in STEPS:
        mod = importlib.import_module(name)
        ctx = mod.main(ctx)
    rule("완료 — outputs/ 의 png와 위 로그를 노션 정리와 대조해보세요")


if __name__ == "__main__":
    main()
