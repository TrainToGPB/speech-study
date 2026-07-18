"""Conformer 블록 step-by-step 러너. build_context를 1회 만들어 STEP 1~6에 관통시킨다.

    python run.py           # STEP 1~6 순차 실행 → outputs/*.png + 로그
    python 03_mhsa_relpos.py   # 단계 개별 실행도 가능 (선행 스텝 자동 빌드)
"""
import importlib

from common import build_context, rule

STEPS = [
    "01_frontend",
    "02_ffn_macaron1",
    "03_mhsa_relpos",
    "04_conv_module",
    "05_ffn2_and_verify",
    "06_stack_and_transcribe",
]


def main() -> None:
    rule("Conformer 블록 — step-by-step (예시 오디오 1개 · encoder.layers[0])")
    ctx = build_context()
    print(f"device={ctx['device']} | 오디오 {len(ctx['audio']) / ctx['sr']:.2f}s "
          f"| 전사 {ctx['text'][:50]!r} | model={ctx['config'].name_or_path}")
    for name in STEPS:
        ctx = importlib.import_module(name).main(ctx)
    rule("완료 — outputs/ 의 png와 위 로그를 노션 정리와 대조해보세요")


if __name__ == "__main__":
    main()
