"""01→02→03→04 일괄 실행. 개별 스텝도 각각 단독 실행 가능하다.

주의: 03(학습)·02·04(생성 평가)는 MPS에서 수 분~수십 분 걸린다.
"""
import runpy


def main() -> None:
    for step in ("01_data", "02_baseline", "03_train", "04_evaluate"):
        print(f"\n########## {step} ##########")
        runpy.run_path(f"{step}.py", run_name="__main__")


if __name__ == "__main__":
    main()
