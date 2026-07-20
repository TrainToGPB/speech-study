"""STEP 1 — 한 샘플로 전처리 전 과정을 눈으로 본다.

raw waveform → 16k → log-Mel (1,80,3000) → (텍스트) tokenizer label ids → decode 복원.
LLM tokenization과의 대응: 오디오는 feature_extractor가, 텍스트는 tokenizer가 담당.
한국어는 띄어쓰기 편차가 커서 WER이 흔들린다 → 문자 단위 CER을 주 지표로 쓴다.
"""
import numpy as np

from common import LANG, TASK, load_processor, load_splits, rule, seed_everything


def main() -> None:
    seed_everything()
    processor = load_processor()
    _, ds_eval = load_splits(n_train=1, n_eval=8)
    ex = ds_eval[0]
    audio, text = ex["audio"], ex["text"]

    rule("STEP 1a · raw waveform")
    arr = np.asarray(audio["array"], dtype=np.float32)
    print(f"sr={audio['sampling_rate']}Hz  len={len(arr)}  "
          f"dur={len(arr)/audio['sampling_rate']:.2f}s  전사={text!r}")

    rule("STEP 1b · log-Mel feature")
    feat = processor.feature_extractor(
        arr, sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    print(f"input_features shape={tuple(feat.shape)}  (1, 80 mel, 3000 frame=30s·100Hz)")
    print(f"값 범위 min={feat.min():.2f} max={feat.max():.2f}")

    rule("STEP 1c · 텍스트 → label ids → 복원")
    ids = processor.tokenizer(text).input_ids
    back = processor.tokenizer.decode(ids)
    print(f"label ids({len(ids)})[:12] = {ids[:12]}")
    print(f"decode 복원 = {back!r}")
    print(f"special 제외 = {processor.tokenizer.decode(ids, skip_special_tokens=True)!r}")

    rule("메모")
    print(f"- 학습 시 디코더 입력은 '<|startoftranscript|><|{LANG[:2]}|><|{TASK}|>...' prefix로 시작")
    print("- collator가 label의 앞 BOS를 하나 잘라내고 -100으로 패딩 마스킹")
    print("- 지표: CER(주)·WER(참고) — 한국어는 CER이 표준")


if __name__ == "__main__":
    main()
