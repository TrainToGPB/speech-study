"""STEP 5 — Coqui 사전학습 모델로 text→음성 end-to-end 합성 (work 우선).

01~04는 random-init from-scratch라 소리가 안 난다. 여기선 실제 사전학습 Tacotron 계열로
문장을 합성해 '진짜 목소리'를 듣는다. Tacotron1 계열(tacotron-DCA) 우선, 없으면
Tacotron2-DDC(v2지만 seq2seq+attention 계보 동일)로 폴백.

coqui-tts는 core .venv를 깨지 않게 격리 venv(.venv-tts)에 설치한다(download.py 참조).
    python download.py
    .venv-tts/bin/python 05_pretrained_synth.py
"""
from common import OUT, rule

TEXT = "Tacotron turns characters directly into a spectrogram."
CANDIDATES = [
    "tts_models/en/ljspeech/tacotron-DCA",    # Tacotron1 계열 우선
    "tts_models/en/ljspeech/tacotron2-DDC",   # 폴백(v2, 같은 seq2seq+attention 계보)
]


def main(ctx=None):
    rule("STEP 5 · Coqui 사전학습 end-to-end 합성")
    try:
        from TTS.api import TTS
    except Exception as e:  # noqa: BLE001
        print(f"[skip] coqui-tts 없음({type(e).__name__}). `python download.py` 후 "
              f"`.venv-tts/bin/python 05_pretrained_synth.py` 로 실행.")
        return ctx

    OUT.mkdir(exist_ok=True)
    for name in CANDIDATES:
        try:
            print(f"  모델 로드 시도: {name}")
            tts = TTS(name, progress_bar=False)
            tag = name.split("/")[-1].replace(".", "_")
            path = OUT / f"tts_{tag}.wav"
            tts.tts_to_file(text=TEXT, file_path=str(path))
            print(f"[저장] {path}")
            print(f"  사용 모델: {name} | 문장: {TEXT!r}")
            return ctx
        except Exception as e:  # noqa: BLE001
            print(f"  [실패] {name}: {type(e).__name__}: {e}")
    print("[skip] 후보 모델 모두 로드 실패 — LOG 막힌 점에 기록.")
    return ctx


if __name__ == "__main__":
    main()
