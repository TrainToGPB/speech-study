"""Stage 2 실습 — Whisper 전사 + pyannote diarization 이어붙이기 (스켈레톤).

Stage 2 진입 시 완성한다. 목표: 두 모듈을 따로 돌려 병합할 때의 error propagation 체감.

완성 체크리스트:
  [ ] faster-whisper로 word-level 타임스탬프 전사
  [ ] pyannote/speaker-diarization-3.1로 화자 세그먼트
  [ ] 타임스탬프 기준 병합 → "화자별 스크립트"
  [ ] overlap/화자 경계에서 어긋나는 지점 로깅

설치:
  uv pip install faster-whisper pyannote.audio
  huggingface-cli login   # pyannote 3.1 약관 동의 필요
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "shared" / "sample_audio" / "two_speakers.wav"


def transcribe(audio_path):
    """faster-whisper 전사 → [(start, end, text), ...]"""
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cuda", compute_type="int8_float16")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
    return [(s.start, s.end, s.text) for s in segments]


def diarize(audio_path):
    """pyannote diarization → [(start, end, speaker), ...]"""
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipe.to("cuda")
    diar = pipe(str(audio_path))
    return [(t.start, t.end, spk) for t, _, spk in diar.itertracks(yield_label=True)]


def merge(asr, diar):
    """타임스탬프로 전사 세그먼트에 화자 라벨 붙이기. (error propagation 관찰 지점)"""
    raise NotImplementedError("Stage 2 진입 시 구현: 각 ASR 세그먼트 중앙 시각이 속한 화자 매칭")


def main() -> None:
    print("Stage 2 스켈레톤. transcribe/diarize/merge를 채우세요.")
    print(f"대상 오디오: {AUDIO}")


if __name__ == "__main__":
    main()
