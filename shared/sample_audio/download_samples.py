"""공통 샘플 오디오 내려받기.

모든 스테이지 실습이 같은 오디오를 쓰도록 짧은 영어 음성 클립 몇 개를 준비한다.
LibriSpeech 계열 공개 샘플을 torchaudio로 받거나, 없으면 합성 톤으로 폴백한다.

사용:
    python shared/sample_audio/download_samples.py
"""
from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path(__file__).parent
TARGET_SR = 16000


def _save(name: str, wav: np.ndarray, sr: int) -> None:
    path = OUT / name
    sf.write(path, wav, sr)
    dur = len(wav) / sr
    print(f"  saved {path.name}  ({dur:.1f}s @ {sr}Hz)")


def try_torchaudio_sample() -> bool:
    """torchaudio 내장 예제 음성을 받는다. 성공하면 True."""
    try:
        import torchaudio
    except Exception as e:  # noqa: BLE001
        print(f"  torchaudio import 실패({e}) → 폴백")
        return False

    # torchaudio가 제공하는 공개 음성 URL (LibriSpeech 계열 짧은 발화)
    url = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    try:
        import torch  # noqa: F401
        local = torchaudio.utils.download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
        )
        wav, sr = torchaudio.load(local)
        wav = wav.mean(0)  # mono
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            sr = TARGET_SR
        _save("speech_en.wav", wav.numpy(), sr)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  torchaudio 샘플 다운로드 실패({e}) → 폴백")
        return False


def synth_speech_fallback() -> None:
    """네트워크가 없을 때: 음성처럼 들리는 합성 신호(포먼트 근사)를 만든다.

    실제 발화는 아니지만 waveform·spectrogram·codec 실습에는 충분하다.
    """
    print("  합성 폴백 신호 생성 (speech_en.wav, 포먼트 근사)")
    sr = TARGET_SR
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # 시간에 따라 변하는 기본주파수 + 포먼트 몇 개 → 모음 흐름 흉내
    f0 = 120 + 40 * np.sin(2 * np.pi * 0.5 * t)
    sig = np.zeros_like(t)
    for k, amp in [(1, 1.0), (2, 0.5), (3, 0.33), (4, 0.2)]:
        sig += amp * np.sin(2 * np.pi * f0 * k * t)
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) ** 2  # 음절처럼 진폭 포락선
    sig = 0.9 * (sig * env) / np.max(np.abs(sig * env))
    _save("speech_en.wav", sig.astype(np.float32), sr)


def synth_two_speakers() -> None:
    """두 화자 대화 흉내 (Stage 2 diarization 실습용): 앞 절반/뒤 절반 음높이 다르게.

    실제 2화자 코퍼스가 아니므로 diarization 정확도보다 '파이프라인 연결'을 익히는 용도.
    """
    if (OUT / "two_speakers.wav").exists():
        return
    sr = TARGET_SR
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) ** 2
    sig2 = np.zeros_like(t)
    half = len(t) // 2
    f0a, f0b = 110.0, 190.0
    for k, amp in [(1, 1.0), (2, 0.5), (3, 0.3)]:
        sig2[:half] += amp * np.sin(2 * np.pi * f0a * k * t[:half])
        sig2[half:] += amp * np.sin(2 * np.pi * f0b * k * t[half:])
    sig2 = 0.9 * (sig2 * env) / np.max(np.abs(sig2 * env))
    _save("two_speakers.wav", sig2.astype(np.float32), sr)


def main() -> None:
    print(f"샘플 오디오 준비 → {OUT}")
    if not try_torchaudio_sample():
        synth_speech_fallback()
    synth_two_speakers()  # Stage 2용, 항상 준비
    print("완료.")


if __name__ == "__main__":
    main()
