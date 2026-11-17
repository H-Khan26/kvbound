"""Audio front end: a 16 kHz waveform to the log-mel features Whisper's encoder takes."""

import io
import wave
from functools import cache

import numpy as np
from torch import Tensor
from transformers import WhisperFeatureExtractor

SAMPLE_RATE = 16000
# Seconds of audio in one encoder window; shorter clips are zero-padded and longer ones cut.
WINDOW_SECONDS = 30


@cache
def extractor(n_mels: int) -> WhisperFeatureExtractor:
    """Whisper's front end: 25 ms Hann windows every 10 ms, n_mels mel bands, log10, clamped to
    8 below the peak and scaled to about [-1, 1]."""
    return WhisperFeatureExtractor(
        feature_size=n_mels, sampling_rate=SAMPLE_RATE, chunk_length=WINDOW_SECONDS
    )


def features(audio_16k: np.ndarray, n_mels: int = 80) -> Tensor:
    """Log-mel features [n_mels, 3000] (float32) of mono 16 kHz audio in [-1, 1].

    The audio is zero-padded or cut to 30 s first, so every clip gives one encoder window.
    """
    if audio_16k.ndim != 1:
        raise ValueError(f"audio has shape {audio_16k.shape}; expected mono samples")
    out = extractor(n_mels)(audio_16k, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    return out.input_features[0]


def decode_wav(data: bytes) -> np.ndarray:
    """Samples of a 16-bit mono 16 kHz PCM WAV file as float32 in [-1, 1)."""
    with wave.open(io.BytesIO(data)) as f:
        layout = (f.getnchannels(), f.getsampwidth(), f.getframerate())
        if layout != (1, 2, SAMPLE_RATE):
            raise ValueError(f"WAV is (channels, bytes, rate) {layout}; expected (1, 2, 16000)")
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0
