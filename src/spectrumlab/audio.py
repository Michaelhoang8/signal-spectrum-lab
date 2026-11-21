"""Explicit mono PCM16 serialization and decoded full-scale signal units."""

import math
from pathlib import Path
import struct
import wave


def write_pcm16(path, samples, sample_rate):
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("WAV sample rate must be a positive integer")
    samples = tuple(samples)
    if not samples or any(
        not math.isfinite(value) or abs(value) > 0.3 for value in samples
    ):
        raise ValueError("Fixture samples must be finite with absolute peak <= 0.3")
    # Keep the quantized sample itself within the same 0.3 full-scale bound.
    limit = math.floor(0.3 * 32768)
    encoded = [max(-limit, min(limit, round(value * 32768))) for value in samples]
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(struct.pack("<" + "h" * len(encoded), *encoded))


def read_pcm16(path):
    with wave.open(str(Path(path)), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getcomptype() != "NONE"
        ):
            raise ValueError("Only uncompressed mono PCM16 WAV is supported")
        sample_rate = stream.getframerate()
        count = stream.getnframes()
        raw = stream.readframes(count)
        if len(raw) != count * 2:
            raise ValueError("Truncated PCM samples")
    if not count:
        raise ValueError("WAV must contain samples")
    return (
        tuple(value / 32768 for value in struct.unpack("<" + "h" * count, raw)),
        sample_rate,
    )
