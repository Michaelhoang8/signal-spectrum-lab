"""Owned synthetic sample generators. Evidence is analyzed after WAV decoding."""

import math
import random

SAMPLE_RATE = 8192
SEED = 20261121


def fixtures():
    fs = SAMPLE_RATE
    tone = lambda frequency, count, amplitude=0.25: tuple(
        amplitude * math.cos(2 * math.pi * frequency * n / fs) for n in range(count)
    )
    close = lambda count: tuple(
        0.12 * math.cos(2 * math.pi * 1000 * n / fs)
        + 0.12 * math.cos(2 * math.pi * 1010 * n / fs)
        for n in range(count)
    )
    rng = random.Random(SEED)
    return {
        "coherent": (
            tone(512, 4096),
            {"kind": "cosine", "frequencies_hz": [512], "peak_amplitudes": [0.25]},
        ),
        "off_bin": (
            tone(513.3, 4096),
            {"kind": "cosine", "frequencies_hz": [513.3], "peak_amplitudes": [0.25]},
        ),
        "close_short": (
            close(512),
            {
                "kind": "two cosines",
                "frequencies_hz": [1000, 1010],
                "peak_amplitudes": [0.12, 0.12],
            },
        ),
        "close_long": (
            close(4096),
            {
                "kind": "two cosines",
                "frequencies_hz": [1000, 1010],
                "peak_amplitudes": [0.12, 0.12],
            },
        ),
        "chirp": (
            tuple(
                0.25
                * math.cos(
                    2 * math.pi * (256 * (n / fs) + 0.5 * (2800 - 256) * (n / fs) ** 2)
                )
                for n in range(8192)
            ),
            {
                "kind": "linear chirp",
                "start_hz": 256,
                "end_at_one_second_hz": 2800,
                "peak_amplitude": 0.25,
            },
        ),
        "noise": (
            tuple(max(-0.29, min(0.29, rng.gauss(0, 0.06))) for _ in range(4096)),
            {
                "kind": "seeded Gaussian noise with explicit +/-0.29 clipping",
                "standard_deviation_before_clipping": 0.06,
                "seed": SEED,
            },
        ),
        "alias_high": (
            tone(6144, 4096),
            {
                "kind": "sampled cosine above Nyquist",
                "generating_frequency_hz": 6144,
                "peak_amplitude": 0.25,
            },
        ),
        "alias_low": (
            tone(2048, 4096),
            {
                "kind": "sampled cosine below Nyquist",
                "generating_frequency_hz": 2048,
                "peak_amplitude": 0.25,
            },
        ),
        "calibration": (
            tuple(
                0.05 + 0.07 * (-1) ** n + 0.1 * math.cos(2 * math.pi * 512 * n / fs)
                for n in range(4096)
            ),
            {
                "kind": "DC plus cosine plus Nyquist",
                "dc": 0.05,
                "tone_hz": 512,
                "tone_peak_amplitude": 0.1,
                "nyquist_peak_amplitude": 0.07,
            },
        ),
    }
