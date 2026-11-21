"""Independent transform identities, odd-window scaling and file boundaries."""

import cmath
import math
from pathlib import Path
import random
import struct
import tempfile
import unittest
import wave

from spectrumlab.audio import read_pcm16
from spectrumlab.core import fft, spectrum


class ReviewTests(unittest.TestCase):
    def test_transform_shift_and_linearity(self):
        rng = random.Random(2919)
        size, shift = 128, 17
        left = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(size)]
        right = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(size)]
        a, b = complex(0.7, -0.3), complex(-0.2, 0.4)
        left_fft, right_fft = fft(left), fft(right)
        combined = fft([a * x + b * y for x, y in zip(left, right)])
        shifted = fft([left[(j - shift) % size] for j in range(size)])
        for k in range(size):
            self.assertLess(
                abs(combined[k] - a * left_fft[k] - b * right_fft[k]), 1e-12
            )
            expected = left_fft[k] * cmath.exp(-2j * math.pi * k * shift / size)
            self.assertLess(abs(shifted[k] - expected), 1e-11)

    def test_odd_observation_hann_scaling_against_direct_sums(self):
        rate = 3200
        for samples in [(0.2, -0.1, 0.05), (0.1, -0.2, 0.17, -0.03, 0.08)]:
            count = len(samples)
            weights = [
                (1 - math.cos(2 * math.pi * j / count)) / 2 for j in range(count)
            ]
            sum_w, sum_w2 = sum(weights), sum(w * w for w in weights)
            for nfft in (8, 16):
                actual = spectrum(samples, rate, "hann", nfft)
                for k in range(nfft // 2 + 1):
                    coefficient = sum(
                        x
                        * w
                        * complex(
                            math.cos(2 * math.pi * k * j / nfft),
                            -math.sin(2 * math.pi * k * j / nfft),
                        )
                        for j, (x, w) in enumerate(zip(samples, weights))
                    )
                    scale = 1 if k in (0, nfft // 2) else 2
                    self.assertAlmostEqual(
                        actual.amplitude[k], scale * abs(coefficient) / sum_w, places=14
                    )
                    self.assertAlmostEqual(
                        actual.psd[k],
                        scale * abs(coefficient) ** 2 / (rate * sum_w2),
                        places=16,
                    )
                power = sum((x * w) ** 2 for x, w in zip(samples, weights)) / sum_w2
                self.assertAlmostEqual(actual.psd_integral, power, places=14)

    def test_reader_rejects_stereo_empty_and_truncated_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.wav"
            for channels, width, samples in [
                (2, 2, b"\0" * 16),
                (1, 1, b"\0" * 8),
                (1, 2, b""),
            ]:
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(channels)
                    output.setsampwidth(width)
                    output.setframerate(3200)
                    output.writeframes(samples)
                with self.assertRaises(ValueError):
                    read_pcm16(path)
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(3200)
                output.writeframes(struct.pack("<hhhh", 12, -17, 99, -100))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Truncated"):
                read_pcm16(path)
