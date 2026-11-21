"""Independent direct-DFT and analytic calibration anchors."""

import cmath
import math
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from spectrumlab.audio import read_pcm16, write_pcm16
from spectrumlab.core import fft, ifft, spectrum, stft, window
from spectrumlab.fixtures import fixtures, SAMPLE_RATE


def direct_dft(values):
    n = len(values)
    return tuple(
        sum(
            value * cmath.exp(-2j * math.pi * k * j / n)
            for j, value in enumerate(values)
        )
        for k in range(n)
    )


class TransformTests(unittest.TestCase):
    def test_independent_dft_small_and_numpy_large(self):
        rng = random.Random(7103)
        for n in [1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096]:
            values = tuple(
                complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)
            )
            actual = np.array(fft(values))
            expected = (
                np.array(direct_dft(values))
                if n <= 64
                else np.fft.fft(np.array(values, dtype=np.complex128))
            )
            with self.subTest(n=n):
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)

    def test_inverse_and_parseval(self):
        rng = random.Random(22)
        for n in [1, 2, 8, 128, 2048]:
            values = tuple(complex(rng.random(), rng.random()) for _ in range(n))
            transformed = fft(values)
            np.testing.assert_allclose(
                ifft(transformed), values, rtol=2e-12, atol=2e-12
            )
            self.assertAlmostEqual(
                sum(abs(x) ** 2 for x in values),
                sum(abs(x) ** 2 for x in transformed) / n,
                places=10,
            )

    def test_invalid_transform_and_spectrum_inputs(self):
        for values in [[], [1, 2, 3], [float("nan"), 1], [complex(0, float("inf")), 0]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                fft(values)
        for rate in [0, -1, float("nan"), float("inf"), True]:
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                spectrum([1, 2], rate)
        for nfft in [0, 3, 4, True]:
            with self.subTest(nfft=nfft), self.assertRaises(ValueError):
                spectrum([1] * 8, 8192, nfft=nfft)
        with self.assertRaises(ValueError):
            spectrum([1j, 0], 8192)
        with self.assertRaises(ValueError):
            window("unknown", 8)


class CalibrationTests(unittest.TestCase):
    def test_coherent_gain_enbw_and_tone_amplitude(self):
        n, fs, amplitude, k = 1024, 8192, 0.2, 37
        values = tuple(amplitude * math.cos(2 * math.pi * k * j / n) for j in range(n))
        for name, gain, enbw_bins in [("rectangular", 1, 1), ("hann", 0.5, 1.5)]:
            result = spectrum(values, fs, name)
            self.assertAlmostEqual(result.coherent_gain, gain, places=14)
            self.assertAlmostEqual(result.enbw_hz, enbw_bins * fs / n, places=12)
            self.assertAlmostEqual(result.amplitude[k], amplitude, places=13)
            self.assertAlmostEqual(result.psd_integral, amplitude**2 / 2, places=13)

    def test_dc_and_nyquist_are_not_doubled(self):
        n, fs = 256, 8192
        for name in ["rectangular", "hann"]:
            dc = spectrum([0.125] * n, fs, name)
            nyquist = spectrum([0.125 * (-1) ** j for j in range(n)], fs, name)
            self.assertAlmostEqual(dc.amplitude[0], 0.125, places=14)
            self.assertAlmostEqual(nyquist.amplitude[-1], 0.125, places=14)
            self.assertAlmostEqual(dc.psd_integral, 0.125**2, places=14)
            self.assertAlmostEqual(nyquist.psd_integral, 0.125**2, places=14)
        singleton = spectrum([0.2], fs, "hann")
        self.assertEqual(len(singleton.amplitude), 1)
        self.assertAlmostEqual(singleton.psd_integral, 0.04)

    def test_psd_integral_and_padding_preserve_weighted_power(self):
        rng = random.Random(77)
        values = [rng.uniform(-0.3, 0.3) for _ in range(173)]
        for name in ["rectangular", "hann"]:
            weights = window(name, len(values))
            expected = sum((x * w) ** 2 for x, w in zip(values, weights)) / sum(
                w * w for w in weights
            )
            for nfft in [256, 1024]:
                result = spectrum(values, 8192, name, nfft)
                self.assertAlmostEqual(result.psd_integral, expected, places=14)
                self.assertAlmostEqual(result.weighted_mean_square, expected, places=14)
            unpadded = spectrum(values, 8192, name, 256)
            padded = spectrum(values, 8192, name, 1024)
            np.testing.assert_allclose(
                padded.amplitude[::4], unpadded.amplitude, rtol=2e-12, atol=2e-12
            )

    def test_stft_frames_times_and_dropped_tail(self):
        data = tuple(math.cos(2 * math.pi * 64 * j / 1024) for j in range(1001))
        result = stft(data, 1024, 128, 64)
        self.assertEqual(result.used_samples, 960)
        self.assertEqual(result.discarded_tail_samples, 41)
        self.assertAlmostEqual(result.times[0], 63.5 / 1024)
        for frame, values in enumerate(result.psd):
            expected = spectrum(data[frame * 64 : frame * 64 + 128], 1024)
            self.assertEqual(values, expected.psd)
        for frame_size, hop in [
            (0, 1),
            (127, 32),
            (2048, 32),
            (128, 0),
            (128, 129),
            (128, True),
        ]:
            with self.assertRaises(ValueError):
                stft(data, 1024, frame_size, hop)


class AudioTests(unittest.TestCase):
    def test_pcm_quantization_peak_bound_and_alias_identity(self):
        sources = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            decoded = {}
            for name, (values, _) in sources.items():
                path = Path(directory) / f"{name}.wav"
                write_pcm16(path, values, SAMPLE_RATE)
                restored, rate = read_pcm16(path)
                self.assertEqual(rate, SAMPLE_RATE)
                self.assertEqual(len(restored), len(values))
                self.assertLessEqual(max(abs(value) for value in restored), 0.3)
                self.assertLessEqual(
                    max(abs(a - b) for a, b in zip(values, restored)),
                    0.5 / 32768 + 1e-14,
                )
                decoded[name] = restored
            self.assertEqual(decoded["alias_high"], decoded["alias_low"])
            self.assertEqual(
                (Path(directory) / "alias_high.wav").read_bytes(),
                (Path(directory) / "alias_low.wav").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
