"""Reconcile the committed PCM bytes, calibrated arrays and measured evidence."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import unittest

import numpy as np

from spectrumlab.audio import read_pcm16
from spectrumlab.core import spectrum, stft

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class EvidenceTests(unittest.TestCase):
    def test_pcm_spectra_stft_hashes_and_timing_reconcile(self):
        data = Path(os.environ.get("SPECTRUM_RESULTS", ROOT / "results/published"))
        env = json.loads((data / "environment.json").read_text())
        replay = json.loads((ROOT / "docs/media/replay.json").read_text())
        video = (ROOT / "docs/media/chirp-replay.mp4").read_bytes()
        self.assertEqual(hashlib.sha256(video).hexdigest(), replay["output_sha256"])
        for relative, expected in replay["input_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected
            )
        renderer = (
            (ROOT / "scripts/render_replay.py")
            .read_bytes()
            .replace(b"\r\n", b"\n")
            .replace(b"\r", b"\n")
        )
        self.assertEqual(
            hashlib.sha256(renderer).hexdigest(), replay["renderer_sha256_lf"]
        )
        self.assertEqual(replay["frames"] / replay["fps"], replay["duration_seconds"])
        self.assertLessEqual(replay["preview_decoded_peak_fs"], 0.3)
        for relative, expected in env["source_sha256_lf"].items():
            normalized = (
                (ROOT / relative)
                .read_bytes()
                .replace(b"\r\n", b"\n")
                .replace(b"\r", b"\n")
            )
            self.assertEqual(hashlib.sha256(normalized).hexdigest(), expected)
        for item in json.loads((data / "manifest.json").read_text()):
            raw = (data / item["path"]).read_bytes()
            self.assertEqual(len(raw), item["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])
        signals = {}
        for fixture in json.loads((data / "fixtures.json").read_text()):
            values, rate = read_pcm16(data / "audio" / f"{fixture['name']}.wav")
            signals[fixture["name"]] = values
            self.assertEqual(rate, env["sample_rate"])
            self.assertEqual(len(values), fixture["samples"])
            self.assertLessEqual(max(abs(x) for x in values), 0.3)
            self.assertEqual(max(abs(x) for x in values), fixture["decoded_peak_fs"])
        for summary in rows(data / "spectral_summary.csv"):
            result = spectrum(
                signals[summary["fixture"]],
                env["sample_rate"],
                summary["window"],
                int(summary["nfft"]),
            )
            stored = rows(data / "spectra" / f"{summary['analysis']}.csv")
            self.assertEqual(len(stored), len(result.frequencies))
            np.testing.assert_allclose(
                [float(r["frequency_hz"]) for r in stored],
                result.frequencies,
                rtol=0,
                atol=0,
            )
            np.testing.assert_allclose(
                [float(r["amplitude_peak_fs"]) for r in stored],
                result.amplitude,
                rtol=1e-10,
                atol=1e-13,
            )
            np.testing.assert_allclose(
                [float(r["psd_fs2_per_hz"]) for r in stored],
                result.psd,
                rtol=1e-10,
                atol=1e-15,
            )
            self.assertAlmostEqual(
                float(summary["psd_integral_fs2"]), result.psd_integral, places=12
            )
            self.assertAlmostEqual(
                result.psd_integral, result.weighted_mean_square, places=12
            )
            self.assertAlmostEqual(
                float(summary["coherent_gain"]), result.coherent_gain, places=13
            )
            self.assertAlmostEqual(float(summary["enbw_hz"]), result.enbw_hz, places=11)
        transform = stft(signals["chirp"], env["sample_rate"], 512, 128)
        stored = rows(data / "chirp_stft.csv")
        self.assertEqual(len(stored), len(transform.times) * len(transform.frequencies))
        np.testing.assert_allclose(
            [float(r["psd_fs2_per_hz"]) for r in stored],
            np.array(transform.psd).ravel(),
            rtol=1e-10,
            atol=1e-15,
        )
        observations = json.loads((data / "observations.json").read_text())
        self.assertTrue(observations["alias_pcm_identical"])
        self.assertEqual(signals["alias_high"], signals["alias_low"])
        self.assertEqual(
            (data / "audio/alias_high.wav").read_bytes(),
            (data / "audio/alias_low.wav").read_bytes(),
        )
        for name, value in observations["off_bin_outside_8hz_power_fraction"].items():
            result = spectrum(signals["off_bin"], env["sample_rate"], name, 16384)
            outside = math.fsum(
                p for f, p in zip(result.frequencies, result.psd) if abs(f - 513.3) > 8
            )
            self.assertAlmostEqual(outside / math.fsum(result.psd), value, places=12)
        timings = rows(data / "measurements.csv")
        self.assertEqual(len(timings), env["repetitions"] * 4 * 2)
        for row in rows(data / "runtime_summary.csv"):
            values = [
                int(t["elapsed_ns"])
                for t in timings
                if t["samples"] == row["samples"] and t["method"] == row["method"]
            ]
            self.assertEqual(len(values), env["repetitions"])
            self.assertTrue(all(v > 0 for v in values))
            self.assertEqual(float(row["median_ns"]), statistics.median(values))
            self.assertEqual(int(row["min_ns"]), min(values))
            self.assertEqual(int(row["max_ns"]), max(values))


if __name__ == "__main__":
    unittest.main()
