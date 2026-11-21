"""Create PCM fixtures, analyze decoded bytes and measure the actual transform."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spectrumlab.audio import read_pcm16, write_pcm16
from spectrumlab.core import fft, ifft, spectrum, stft
from spectrumlab.fixtures import fixtures, SAMPLE_RATE, SEED


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def peak_bins(result):
    candidates = []
    for k, amplitude in enumerate(result.amplitude):
        left = result.amplitude[k - 1] if k else -math.inf
        right = result.amplitude[k + 1] if k + 1 < len(result.amplitude) else -math.inf
        if amplitude > left and amplitude >= right:
            candidates.append(
                {
                    "bin": k,
                    "frequency_hz": result.frequencies[k],
                    "amplitude_peak_fs": amplitude,
                }
            )
    return sorted(candidates, key=lambda row: row["amplitude_peak_fs"], reverse=True)[
        :5
    ]


def run(output, repetitions):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Choose a new empty directory; preserved evidence is not overwritten"
        )
    if not 1 <= repetitions <= 50:
        raise ValueError("Repetitions must lie in 1..50")
    for folder in ["audio", "spectra"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    source_paths = [
        "src/spectrumlab/core.py",
        "src/spectrumlab/audio.py",
        "src/spectrumlab/fixtures.py",
        "scripts/run_study.py",
    ]
    hashes = {
        path: hashlib.sha256(
            (ROOT / path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        ).hexdigest()
        for path in source_paths
    }
    signals = {}
    descriptions = []
    summaries = []
    peaks = {}
    analyses = {}
    for name, (generated, parameters) in fixtures().items():
        path = output / "audio" / f"{name}.wav"
        write_pcm16(path, generated, SAMPLE_RATE)
        # Every analysis and timing input below uses these actual decoded bytes.
        samples, rate = read_pcm16(path)
        signals[name] = samples
        descriptions.append(
            {
                "name": name,
                "sample_rate": rate,
                "samples": len(samples),
                "duration_seconds": len(samples) / rate,
                "decoded_peak_fs": max(abs(x) for x in samples),
                "decoded_rms_fs": math.sqrt(
                    math.fsum(x * x for x in samples) / len(samples)
                ),
                "quantization_max_abs_error": max(
                    abs(a - b) for a, b in zip(generated, samples)
                ),
                "parameters": parameters,
            }
        )
        configurations = [
            (window_name, len(samples)) for window_name in ["rectangular", "hann"]
        ]
        if name == "off_bin":
            configurations.extend(
                [(window_name, 16384) for window_name in ["rectangular", "hann"]]
            )
        if name == "close_short":
            configurations.extend(
                [(window_name, 4096) for window_name in ["rectangular", "hann"]]
            )
        for window_name, nfft in configurations:
            result = spectrum(samples, rate, window_name, nfft)
            key = f"{name}__{window_name}__{nfft}"
            analyses[key] = result
            write_csv(
                output / "spectra" / f"{key}.csv",
                [
                    {
                        "bin": k,
                        "frequency_hz": frequency,
                        "amplitude_peak_fs": result.amplitude[k],
                        "psd_fs2_per_hz": result.psd[k],
                    }
                    for k, frequency in enumerate(result.frequencies)
                ],
            )
            peaks[key] = peak_bins(result)
            summaries.append(
                {
                    "analysis": key,
                    "fixture": name,
                    "window": window_name,
                    "samples": len(samples),
                    "nfft": nfft,
                    "duration_seconds": len(samples) / rate,
                    "bin_spacing_hz": result.bin_spacing_hz,
                    "coherent_gain": result.coherent_gain,
                    "enbw_hz": result.enbw_hz,
                    "weighted_mean_square_fs2": result.weighted_mean_square,
                    "psd_integral_fs2": result.psd_integral,
                    "largest_bin_hz": peaks[key][0]["frequency_hz"],
                    "largest_bin_amplitude_peak_fs": peaks[key][0]["amplitude_peak_fs"],
                }
            )
    write_json(output / "fixtures.json", descriptions)
    write_json(
        output / "peaks.json",
        {
            "policy": "Five highest local bin maxima, no interpolation or component-fitting claim",
            "analyses": peaks,
        },
    )
    write_csv(output / "spectral_summary.csv", summaries)
    chirp = stft(signals["chirp"], SAMPLE_RATE, 512, 128)
    write_csv(
        output / "chirp_stft.csv",
        [
            {
                "frame": frame,
                "time_seconds": t,
                "frequency_hz": frequency,
                "psd_fs2_per_hz": chirp.psd[frame][k],
            }
            for frame, t in enumerate(chirp.times)
            for k, frequency in enumerate(chirp.frequencies)
        ],
    )
    write_json(
        output / "stft.json",
        {
            "fixture": "chirp",
            "window": "periodic Hann",
            "frame_size": chirp.frame_size,
            "hop": chirp.hop,
            "frames": len(chirp.times),
            "bins": len(chirp.frequencies),
            "used_samples": chirp.used_samples,
            "discarded_tail_samples": chirp.discarded_tail_samples,
            "timestamp": "Midpoint between first and last sample of each complete frame; no centering or padding",
        },
    )
    observations = {
        "alias_pcm_identical": signals["alias_high"] == signals["alias_low"],
        "alias_wav_identical": (output / "audio/alias_high.wav").read_bytes()
        == (output / "audio/alias_low.wav").read_bytes(),
        "leakage_exclusion_half_width_hz": 8,
        "leakage_grid_spacing_hz": 0.5,
        "off_bin_outside_8hz_power_fraction": {},
    }
    for name in ["rectangular", "hann"]:
        result = analyses[f"off_bin__{name}__16384"]
        outside = math.fsum(
            power
            for frequency, power in zip(result.frequencies, result.psd)
            if abs(frequency - 513.3) > 8
        )
        observations["off_bin_outside_8hz_power_fraction"][name] = outside / math.fsum(
            result.psd
        )
    write_json(output / "observations.json", observations)

    timings = []
    comparisons = []
    runtime = []
    rng = random.Random(807)
    for n in [64, 256, 1024, 4096]:
        source = signals["chirp"][:n]
        array = np.asarray(source, dtype=np.complex128)
        actual = fft(source)
        reference = np.fft.fft(array)
        inverse = ifft(actual)
        comparisons.append(
            {
                "samples": n,
                "reference": "numpy.fft.fft complex128 default backward normalization",
                "max_abs_complex_error": float(
                    np.max(np.abs(np.asarray(actual) - reference))
                ),
                "inverse_max_abs_error": max(
                    abs(a - b) for a, b in zip(inverse, source)
                ),
                "parseval_abs_error": abs(
                    math.fsum(x * x for x in source)
                    - math.fsum(abs(x) ** 2 for x in actual) / n
                ),
            }
        )
        operations = {
            "own_radix2": lambda: fft(source),
            "numpy_fft": lambda: np.fft.fft(array),
        }
        for operation in operations.values():
            operation()
        for repeat in range(repetitions):
            names = list(operations)
            rng.shuffle(names)
            for order, name in enumerate(names):
                before = time.perf_counter_ns()
                result = operations[name]()
                elapsed = time.perf_counter_ns() - before
                np.testing.assert_allclose(result, reference, rtol=2e-11, atol=2e-11)
                timings.append(
                    {
                        "samples": n,
                        "method": name,
                        "repeat": repeat,
                        "order": order,
                        "elapsed_ns": elapsed,
                    }
                )
        for name in operations:
            values = [
                r["elapsed_ns"]
                for r in timings
                if r["samples"] == n and r["method"] == name
            ]
            runtime.append(
                {
                    "samples": n,
                    "method": name,
                    "median_ns": statistics.median(values),
                    "min_ns": min(values),
                    "max_ns": max(values),
                }
            )
    write_csv(output / "measurements.csv", timings)
    write_csv(output / "runtime_summary.csv", runtime)
    write_csv(output / "transform_comparison.csv", comparisons)
    write_json(
        output / "environment.json",
        {
            "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.system(),
            "architecture": platform.machine(),
            "repetitions": repetitions,
            "timed_calls": len(timings),
            "sample_rate": SAMPLE_RATE,
            "fixture_seed": SEED,
            "order_seed": 807,
            "source_sha256_lf": hashes,
            "source_hash_policy": "CRLF and CR normalized to LF before SHA-256, captured before analysis and timing",
            "precision": "Python complex uses double-precision components; reference array explicitly complex128",
            "timing_scope": "Decoded chirp prefixes; warmed whole FFT calls with prepared tuple/complex128 array; no WAV I/O, plotting, windowing or calibration timed",
            "provenance": "All audio synthesized by this repository, written as mono PCM16, then decoded before analysis; no recordings or external music",
        },
    )
    manifest = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "wav_fixtures": len(signals),
                "spectra": len(summaries),
                "timed_calls": len(timings),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/latest")
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    run(args.out, args.repetitions)
