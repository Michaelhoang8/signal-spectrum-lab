"""Radix-2 complex FFT and one-sided real-signal calibration, using only stdlib."""

import cmath
from dataclasses import dataclass
import math


def _power_of_two(length):
    return type(length) is int and length > 0 and length & (length - 1) == 0


def fft(values):
    """Unnormalized forward DFT, exp(-2 pi i kn/N), for a positive power of two."""
    data = [complex(value) for value in values]
    n = len(data)
    if not _power_of_two(n):
        raise ValueError("FFT length must be a positive power of two")
    if any(
        not math.isfinite(value.real) or not math.isfinite(value.imag) for value in data
    ):
        raise ValueError("FFT input must be finite")
    # Bit-reversal permutation followed by iterative radix-2 butterflies.
    reversed_index = 0
    for index in range(1, n):
        bit = n >> 1
        while reversed_index & bit:
            reversed_index ^= bit
            bit >>= 1
        reversed_index ^= bit
        if index < reversed_index:
            data[index], data[reversed_index] = data[reversed_index], data[index]
    width = 2
    while width <= n:
        half = width // 2
        # Compute each twiddle directly to avoid long repeated-multiply drift.
        twiddles = [cmath.exp(-2j * math.pi * k / width) for k in range(half)]
        for start in range(0, n, width):
            for offset, twiddle in enumerate(twiddles):
                even = data[start + offset]
                odd = twiddle * data[start + offset + half]
                data[start + offset] = even + odd
                data[start + offset + half] = even - odd
        width *= 2
    return tuple(data)


def ifft(values):
    """Inverse DFT via conjugation, with the 1/N factor applied exactly once."""
    values = tuple(complex(value) for value in values)
    transformed = fft(value.conjugate() for value in values)
    return tuple(value.conjugate() / len(values) for value in transformed)


def window(name, length):
    """Rectangular or periodic Hann; the length-one case is explicitly [1]."""
    if type(length) is not int or length <= 0:
        raise ValueError("Window length must be a positive integer")
    if name not in {"rectangular", "hann"}:
        raise ValueError("Window must be rectangular or hann")
    if name == "rectangular" or length == 1:
        return (1.0,) * length
    return tuple(
        0.5 - 0.5 * math.cos(2 * math.pi * index / length) for index in range(length)
    )


def _samples(values):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Real samples are required") from exc
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("Samples must be nonempty and finite")
    return result


def _sample_rate(sample_rate):
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not math.isfinite(sample_rate)
        or sample_rate <= 0
    ):
        raise ValueError("Sample rate must be finite and positive")


@dataclass(frozen=True)
class Spectrum:
    frequencies: tuple[float, ...]
    amplitude: tuple[float, ...]
    psd: tuple[float, ...]
    sample_rate: float
    sample_count: int
    nfft: int
    window_name: str
    coherent_gain: float
    enbw_hz: float
    weighted_mean_square: float

    @property
    def bin_spacing_hz(self):
        return self.sample_rate / self.nfft

    @property
    def psd_integral(self):
        return math.fsum(self.psd) * self.bin_spacing_hz


def spectrum(values, sample_rate, window_name="hann", nfft=None):
    """One-sided peak-amplitude and density spectra of real input.

    Input values are in signal units. PSD is signal-unit squared per Hz. The
    coherent-gain amplitude correction applies to isolated coherent tones;
    off-bin or overlapping peaks need not recover the generating amplitudes.
    """
    data = _samples(values)
    _sample_rate(sample_rate)
    n = len(data)
    if nfft is None:
        nfft = 1 << (n - 1).bit_length()
    if not _power_of_two(nfft) or nfft < n:
        raise ValueError(
            "nfft must be a power of two at least as large as the observation"
        )
    weights = window(window_name, n)
    sum_w = math.fsum(weights)
    sum_w2 = math.fsum(value * value for value in weights)
    weighted = [value * weight for value, weight in zip(data, weights)]
    transformed = fft(weighted + [0.0] * (nfft - n))
    amplitudes = []
    densities = []
    for k, value in enumerate(transformed[: nfft // 2 + 1]):
        endpoint = k == 0 or (nfft > 1 and k == nfft // 2)
        factor = 1 if endpoint else 2
        amplitudes.append(factor * abs(value) / sum_w)
        densities.append(factor * abs(value) ** 2 / (sample_rate * sum_w2))
    return Spectrum(
        tuple(k * sample_rate / nfft for k in range(len(amplitudes))),
        tuple(amplitudes),
        tuple(densities),
        sample_rate,
        n,
        nfft,
        window_name,
        sum_w / n,
        sample_rate * sum_w2 / sum_w**2,
        math.fsum(value * value for value in weighted) / sum_w2,
    )


@dataclass(frozen=True)
class STFT:
    times: tuple[float, ...]
    frequencies: tuple[float, ...]
    psd: tuple[tuple[float, ...], ...]
    frame_size: int
    hop: int
    sample_rate: float
    used_samples: int
    discarded_tail_samples: int


def stft(values, sample_rate, frame_size=512, hop=128, window_name="hann"):
    """PSD per complete frame. No centering, reflection, or tail padding."""
    data = _samples(values)
    _sample_rate(sample_rate)
    if not _power_of_two(frame_size) or frame_size > len(data):
        raise ValueError("Frame size must be a power of two within the observation")
    if type(hop) is not int or not 1 <= hop <= frame_size:
        raise ValueError("Hop must be an integer in 1..frame_size")
    starts = tuple(range(0, len(data) - frame_size + 1, hop))
    frames = [
        spectrum(data[start : start + frame_size], sample_rate, window_name, frame_size)
        for start in starts
    ]
    # Timestamp is the midpoint between the first and last sample in the frame.
    times = tuple((start + (frame_size - 1) / 2) / sample_rate for start in starts)
    used = starts[-1] + frame_size
    return STFT(
        times,
        frames[0].frequencies,
        tuple(frame.psd for frame in frames),
        frame_size,
        hop,
        sample_rate,
        used,
        len(data) - used,
    )
