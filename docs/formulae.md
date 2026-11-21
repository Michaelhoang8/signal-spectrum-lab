# Formulae and calibration

## Transform convention

For `N` complex samples, the implemented transform is:

```text
X[k] = sum(x[n] * exp(-2*pi*i*k*n/N), n=0..N-1)
x[n] = sum(X[k] * exp(+2*pi*i*k*n/N), k=0..N-1) / N
```

The forward transform is unnormalized. The inverse applies `1/N` once. This matches the default “backward” normalization in the [NumPy FFT documentation](https://numpy.org/doc/2.3/reference/routines.fft.html). The implementation performs its own bit-reversal permutation and iterative radix-2 butterflies. It calculates each stage's twiddle factors directly with `cmath.exp`, rather than using NumPy's transform.

The core FFT accepts positive power-of-two lengths, including one, and finite complex values. Arithmetic uses Python complex numbers with double-precision components. Finite-input validation is not an overflow guarantee for arbitrarily large magnitudes. The numerical study is restricted to bounded decoded PCM signals and the documented bounded test vectors.

For the forward convention, Parseval's identity is:

```text
sum(abs(x[n])**2) = sum(abs(X[k])**2) / N
```

## Windows

For `N` observed real samples, the rectangular window is `w[n] = 1`. The periodic Hann window is:

```text
w[n] = 0.5 - 0.5*cos(2*pi*n/N), n=0..N-1
```

The denominator is `N`, not `N-1`. This is the periodic spectral-analysis form; a symmetric filter-design window is different. The distinction is documented by [SciPy's Hann window API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html). For length one this project explicitly defines either window as `[1]`.

Define:

```text
S1 = sum(w[n])
S2 = sum(w[n]**2)
coherent_gain = S1 / N
equivalent_noise_bandwidth_hz = sample_rate * S2 / S1**2
```

For the nondegenerate periodic Hann lengths used in the study, coherent gain is 1/2 and ENBW is 1.5 times `sample_rate/N`. Rectangular ENBW is `sample_rate/N`. These depend on the observation/window length, not the zero-padded transform length.

## One-sided spectra

Window the observed samples first, then append zeros to length `M = nfft`:

```text
y = [x[n]*w[n] for n in 0..N-1] + [0] * (M-N)
X = FFT(y)
frequency[k] = k * sample_rate / M
```

Only bins `0..M/2` are retained for even `M`. Their one-sided factor `d[k]` is 1 at DC and Nyquist and 2 at interior bins. A one-point transform contains DC only, with factor 1. For real input, interior positive and negative frequency bins form conjugate pairs; DC and Nyquist do not have distinct partners.

```text
amplitude[k] = d[k] * abs(X[k]) / S1
PSD[k] = d[k] * abs(X[k])**2 / (sample_rate * S2)
```

The two denominators serve different purposes. Amplitude normalization corrects coherent gain. PSD normalization accounts for window energy and sample rate. The [SciPy spectral-analysis tutorial](https://docs.scipy.org/doc/scipy/tutorial/signal.html#spectral-analysis) discusses the distinction between magnitude and density scaling.

Amplitude is a peak-amplitude convention for an isolated bin-centered real sinusoid, away from endpoint interaction and overlapping components. Off-bin sampling, neighboring components and spectral leakage can alter the largest bin. DC amplitude is the absolute DC level; Nyquist amplitude is the peak value of the alternating sample sequence. The code does not fit sinusoid parameters or claim that every local maximum identifies a physical component.

## PSD integration

With `df = sample_rate/M`, summing all retained bins with the endpoint factors gives:

```text
sum(PSD[k]) * df = sum((x[n]*w[n])**2) / S2
```

This is a bin sum, not a trapezoidal integral that would halve the endpoint bins again. For a rectangular window it equals the signal mean square. For Hann it equals the window-weighted mean square. A single deterministic periodogram is not automatically an unbiased or low-variance estimate of an unknown stochastic source's spectrum. The noise figure is one realization without Welch averaging.

## Digital units and decibels

PCM16 values are decoded as `integer_sample / 32768`. One full-scale unit (FS) is that normalized digital sample unit: the representable interval is `[-1, 32767/32768]`. This project does not infer volts, pascals or dB SPL.

- Linear amplitude: FS peak under the convention above.
- Linear PSD: FS²/Hz.
- Plotted amplitude decibels: `20*log10(amplitude / 1 FS peak)`.
- Plotted PSD decibels: `10*log10(PSD / (1 FS²/Hz))`.

The amplitude convention uses a peak-one reference, not an RMS-full-scale-sine convention. Plot floors are display choices; zero spectral values remain zero in the CSVs. Spectrogram color clipping at −100 dB is explicitly labeled and does not alter saved numerical results.

## Zero-padding and observation length

Increasing `M` decreases the sampled frequency spacing `sample_rate/M`. It evaluates the same finite-record transform on a denser grid. It does not increase observation duration `N/sample_rate`, narrow the window's main lobe or create missing information. A more finely sampled maximum can aid peak inspection, but it is not additional ability to separate arbitrary nearby components.

For a window of duration `T = N/sample_rate`, ideal isolated-tone rectangular and periodic-Hann window responses have approximate null-to-null main-lobe widths `2/T` and `4/T`, respectively. The plotted finite, quantized, real-tone spectra also contain the negative-frequency component and quantization terms, so a visual minimum need not be an exact zero.

## STFT framing

The STFT calls the same one-sided PSD routine independently for each complete frame. A 512-sample periodic Hann window with hop 128 is used for the chirp. Frames start at `0, hop, 2*hop, ...` while the entire frame fits. There is no reflection, centering, leading padding or tail padding.

Timestamps are `(start + (frame_size-1)/2) / sample_rate`, the midpoint between the first and last samples. Any incomplete tail is reported as discarded. Each column is a calibrated frame periodogram; this project does not implement inverse-STFT reconstruction, overlap-add synthesis or a claim that summing overlapping frames gives whole-record energy without further weighting.

## Sampling identity

For integer `n`, a cosine at `sample_rate - f` has the same samples as a cosine at `f`:

```text
cos(2*pi*(sample_rate-f)*n/sample_rate) = cos(2*pi*f*n/sample_rate)
```

Thus the 6144 Hz and 2048 Hz fixtures are identical when sampled at 8192 samples/s and serialized here. The FFT cannot distinguish candidate continuous signals with identical samples. No anti-alias filter is modeled.
