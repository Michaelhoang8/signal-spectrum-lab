# Findings

## Windowing changes leakage and width

For the decoded 513.3 Hz fixture, the fraction of periodogram power outside ±8 Hz is approximately 2.5254% with the rectangular window and 0.0013795% with periodic Hann. Both use the same 4096 samples and 16384-point transform. The Hann main lobe is wider in the plotted detail. This observation illustrates a tradeoff, not a universal ranking of windows.

The amplitudes use coherent-gain normalization. That calibration does not eliminate the effect of an off-bin frequency or interference between nearby components. Bin maxima are therefore reported as observed grid values, not fitted generating amplitudes.

## Padding differs from longer observation

The short close-tone fixture has 512 samples, or 62.5 ms. Padding to 4096 transform points changes the frequency spacing from 16 Hz to 2 Hz, but leaves the broad, overlapping spectral shape. Its native-grid samples lie on the padded curve.

The long fixture observes 4096 actual samples, or 500 ms. Its periodic-Hann spectrum separates the 1000 Hz and 1010 Hz tone peaks. The improved separation in this selected case comes from a longer observation, not from appending zeros.

## Units and endpoint handling reconcile

The calibration fixture contains DC 0.05 FS, a 512 Hz cosine with peak 0.10 FS, and a Nyquist alternating sequence with peak 0.07 FS. Decoded amplitudes track these values subject to PCM quantization. DC and Nyquist are not doubled; interior one-sided bins are.

PSD bin sums reconcile with the specified window-weighted mean square. The rectangular case equals ordinary mean square. The single Hann noise periodogram has visible fluctuations and is not an averaged or confidence-bounded estimate of a noise process.

## Aliasing removes information

The 6144 Hz and 2048 Hz cosine files are byte-for-byte identical after sampling at 8192 samples/s and PCM16 encoding. Both spectra peak at 2048 Hz. No post hoc transform can distinguish which of those generating candidates produced the shared sample sequence.

## Numerical and runtime observations

Across the four preserved decoded-chirp reference cases, maximum absolute complex disagreement with NumPy ranges from about 2.66 × 10⁻¹⁵ to 2.39 × 10⁻¹⁴. Maximum inverse reconstruction error is about 2.41 × 10⁻¹⁶. These are observed errors for these inputs, not universal bounds.

At length 4096 the recorded median own-FFT call is about 6.11 ms versus 0.0745 ms for NumPy. This compares readable Python radix-2 code with an optimized native transform, including different input-preparation costs as documented. The purpose is inspectable correctness and calibration; the measurements do not support a real-time or competitive-performance claim.
