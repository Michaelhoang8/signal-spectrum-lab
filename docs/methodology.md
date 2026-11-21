# Methodology

## Synthetic PCM inputs

`src/spectrumlab/fixtures.py` generates nine owned fixtures at 8192 samples/s:

| Fixture | Samples | Generating signal |
| --- | ---: | --- |
| `coherent` | 4096 | 512 Hz cosine, peak 0.25 |
| `off_bin` | 4096 | 513.3 Hz cosine, peak 0.25 |
| `close_short` | 512 | 1000/1010 Hz cosines, peak 0.12 each |
| `close_long` | 4096 | Same two cosines, longer observation |
| `chirp` | 8192 | Linear cosine chirp from 256 toward 2800 Hz over one second, peak 0.25 |
| `noise` | 4096 | Seeded Gaussian samples, standard deviation 0.06, explicitly clipped at ±0.29 |
| `alias_high` | 4096 | 6144 Hz cosine, peak 0.25 |
| `alias_low` | 4096 | 2048 Hz cosine, peak 0.25 |
| `calibration` | 4096 | DC 0.05 + 512 Hz cosine peak 0.10 + Nyquist alternating component peak 0.07 |

The noise seed is `20261121`. Inputs are selected to demonstrate mechanisms; they are not a representative audio benchmark. No external music, speech or sensor input is used. The chirp's instantaneous generating frequency reaches 2800 Hz at time 1 second, just after the last stored sample.

Each signal is rounded to signed 16-bit integers using scale 32768 and Python's nearest-even rounding, with a numerical bound that also keeps decoded peaks at or below 0.3 FS. The fixture values lie comfortably inside the PCM range. There is no dither, resampling or amplitude normalization after generation.

The study writes each WAV, reads it back through the mono PCM16 decoder, and uses those decoded samples for every spectrum, STFT, numerical comparison and timing input. It records peak, RMS and maximum quantization error versus the generating array. The saved files therefore define the analysis input; the unquantized generator is not silently substituted.

## Spectral configurations

All nine fixtures receive native-length rectangular and periodic-Hann analyses. The off-bin fixture additionally uses a 16384-point transform with both windows, and the short close-tone fixture uses a 4096-point padded transform with both windows. This yields 22 complete calibrated spectra.

`spectra/*.csv` stores every retained bin, frequency, peak-amplitude value and PSD value. `peaks.json` stores up to five highest local bin maxima, including endpoints when appropriate. These are grid maxima without interpolation, component fitting or a noise-detection threshold. They must not be interpreted as five known physical sources.

The chirp uses 512-sample Hann frames with hop 128, producing 61 complete frames and 257 bins per frame. There is no discarded tail in this particular fixture. `stft.json` records the policy; tests also exercise nonzero tails.

For the window comparison, the leakage statistic is the fraction of summed one-sided periodogram power at frequencies more than 8 Hz from 513.3 Hz. Both windows use the same ±8 Hz exclusion and the same padded 0.5 Hz grid. The exclusion is deliberately wider than the main lobes in this fixture. It is a defined comparison, not an intrinsic window-quality number.

## Numerical references

Tests compare the own FFT to an independent direct trigonometric DFT for small complex arrays and to NumPy for larger arrays. Additional tests use inverse recovery, Parseval and analytic amplitude/power identities, avoiding reliance on only one implementation oracle.

The preserved `transform_comparison.csv` records comparisons on the first 64, 256, 1024 and 4096 decoded chirp samples. The NumPy reference input is explicitly `complex128`; the own transform uses Python complex values with double-precision components. The maximum complex absolute error, inverse error and Parseval discrepancy are recorded. Zero discrepancy in a rounded diagnostic is not proof of exact floating-point arithmetic.

## Runtime protocol

The four decoded chirp prefixes are prepared outside the timed region, as Python tuples for the own FFT and `complex128` arrays for NumPy. Each transform is warmed once. Seven repetitions per method and length follow, with method order shuffled using seed `807`.

`time.perf_counter_ns()` measures complete transform calls, including the own routine's conversion to complex values, bit reversal, twiddle construction and output allocation. NumPy's input-array preparation is outside timing. WAV I/O, generation, windowing, amplitude/PSD calibration, reference assertions and plotting are excluded. Every output is numerically checked outside the timed region.

There are `4 lengths × 2 implementations × 7 repeats = 56` raw observations. Summaries use the median, minimum and maximum without dropping observations. Ranges are not confidence intervals. There is no CPU affinity, isolated process, background-load control or real-time scheduling. Language, library and implementation differences prevent an algorithm-only speed interpretation. No real-time audio processing is demonstrated.

## Provenance and reproduction

`environment.json` records UTC timestamps, Python and NumPy versions, OS family, architecture, seeds and LF-normalized hashes of the core, audio module, fixture generator and runner as they existed before analysis. `manifest.json` records exact-byte SHA-256 hashes and sizes of every other study artifact. All generated text uses UTF-8/LF; WAV files are binary.

The runner refuses to overwrite a nonempty destination. New experiments go under ignored `outputs/` by default; the published run remains under `results/published/`. CI tests the preserved run and a separate two-repetition experiment. Artifact validation checks PCM data, hashes, full spectra, STFT values, power identities, alias equality and timing reconciliation. Plot dependencies are NumPy `2.3.5` and Matplotlib `3.11.2`; replay rendering also uses Pillow `13.0.0`. They are pinned in `requirements.txt`; the core transform depends on none of them.

## Synchronized audio replay

`scripts/render_replay.py` concatenates one second of PCM silence, four exact copies of the committed one-second chirp PCM, and a final silent second. It renders the saved STFT at 1200 × 720 and 30 frames/s, moving a cursor according to each video frame's midpoint in playback time. The display is a replay, not live analysis. Repeats and elapsed time are labeled.

The MP4 uses H.264 video and an AAC preview resampled from 8192 to 48000 Hz. Its container audio and video durations are both six seconds. AAC decoding emits 288,768 samples rather than the nominal 288,000 because of codec tail padding after the silent outro. This encoded preview is not used for any spectral measurement; the raw WAV is canonical. The decoded preview's measured absolute peak is approximately 0.283 FS, below the declared 0.3 numerical cap.

`docs/media/replay.json` records the actual ffmpeg and Pillow versions, frame counts, timing policy, source WAV/STFT hashes, normalized renderer source hash and output MP4 hash. Tests check those committed bytes. Reproduction requires an ffmpeg installation with libx264/AAC support; encoding bytes may vary by version. Browser playback is user-initiated, without autoplay.

## Limits

This study does not model analog anti-alias filters, ADC electronics, microphone calibration, acoustic propagation, real recordings, perceptual loudness, streaming latency, inverse-STFT reconstruction or production deployment. Digital peak limits do not specify playback sound pressure. All audio controls require deliberate playback and do not autoplay. This work was developed with AI assistance.
