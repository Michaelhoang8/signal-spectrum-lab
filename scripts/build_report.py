"""Render calibrated plots from preserved decoded-PCM analysis artifacts."""

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spectrumlab.audio import read_pcm16

DATA = ROOT / "results/published"
FIG = ROOT / "docs/figures"
FIG.mkdir(parents=True, exist_ok=True)
BG = "#171225"
PANEL = "#20192f"
TEXT = "#eee7f7"
MUTED = "#b8adc8"
AMBER = "#ffbc59"
CYAN = "#79d6d0"
PINK = "#db83d2"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": "#625472",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#47394f",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 165,
        "legend.facecolor": BG,
        "legend.edgecolor": BG,
    }
)


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def spectral(name, win="hann", nfft=4096):
    rows = read_rows(DATA / "spectra" / f"{name}__{win}__{nfft}.csv")
    return tuple(
        np.array([float(r[key]) for r in rows])
        for key in ["frequency_hz", "amplitude_peak_fs", "psd_fs2_per_hz"]
    )


def save(fig, name):
    fig.tight_layout(pad=1.6)
    for suffix in ["png", "svg"]:
        fig.savefig(FIG / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


stft_meta = json.loads((DATA / "stft.json").read_text())
stft_rows = read_rows(DATA / "chirp_stft.csv")
times = np.array(sorted({float(row["time_seconds"]) for row in stft_rows}))
frequencies = np.array(sorted({float(row["frequency_hz"]) for row in stft_rows}))
density = np.array([float(row["psd_fs2_per_hz"]) for row in stft_rows]).reshape(
    len(times), len(frequencies)
)
fig, axes = plt.subplots(
    2, 1, figsize=(12.5, 6.6), gridspec_kw={"height_ratios": [1, 3]}
)
samples, rate = read_pcm16(DATA / "audio/chirp.wav")
axes[0].plot(np.arange(len(samples)) / rate, samples, color=AMBER, lw=0.4)
axes[0].set_ylabel("Sample value (FS)")
axes[0].set_xlim(0, 1)
axes[0].set_xticks([])
mesh = axes[1].pcolormesh(
    times,
    frequencies,
    10 * np.log10(np.maximum(density.T, 1e-12)),
    shading="nearest",
    cmap="magma",
    vmin=-100,
    vmax=-25,
    rasterized=True,
)
axes[1].set_xlabel("Frame midpoint (seconds)")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_ylim(0, 3500)
fig.colorbar(mesh, ax=axes[1], label="PSD · dB re 1 FS²/Hz", pad=0.02)
save(fig, "chirp_stft")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
for win, color, label in [
    ("rectangular", CYAN, "Rectangular"),
    ("hann", AMBER, "Periodic Hann"),
]:
    f, a, p = spectral("off_bin", win, 16384)
    axes[0].plot(
        f, 20 * np.log10(np.maximum(a, 1e-6)), color=color, label=label, lw=1.1
    )
    axes[1].plot(f, a, color=color, label=label, lw=1.4)
axes[0].set_xlim(440, 585)
axes[0].set_ylim(-105, -8)
axes[0].set_ylabel("Amplitude · dB re 1 FS peak")
axes[1].set_xlim(504, 523)
axes[1].set_ylabel("Coherent-gain-corrected amplitude (FS peak)")
for ax in axes:
    ax.axvline(513.3, color=MUTED, ls=":", lw=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.legend(frameon=False)
axes[0].set_title("Off-bin spectrum", loc="left")
axes[1].set_title("Main-lobe detail", loc="left")
save(fig, "window_comparison")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
f, a, _ = spectral("close_short", "hann", 512)
axes[0].plot(f, a, "o", color=PINK, ms=5, label="512 samples / 512-point FFT")
f, a, _ = spectral("close_short", "hann", 4096)
axes[0].plot(f, a, color=CYAN, lw=1.5, label="Same 512 samples / 4096-point FFT")
f, a, _ = spectral("close_long", "hann", 4096)
axes[1].plot(
    f,
    a,
    "o-",
    color=AMBER,
    ms=3,
    lw=1.4,
    label="4096 observed samples / 4096-point FFT",
)
for ax in axes:
    ax.set_xlim(968, 1040)
    ax.set_ylim(0, 0.18)
    ax.set_xlabel("Frequency (Hz)")
    ax.legend(frameon=False, fontsize=8)
    for frequency in [1000, 1010]:
        ax.axvline(frequency, color=MUTED, lw=0.7, ls=":")
axes[0].set_ylabel("Coherent-gain-corrected amplitude (FS peak)")
axes[0].set_title("62.5 ms observation / zero-padding", loc="left")
axes[1].set_title("500 ms observation / two tone peaks", loc="left")
save(fig, "padding_and_duration")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
f, a, p = spectral("calibration", "rectangular")
indices = [0, 256, 2048]
axes[0].bar(range(3), a[indices], color=[PINK, AMBER, CYAN], width=0.5)
axes[0].scatter(
    range(3),
    [0.05, 0.1, 0.07],
    color=TEXT,
    marker="_",
    s=180,
    label="Generating values",
)
axes[0].set_xticks(range(3), ["DC", "512 Hz", "Nyquist / 4096 Hz"])
axes[0].set_ylim(0, 0.13)
axes[0].set_ylabel("Decoded-bin amplitude (FS peak)")
axes[0].legend(frameon=False)
f, a, p = spectral("noise", "hann")
axes[1].plot(f, 10 * np.log10(np.maximum(p, 1e-12)), color=CYAN, lw=0.6)
axes[1].set_xlim(0, 4096)
axes[1].set_ylim(-110, -40)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("PSD · dB re 1 FS²/Hz")
axes[0].set_title("DC and Nyquist endpoint scaling", loc="left")
axes[1].set_title("One noise periodogram / no averaging", loc="left")
save(fig, "calibration")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
decoded, fs = read_pcm16(DATA / "audio/alias_high.wav")
t = np.linspace(0, 2 / fs * 8, 1600)
axes[0].plot(
    t * 1000,
    0.25 * np.cos(2 * np.pi * 6144 * t),
    color=PINK,
    lw=1,
    label="6144 Hz candidate",
)
axes[0].plot(
    t * 1000,
    0.25 * np.cos(2 * np.pi * 2048 * t),
    color=CYAN,
    lw=1,
    label="2048 Hz candidate",
)
axes[0].scatter(
    np.arange(17) / fs * 1000,
    decoded[:17],
    s=21,
    color=AMBER,
    zorder=5,
    label="Identical decoded samples",
)
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Signal value (FS)")
axes[0].legend(frameon=False, fontsize=8)
f, a, p = spectral("alias_high", "rectangular")
axes[1].plot(f, a, color=AMBER, lw=1.5)
axes[1].set_xlim(0, 4096)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude (FS peak)")
axes[0].set_title("Two candidate cosines / one sampled sequence", loc="left")
axes[1].set_title("Decoded spectrum / 2048 Hz", loc="left")
save(fig, "aliasing")

runtime = read_rows(DATA / "runtime_summary.csv")
fig, ax = plt.subplots(figsize=(11.5, 3.8))
for method, color, label in [
    ("own_radix2", AMBER, "Python radix-2 FFT"),
    ("numpy_fft", CYAN, "NumPy FFT / complex128"),
]:
    values = [row for row in runtime if row["method"] == method]
    sizes = np.array([int(row["samples"]) for row in values])
    med = np.array([float(row["median_ns"]) / 1e6 for row in values])
    low = np.array([float(row["min_ns"]) / 1e6 for row in values])
    high = np.array([float(row["max_ns"]) / 1e6 for row in values])
    ax.errorbar(
        sizes,
        med,
        yerr=[med - low, high - med],
        color=color,
        marker="o",
        lw=1.5,
        capsize=4,
        label=label,
    )
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([64, 256, 1024, 4096], ["64", "256", "1024", "4096"])
ax.set_xlabel("Transform samples")
ax.set_ylabel("Milliseconds per call / log scale")
ax.legend(frameon=False)
save(fig, "runtime")

env = json.loads((DATA / "environment.json").read_text())
obs = json.loads((DATA / "observations.json").read_text())
comparisons = read_rows(DATA / "transform_comparison.csv")
largest_error = max(float(row["max_abs_complex_error"]) for row in comparisons)
rect = obs["off_bin_outside_8hz_power_fraction"]["rectangular"]
hann = obs["off_bin_outside_8hz_power_fraction"]["hann"]
fixture_data = json.loads((DATA / "fixtures.json").read_text())
audio_links = "".join(
    f'<a href="../results/published/audio/{item["name"]}.wav">{item["name"]}.wav</a>'
    for item in fixture_data
)
report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Signal spectrum lab | FFT and calibration study</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:{BG};color:{TEXT};font:17px/1.6 Arial,sans-serif}}header{{padding:20px 5vw;border-bottom:1px solid #40344f;display:flex;justify-content:space-between;font:12px Consolas,monospace;color:{MUTED}}}main{{max-width:1370px;margin:auto;padding:0 48px}}section{{padding:46px 0;border-bottom:1px solid #40344f}}h1{{font-size:66px;line-height:1.05;letter-spacing:-.04em;margin:18px 0}}h2{{font-size:35px;letter-spacing:-.025em;line-height:1.15;margin:12px 0 20px}}h3{{font-size:17px}}.label{{font:12px Consolas,monospace;letter-spacing:.12em;color:{AMBER}}}p{{max-width:1030px}}.lead{{font-size:21px}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:50px}}.stats{{display:flex;gap:60px;padding:18px 0}}.stats strong{{font:35px Consolas,monospace;display:block;color:{AMBER}}}.stats span{{font-size:13px;color:{MUTED}}}img,video{{display:block;width:100%;height:auto;margin:20px 0}}.note{{background:{PANEL};border-left:3px solid {AMBER};padding:14px 20px;font-size:15px}}audio{{width:100%;max-width:500px;display:block;margin:16px 0;color-scheme:dark}}a{{color:{CYAN}}}.audio-links{{display:flex;flex-wrap:wrap;gap:10px 24px;font-size:14px}}pre{{background:#0e0a16;padding:22px;color:#e5dcf2;font:14px/1.7 Consolas,monospace;overflow:auto}}figcaption,footer{{font-size:13px;color:{MUTED}}}footer{{padding:28px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px;border-bottom:1px solid #40344f;text-align:left}}th{{color:{MUTED};font:12px Consolas,monospace}}@media(max-width:800px){{main{{padding:0 20px}}h1{{font-size:43px}}.split{{display:block}}.stats{{gap:24px}}}}
</style><header><span>DIGITAL SIGNAL PROCESSING / SYNTHETIC PCM AUDIO</span><span>{env['started_utc'][:10]} · 8192 SAMPLES / SECOND</span></header><main>
<section id="overview"><div class="label">RADIX-2 FFT / AMPLITUDE / POWER SPECTRAL DENSITY</div><h1>Signal spectrum lab</h1><p class="lead">A complex FFT implemented from source, calibrated one-sided spectra and an STFT of actual decoded PCM16 samples.</p><div class="stats"><div><strong>9</strong><span>synthetic WAV fixtures</span></div><div><strong>22</strong><span>preserved spectra</span></div><div><strong>56</strong><span>timed transform calls</span></div></div><video controls preload="none" width="1200" height="720" poster="figures/chirp_stft.png"><source src="media/chirp-replay.mp4" type="video/mp4"></video><figcaption>512-sample periodic Hann frames, 128-sample hop, 61 complete frames. Frame times use sample midpoints. Display floor: −100 dB re 1 FS²/Hz.</figcaption><p>Replay: four one-second chirps at original speed, with a silent second before and after. The cursor follows stored analysis; this is not live processing. <a href="../results/published/audio/chirp.wav">Canonical PCM16 WAV</a> · <a href="media/replay.json">Replay metadata</a>. The MP4 uses AAC at 48 kHz; analysis uses the original 8192 Hz PCM.</p><p class="note">All fixtures are generated here, encoded as mono PCM16, then decoded before analysis. Decoded sample peaks are at most 0.3 full scale. Playback is optional; nothing autoplays. FS denotes the digital sample unit, not acoustic sound pressure.</p></section>
<section id="windows"><div class="label">01 / WINDOWS</div><h2>Leakage and main-lobe width</h2><p>The 513.3 Hz tone is off the native 2 Hz bin grid. A periodic Hann window suppresses distant leakage while broadening the main lobe. Both curves use the same 4096 decoded samples and a 16384-point FFT for a denser plotted frequency grid.</p><img src="figures/window_comparison.png" alt="Rectangular and periodic Hann spectra of an off-bin cosine, with main-lobe detail"><div class="split"><p>Outside ±8 Hz of the generating frequency, the rectangular window places {100*rect:.3f}% of integrated periodogram power; Hann places {100*hann:.5f}%. This is a defined measurement on this fixture, not a universal window score.</p><p>Amplitude uses the window sum for coherent-gain correction. It recovers isolated coherent tones under the stated conditions; an off-bin bin maximum is not automatically the generating amplitude.</p></div></section>
<section id="padding"><div class="label">02 / OBSERVATION LENGTH</div><h2>Zero-padding and close tones</h2><p>Two cosines at 1000 and 1010 Hz use equal peak amplitudes of 0.12 FS. Padding the 512-sample observation changes bin spacing from 16 Hz to 2 Hz, while leaving its 62.5 ms observation unchanged.</p><img src="figures/padding_and_duration.png" alt="Short observation with and without zero-padding compared with a longer observation that separates two tones"><p class="note">Zero-padding samples the same finite-record transform more densely. It adds no observed samples and does not narrow the window's main lobe. The 4096-sample observation supplies 500 ms of actual data and separates the two peaks in this case.</p><audio controls preload="none" src="../results/published/audio/close_long.wav"></audio></section>
<section id="calibration"><div class="label">03 / UNITS AND ENDPOINTS</div><h2>Amplitude and PSD calibration</h2><img src="figures/calibration.png" alt="Calibrated DC, sinusoid and Nyquist amplitudes plus a one-sided noise PSD"><table><tr><th>QUANTITY</th><th>NORMALIZATION</th><th>UNIT</th></tr><tr><td>Interior-bin amplitude</td><td>2 |X[k]| / Σw</td><td>FS peak</td></tr><tr><td>DC / Nyquist amplitude</td><td>|X[k]| / Σw</td><td>FS peak</td></tr><tr><td>Interior-bin PSD</td><td>2 |X[k]|² / (fₛ Σw²)</td><td>FS²/Hz</td></tr><tr><td>DC / Nyquist PSD</td><td>|X[k]|² / (fₛ Σw²)</td><td>FS²/Hz</td></tr></table><p>Summing PSD × bin spacing gives Σ(xw)² / Σw². With a rectangular window this is mean-square signal value; for Hann it is the window-weighted mean square. The noise plot is one periodogram, with no averaging or uncertainty interval.</p></section>
<section id="aliasing"><div class="label">04 / SAMPLING</div><h2>Aliasing at 8192 samples per second</h2><p>A 6144 Hz cosine and a 2048 Hz cosine produce the same sampled sequence at this rate. Their committed PCM16 WAV files are byte-for-byte identical. The spectrum cannot recover which continuous cosine generated those samples.</p><img src="figures/aliasing.png" alt="Two continuous cosine candidates crossing identical decoded sample values, with a peak at 2048 Hz"><audio controls preload="none" src="../results/published/audio/alias_high.wav"></audio><div class="audio-links">{audio_links}</div></section>
<section id="validation"><div class="label">05 / NUMERICAL CHECKS</div><h2>Transform accuracy and measured runtime</h2><p>Own-FFT outputs are checked against an independent direct DFT on small inputs and NumPy on larger inputs. In the four recorded decoded-chirp comparisons, the largest absolute complex difference from NumPy is {largest_error:.3g}. That is an observed error, not a universal error bound.</p><img src="figures/runtime.png" alt="Own Python radix-2 FFT and NumPy FFT measured runtime with observed ranges"><figcaption>Seven warmed repetitions, median and observed minimum/maximum. Python source code and native NumPy differ in implementation and input preparation. No real-time claim is made.</figcaption><div class="split"><div><h3>Validation</h3><p>12 tests cover direct DFT agreement, linearity, circular shift, inverse recovery, Parseval, odd-length Hann calibration, DC/Nyquist scaling, PSD integration, STFT framing, PCM boundaries, alias identity and artifact reconciliation.</p><p><a href="formulae.md">Formulae and calibration</a> · <a href="methodology.md">Methodology</a> · <a href="findings.md">Findings</a></p></div><div><h3>Evidence</h3><p><a href="../results/published/spectral_summary.csv">Spectral summaries</a> · <a href="../results/published/transform_comparison.csv">Reference comparisons</a> · <a href="../results/published/measurements.csv">Raw timings</a> · <a href="../results/published/manifest.json">Artifact hashes</a> · <a href="../results/published/environment.json">Run environment</a></p></div></div><pre>python -m pip install -e . -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_study.py --out outputs/my-run</pre></section><footer>AI-assisted engineering study · Synthetic audio only · No external music, human recordings or physical audio calibration.</footer></main></html>"""
(ROOT / "docs/report.html").write_text(report, encoding="utf-8", newline="\n")
(FIG / "environment.json").write_text(
    json.dumps(
        {
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "source": "results/published",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print(ROOT / "docs/report.html")
