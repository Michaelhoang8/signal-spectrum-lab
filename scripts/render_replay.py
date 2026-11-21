"""Replay the actual chirp WAV over its saved STFT with an original-speed cursor."""

import array
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/published"
OUT = ROOT / "docs/media"
TEMP = ROOT / "outputs/replay-build"
OUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT, FPS = 1200, 720, 30
INTRO, REPEATS, OUTRO = 1, 4, 1
BG = "#171225"
TEXT = "#eee7f7"
MUTED = "#b8adc8"
AMBER = "#ffbc59"
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    raise RuntimeError("ffmpeg with libx264 and AAC support must be available on PATH")

with wave.open(str(DATA / "audio/chirp.wav"), "rb") as stream:
    if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
        raise ValueError("Replay requires mono PCM16")
    rate = stream.getframerate()
    count = stream.getnframes()
    raw = stream.readframes(count)
if count != rate:
    raise ValueError("The replay fixture must be exactly one second")
with wave.open(str(TEMP / "playback.wav"), "wb") as stream:
    stream.setnchannels(1)
    stream.setsampwidth(2)
    stream.setframerate(rate)
    stream.writeframes(
        bytes(rate * 2 * INTRO) + raw * REPEATS + bytes(rate * 2 * OUTRO)
    )

with (DATA / "chirp_stft.csv").open() as stream:
    rows = list(csv.DictReader(stream))
times = np.array(sorted({float(row["time_seconds"]) for row in rows}))
frequencies = np.array(sorted({float(row["frequency_hz"]) for row in rows}))
power = np.array([float(row["psd_fs2_per_hz"]) for row in rows]).reshape(
    len(times), len(frequencies)
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": "#625472",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)
fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor=BG)
ax = fig.add_axes([0.08, 0.20, 0.78, 0.57], facecolor=BG)
mesh = ax.pcolormesh(
    times,
    frequencies,
    10 * np.log10(np.maximum(power.T, 1e-12)),
    shading="nearest",
    cmap="magma",
    vmin=-100,
    vmax=-25,
    rasterized=True,
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 3500)
ax.set_xlabel("Time within one chirp (seconds)")
ax.set_ylabel("Frequency (Hz)")
bar = fig.add_axes([0.89, 0.20, 0.018, 0.57])
fig.colorbar(mesh, cax=bar, label="PSD · dB re 1 FS²/Hz")
fig.text(0.08, 0.92, "Chirp spectrogram replay", fontsize=30, color=TEXT)
fig.text(
    0.08,
    0.865,
    "Actual decoded PCM16 fixture / stored own-FFT STFT / original playback speed",
    fontsize=12,
    color=MUTED,
)
fig.text(
    0.08,
    0.055,
    "Cursor follows playback time. This is a replay of saved analysis, not live processing.",
    fontsize=12,
    color=MUTED,
)
fig.canvas.draw()
base = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(), "RGB")
plt.close(fig)
font_path = matplotlib.font_manager.findfont("DejaVu Sans")
font = ImageFont.truetype(font_path, 19)
font_small = ImageFont.truetype(font_path, 15)
command = [
    ffmpeg,
    "-y",
    "-loglevel",
    "error",
    "-f",
    "rawvideo",
    "-pixel_format",
    "rgb24",
    "-video_size",
    f"{WIDTH}x{HEIGHT}",
    "-framerate",
    str(FPS),
    "-i",
    "-",
    "-i",
    str(TEMP / "playback.wav"),
    "-map",
    "0:v:0",
    "-map",
    "1:a:0",
    "-c:v",
    "libx264",
    "-crf",
    "22",
    "-preset",
    "medium",
    "-pix_fmt",
    "yuv420p",
    "-c:a",
    "aac",
    "-b:a",
    "128k",
    "-ar",
    "48000",
    "-shortest",
    "-movflags",
    "+faststart",
    str(OUT / "chirp-replay.mp4"),
]
process = subprocess.Popen(command, stdin=subprocess.PIPE)
try:
    total_frames = (INTRO + REPEATS + OUTRO) * FPS
    for frame in range(total_frames):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        elapsed = (frame + 0.5) / FPS
        if elapsed < INTRO:
            status = "Silent intro / four one-second chirps follow"
        elif elapsed < INTRO + REPEATS:
            relative = elapsed - INTRO
            repeat = int(relative) + 1
            within = relative % 1
            x = round((0.08 + 0.78 * within) * WIDTH)
            y0 = round((1 - 0.77) * HEIGHT)
            y1 = round((1 - 0.20) * HEIGHT)
            draw.line((x, y0, x, y1), fill=AMBER, width=2)
            draw.ellipse((x - 4, y0 - 4, x + 4, y0 + 4), fill=AMBER)
            status = f"Repeat {repeat} / {REPEATS}   |   chirp time {within:.3f} s"
        else:
            status = "Silent end / four repeats completed"
        draw.text((96, 642), status, font=font, fill=AMBER)
        draw.text((900, 642), f"{elapsed:.2f} / 6.00 s", font=font_small, fill=MUTED)
        process.stdin.write(image.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the replay")
except BaseException:
    process.kill()
    process.wait()
    raise

# Measure the actual encoded preview's decoded peak, since resampling/AAC may
# change sample peaks. The raw committed WAV remains the analysis authority.
decoded = array.array("f")
decoded.frombytes(
    subprocess.check_output(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(OUT / "chirp-replay.mp4"),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-",
        ]
    )
)
preview_peak = max(abs(value) for value in decoded)
if preview_peak > 0.3:
    raise RuntimeError("Encoded preview exceeded the declared numerical peak cap")
sources = [DATA / "audio/chirp.wav", DATA / "chirp_stft.csv", DATA / "stft.json"]
metadata = {
    "frames": total_frames,
    "fps": FPS,
    "duration_seconds": 6,
    "width": WIDTH,
    "height": HEIGHT,
    "repeats": REPEATS,
    "intro_silence_seconds": INTRO,
    "outro_silence_seconds": OUTRO,
    "audio_policy": "Four exact original-speed PCM repeats, then preview encoded as AAC at 48000 Hz; raw 8192-Hz WAV remains canonical",
    "cursor_policy": "Playback time at each 30-fps frame midpoint; stored STFT is not recomputed live",
    "preview_decoded_peak_fs": preview_peak,
    "preview_decoded_samples": len(decoded),
    "decoder_tail_note": "The container duration is 6 seconds. AAC decoding emits 288768 samples at 48000 Hz because of codec tail padding, rather than exactly 288000; the tail follows the silent outro.",
    "output_sha256": hashlib.sha256(
        (OUT / "chirp-replay.mp4").read_bytes()
    ).hexdigest(),
    "pillow": pillow_version,
    "input_sha256": {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    },
    "renderer_sha256_lf": hashlib.sha256(
        Path(__file__).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest(),
    "ffmpeg": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
}
(OUT / "replay.json").write_text(
    json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
)
print(
    json.dumps(
        {"path": str(OUT / "chirp-replay.mp4"), "preview_decoded_peak_fs": preview_peak}
    )
)
