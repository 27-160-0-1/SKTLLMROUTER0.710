# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Render a captured terminal recording (tools/capture_demo.ps1) to an MP4.

Draws the session on a fixed character grid -- Consolas for ASCII, Malgun Gothic for the
Korean the official tool prints, both advanced by whole cells so the banner rules stay
aligned -- and pipes the frames to ffmpeg.

Idle stretches (a container running for half a minute) are compressed so the video stays
watchable.  The clock burned into the corner always shows the *real* elapsed time of the
captured run, and a marker appears while a wait is being skipped, so nothing on screen
misrepresents how long the run actually took.

Usage:
    python tools/render_demo_video.py --recording demo_recording_official.tsv \
        --output demo_official.mp4 --title "SKT LLM Router - official spec check"
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

BG = (16, 20, 26)
FG = (201, 211, 220)
DIM = (107, 122, 136)
CYAN = (111, 179, 210)
YELLOW = (217, 164, 65)
GREEN = (95, 191, 143)
HASH = (169, 198, 216)
PANEL = (23, 29, 37)

WIN_FONTS = pathlib.Path("C:/Windows/Fonts")


def classify(line: str) -> tuple[int, int, int]:
    s = line.strip()
    if s.startswith("=") and set(s) == {"="}:
        return CYAN
    if s.startswith("STEP ") and s[5:6].isdigit():
        return CYAN
    if s.startswith(">> tier="):
        return YELLOW
    if s.startswith("FINAL SCORE") or ": PASS (" in s:
        return GREEN
    if s.startswith(("OK:", "-> ", "Expected:", "Report these", "Quote for accuracy")):
        return GREEN
    if s.startswith(("wall time", "주의:", "The STEP", "public Train+Dev check",
                     "path, measured", "STEP 2 times")):
        return DIM
    if "sha256" in s:
        return HASH
    return FG


class Grid:
    """A monospace cell grid that can mix a Latin and a Korean face.

    Every character is placed at its own cell and drawn from a shared baseline, so the two
    faces cannot drift apart and the banner rules stay aligned.  Korean glyphs occupy two
    cells, as they do in a terminal, and the Korean face is sized to fill them.
    """

    def __init__(self, size: int):
        self.mono = ImageFont.truetype(str(WIN_FONTS / "consola.ttf"), size)
        self.cw = int(round(self.mono.getlength("M")))
        self.ch = int(round(size * 1.62))
        # size the Korean face so one glyph fills the two cells a wide character occupies
        self.kr = ImageFont.truetype(str(WIN_FONTS / "malgun.ttf"), max(8, int(self.cw * 1.9)))
        self.ascent = self.mono.getmetrics()[0]

    @staticmethod
    def _wide(ch: str) -> bool:
        return ord(ch) > 0x2E7F

    def cells(self, text: str) -> int:
        return sum(2 if self._wide(ch) else 1 for ch in text)

    def draw_line(self, d: ImageDraw.ImageDraw, x: int, y: int, text: str, color) -> None:
        baseline = y + self.ascent
        col = 0
        for ch in text:
            if ch != " ":
                wide = self._wide(ch)
                d.text((x + col * self.cw, baseline), ch,
                       font=(self.kr if wide else self.mono), fill=color, anchor="ls")
            col += 2 if self._wide(ch) else 1


def load(path: pathlib.Path) -> list[tuple[int, str]]:
    rows = []
    for line in path.read_text("utf-8").splitlines():
        ms, _, text = line.partition("\t")
        rows.append((int(ms), text.rstrip()))
    return rows


def build_timeline(rows, fps: int, max_gap_ms: int):
    """Map video frames to (real elapsed ms, lines shown, skipping?)."""
    frames = []
    video_ms = 0
    step = 1000 // fps
    for index, (real_ms, _) in enumerate(rows):
        prev_real = rows[index - 1][0] if index else 0
        gap = real_ms - prev_real
        shown = min(gap, max_gap_ms)
        skipping = gap > max_gap_ms
        elapsed = 0
        while elapsed < shown:
            frames.append((prev_real + int(gap * (elapsed / shown)) if shown else real_ms,
                           index, skipping and elapsed > max_gap_ms // 3))
            elapsed += step
            video_ms += step
    frames.append((rows[-1][0], len(rows), False))
    for _ in range(fps * 3):  # hold the last screen
        frames.append((rows[-1][0], len(rows), False))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recording", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--font-size", type=int, default=17)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-gap-ms", type=int, default=2000,
                    help="longest idle stretch shown before it is compressed")
    a = ap.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 2

    rows = load(a.recording)
    grid = Grid(a.font_size)
    frames = build_timeline(rows, a.fps, a.max_gap_ms)
    header_h = grid.ch + 14
    body_rows = (a.height - header_h - 16) // grid.ch

    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{a.width}x{a.height}", "-r", str(a.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.output)],
        stdin=subprocess.PIPE,
    )

    total_real = rows[-1][0]
    for real_ms, count, skipping in frames:
        image = Image.new("RGB", (a.width, a.height), BG)
        d = ImageDraw.Draw(image)
        d.rectangle([0, 0, a.width, header_h], fill=PANEL)
        for i, cx in enumerate((16, 32, 48)):
            d.ellipse([cx - 5, header_h // 2 - 5, cx + 5, header_h // 2 + 5], fill=(57, 66, 77))
        grid.draw_line(d, 70, 8, a.title, DIM)
        clock = f"{real_ms // 60000}:{real_ms // 1000 % 60:02d} / {total_real // 60000}:{total_real // 1000 % 60:02d}"
        if skipping:
            clock = ">> 대기 압축   " + clock
        grid.draw_line(d, a.width - grid.cells(clock) * grid.cw - 40, 8, clock,
                       YELLOW if skipping else DIM)
        d.line([0, header_h, a.width, header_h], fill=(35, 43, 52))

        visible = rows[max(0, count - body_rows):count]
        y = header_h + 10
        for _, text in visible:
            grid.draw_line(d, 16, y, text[:150], classify(text))
            y += grid.ch
        if count < len(rows):
            d.rectangle([16, y + 3, 16 + grid.cw, y + grid.ch - 3], fill=FG)
        proc.stdin.write(image.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        return proc.returncode
    size_mb = a.output.stat().st_size / 1e6
    print(f"wrote {a.output}  {len(frames)} frames @ {a.fps}fps "
          f"= {len(frames) / a.fps:.1f}s video  {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
