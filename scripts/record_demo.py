#!/usr/bin/env python3
"""Record the Sloth Mode web UI doing a full plan → apply → undo → brief run.

Produces, in the output directory:

* ``sloth-demo.mp4``   — the screen recording, for YouTube (needs ffmpeg)
* ``sloth-demo.gif``   — an animated still-frame version, for the README,
  because GitHub will not play a video file committed to a repo
* ``frames/*.png``     — the individual screenshots

It works on a throwaway copy of ``examples/messy-downloads``, so your real
folders are never involved.

    pip install -e ".[demo]"
    python scripts/record_demo.py --api cloud          # real model
    python scripts/record_demo.py --no-model           # heuristics only

Not part of the test suite: it needs a browser, and sometimes a model.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
URL_LINE = re.compile(r"the sloth is at (http://\S+)")

#: Milliseconds between frames while filming. 250ms (4fps) is plenty for a
#: UI that mostly sits still, and keeps the frame count sane.
FRAME_MS = 250


def start_ui(folder: Path, port: int, extra: list[str]) -> tuple[subprocess.Popen, str]:
    """Launch `sloth gui` and wait for it to print its URL."""
    command = [
        sys.executable,
        "-m",
        "sloth",
        *extra,
        "gui",
        str(folder),
        "--port",
        str(port),
        "--no-browser",
    ]
    print(f"$ {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                raise RuntimeError("the UI exited before it was ready")
            continue
        match = URL_LINE.search(line)
        if match:
            return process, match.group(1)
    raise TimeoutError("the UI never announced a URL")


#: Where a prebuilt Chromium may already live (some CI images ship one whose
#: build number does not match the installed Playwright). Using it beats
#: downloading 150MB of browser to record a 40-second clip.
PREINSTALLED = Path("/opt/pw-browsers")


def find_chromium() -> str | None:
    """An existing chrome binary, newest build first, or None."""
    if not PREINSTALLED.is_dir():
        return None
    candidates = sorted(PREINSTALLED.glob("chromium-*/chrome-linux/chrome"), reverse=True)
    return str(candidates[0]) if candidates else None


def record(url: str, out_dir: Path, slow: float) -> Path:
    """Drive the UI, saving a frame several times a second.

    Playwright can record video itself, but only with its own bundled ffmpeg
    build, which is one more thing to download and keep in step. Screenshots
    plus the system ffmpeg need neither, and give the GIF its frames for free.
    """
    from playwright.sync_api import sync_playwright

    raw = out_dir / "_frames"
    frames = out_dir / "frames"
    for directory in (raw, frames):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    counter = {"n": 0}

    with sync_playwright() as playwright:
        executable = find_chromium()
        if executable:
            print(f"  using {executable}")
        browser = playwright.chromium.launch(executable_path=executable)
        context = browser.new_context(
            viewport={"width": 1180, "height": 880},
            color_scheme="light",
        )
        page = context.new_page()

        def frame() -> Path:
            path = raw / f"{counter['n']:05d}.png"
            page.screenshot(path=str(path))
            counter["n"] += 1
            return path

        def hold(seconds: float) -> Path:
            """Keep filming for `seconds`. Returns the last frame taken."""
            deadline = time.monotonic() + seconds * slow
            last = frame()
            while time.monotonic() < deadline:
                page.wait_for_timeout(int(FRAME_MS))
                last = frame()
            return last

        def keyframe(name: str, last: Path) -> None:
            """Mark a frame as one of the GIF's stills."""
            target = frames / f"{len(list(frames.glob('*.png'))):02d}-{name}.png"
            shutil.copyfile(last, target)
            print(f"  keyframe: {target.name}")

        print("recording ...")
        page.goto(url)
        page.wait_for_selector("#status .dot.ok, #status .dot.off", timeout=30_000)
        keyframe("open", hold(1.8))

        # 1. Plan — reads the folder, proposes, changes nothing.
        page.click("#btn-plan")
        hold(0.4)
        page.wait_for_selector("#tidy-out table tr:nth-child(3)", timeout=240_000)
        keyframe("plan", hold(3.0))

        # 2. Apply — the only step that touches the disk.
        page.click("#btn-apply")
        page.wait_for_selector("text=/Moved \\d+ file/", timeout=60_000)
        keyframe("applied", hold(2.4))

        # 3. Undo — the promise the whole thing rests on.
        page.click("#btn-undo")
        page.wait_for_selector("text=/Put \\d+ file\\(s\\) back/", timeout=60_000)
        keyframe("undone", hold(2.4))

        # 4. Brief — the other agent, on the example chat logs.
        page.fill("#brief-path", str(REPO / "examples" / "chats"))
        hold(0.6)
        page.click("#btn-brief")
        page.wait_for_selector("#brief-out p", timeout=240_000)
        page.locator("#brief-out").scroll_into_view_if_needed()
        keyframe("brief", hold(3.4))

        context.close()
        browser.close()

    print(f"captured {counter['n']} frames")
    return raw


def to_mp4(raw: Path, out_dir: Path) -> Path | None:
    """Encode the frames into something YouTube will take happily."""
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found; skipping mp4 (the frames are still there)")
        return None
    mp4 = out_dir / "sloth-demo.mp4"
    mp4.unlink(missing_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(1000 / FRAME_MS),
            "-i",
            str(raw / "%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            # The browser viewport is not 16:9, so centre it on a 1080p
            # canvas in the page's own background colour, rather than letting
            # YouTube put black bars down either side.
            "-vf",
            (
                "fps=25,"
                "scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xfaf7f2"
            ),
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
    )
    print(f"mp4:   {mp4} ({mp4.stat().st_size / 1_000_000:.1f} MB)")
    return mp4


def to_gif(out_dir: Path, width: int = 900, frame_ms: int = 1800) -> Path | None:
    """A still-frame GIF, because a README cannot play a video."""
    from PIL import Image

    frames = sorted((out_dir / "frames").glob("*.png"))
    if not frames:
        return None
    images = []
    for path in frames:
        image = Image.open(path).convert("RGB")
        height = round(image.height * width / image.width)
        images.append(image.resize((width, height), Image.LANCZOS))

    gif = out_dir / "sloth-demo.gif"
    images[0].save(
        gif,
        save_all=True,
        append_images=images[1:],
        duration=[frame_ms] * (len(images) - 1) + [frame_ms * 2],
        loop=0,
        optimize=True,
    )
    print(f"gif:   {gif} ({gif.stat().st_size / 1_000_000:.1f} MB)")
    return gif


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", help="backend to demo with (e.g. cloud, hf, ollama)")
    parser.add_argument("--model", help="model name override")
    parser.add_argument(
        "--no-model", action="store_true", help="heuristics only, no inference"
    )
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "demo")
    parser.add_argument(
        "--slow", type=float, default=1.0, help="multiply every pause by this"
    )
    args = parser.parse_args()

    extra: list[str] = []
    if args.no_model:
        extra.append("--no-model")
    if args.api:
        extra += ["--api", args.api]
    if args.model:
        extra += ["--model", args.model]

    # Never record against someone's real Downloads folder.
    workspace = args.out / "_folder"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "examples" / "messy-downloads", workspace)
    print(f"demo folder: {workspace} ({len(list(workspace.iterdir()))} files)")

    process, url = start_ui(workspace, args.port, extra)
    try:
        raw = record(url, args.out, args.slow)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()

    to_mp4(raw, args.out)
    to_gif(args.out)
    shutil.rmtree(workspace, ignore_errors=True)
    shutil.rmtree(raw, ignore_errors=True)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
