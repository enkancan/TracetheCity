from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np

# Reuse parsing and pairing from the existing module
from compose_year_layers import (
    collect_ids_only,
)

try:
    import moviepy.editor as mpy  # type: ignore
    from moviepy.editor import concatenate_videoclips as CONCAT  # type: ignore
except ImportError:  # pragma: no cover
    from moviepy import ImageSequenceClip, ImageClip, CompositeVideoClip  # type: ignore
    from moviepy import concatenate_videoclips as CONCAT  # type: ignore
    class _MPY:
        ImageSequenceClip = ImageSequenceClip
        ImageClip = ImageClip
        CompositeVideoClip = CompositeVideoClip
    mpy = _MPY()  # type: ignore


def _ensure_even_size(w: int, h: int) -> tuple[int, int]:
    ew = int(w) - (int(w) % 2)
    eh = int(h) - (int(h) % 2)
    return max(2, ew), max(2, eh)


def _ensure_even_size_clip(clip):
    size = getattr(clip, "size", None)
    if not size:
        return clip
    w, h = size
    ew, eh = _ensure_even_size(w, h)
    if hasattr(clip, "resize"):
        return clip.resize((ew, eh))
    if hasattr(clip, "resized"):
        return clip.resized((ew, eh))
    return clip


def _build_seq_clip(paths: List[Path], fps: int, hold_sec: float | None):
    # Build slideshow via ImageClip to honor hold_sec
    if hold_sec and hold_sec > 0:
        clips = []
        for p in paths:
            c = mpy.ImageClip(str(p))
            if hasattr(c, 'set_duration'):
                c = c.set_duration(hold_sec)
            else:
                c = c.with_duration(hold_sec)
            clips.append(c)
        try:
            return CONCAT(clips, method="chain")
        except TypeError:
            return CONCAT(clips)
    return mpy.ImageSequenceClip([str(p) for p in paths], fps=fps)


def _write_clip(clip, out_path: Path, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(codec="libx264", audio=False, fps=fps)
    try:
        clip.write_videofile(str(out_path), ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"], **extra)
    except TypeError:
        clip.write_videofile(str(out_path), **extra)


def main():
    ap = argparse.ArgumentParser(description="IDs-only quick video builder (2018 vs 2024)")
    ap.add_argument("--dir", default=r"C:\\Users\\user\\OneDrive\\Desktop\\Codes\\images_galataport", help="Images folder path")
    ap.add_argument("--fg", type=int, default=2024, help="Foreground year")
    ap.add_argument("--bg", type=int, default=2018, help="Background year")
    ap.add_argument("--fps", type=int, default=2, help="Frames per second (if no hold)")
    ap.add_argument("--seconds", type=float, default=2.0, help="Seconds per image (slideshow)")
    ap.add_argument("--composite", action="store_true", help="Output a single composite video instead of two separate")
    ap.add_argument("--opacity", type=float, default=0.65, help="Foreground opacity in composite mode")
    ap.add_argument("--angle-index", type=int, default=6, help="Angle field index in filename (4 or 6)")
    ap.add_argument("--angle-round", type=int, default=2, help="Angle rounding (not used in IDs-only, kept for parser)")
    ap.add_argument("--prefer", choices=["earliest", "latest"], default="earliest", help="Pick earliest or latest date per year per ID")
    ap.add_argument("--out-bg", default=None, help="Background video path override")
    ap.add_argument("--out-fg", default=None, help="Foreground video path override")
    ap.add_argument("--out", default=None, help="Composite video path override")
    args = ap.parse_args()

    img_dir = Path(os.path.expanduser(args.dir))
    if not img_dir.exists():
        raise SystemExit(f"Images folder not found: {img_dir}")

    fg_items, bg_items = collect_ids_only(img_dir, args.fg, args.bg, args.angle_round, args.angle_index, prefer=args.prefer)
    if len(fg_items) < 1 or len(bg_items) < 1:
        raise SystemExit("No common IDs found between years.")

    fg_paths = [it.path for it in fg_items]
    bg_paths = [it.path for it in bg_items]

    if args.composite:
        out_path = Path(args.out) if args.out else img_dir / f"composite_ids_{args.bg}_bg_{args.fg}_fg.mp4"
        bg_clip = _build_seq_clip(bg_paths, fps=args.fps, hold_sec=args.seconds)
        fg_clip = _build_seq_clip(fg_paths, fps=args.fps, hold_sec=args.seconds)
        bg_clip = _ensure_even_size_clip(bg_clip)
        if hasattr(fg_clip, 'resize'):
            fg_clip = fg_clip.resize(bg_clip.size)
            if hasattr(fg_clip, 'set_opacity'):
                fg_clip = fg_clip.set_opacity(args.opacity)
            else:
                fg_clip = fg_clip.with_opacity(args.opacity)
        else:
            fg_clip = fg_clip.resized(bg_clip.size).with_opacity(args.opacity)
        comp = mpy.CompositeVideoClip([bg_clip, fg_clip])
        dur = min(bg_clip.duration, fg_clip.duration)
        if hasattr(comp, 'set_duration'):
            comp = comp.set_duration(dur)
        else:
            comp = comp.with_duration(dur)
        _write_clip(comp, out_path, fps=args.fps)
        print(f"Saved composite: {out_path}")
    else:
        out_bg = Path(args.out_bg) if args.out_bg else img_dir / f"bg_ids_{args.bg}.mp4"
        out_fg = Path(args.out_fg) if args.out_fg else img_dir / f"fg_ids_{args.fg}.mp4"
        bg_clip = _build_seq_clip(bg_paths, fps=args.fps, hold_sec=args.seconds)
        fg_clip = _build_seq_clip(fg_paths, fps=args.fps, hold_sec=args.seconds)
        bg_clip = _ensure_even_size_clip(bg_clip)
        if hasattr(fg_clip, 'resize'):
            fg_clip = fg_clip.resize(bg_clip.size)
        else:
            fg_clip = fg_clip.resized(bg_clip.size)
        _write_clip(bg_clip, out_bg, fps=args.fps)
        _write_clip(fg_clip, out_fg, fps=args.fps)
        print(f"Saved background: {out_bg}")
        print(f"Saved foreground: {out_fg}")


if __name__ == "__main__":
    main()

