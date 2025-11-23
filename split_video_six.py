"""
Utility script to split a video into six consecutive segments.

Usage:
    python split_video_six.py --input path/to/video.mp4 --output-dir out/

Requires moviepy (already listed in requirements.txt).
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from moviepy.editor import VideoFileClip  # type: ignore
except ImportError:
    # MoviePy v2 fallback
    from moviepy import VideoFileClip  # type: ignore


def split_into_six(input_path: Path, output_dir: Path | None = None, prefix: str | None = None) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    out_dir = output_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    with VideoFileClip(str(input_path)) as clip:
        duration = float(clip.duration or 0.0)
        if duration <= 0:
            raise ValueError("Video duration must be greater than zero.")

        segment_count = 6
        segment_length = duration / segment_count

        base_name = prefix or input_path.stem
        for idx in range(segment_count):
            start = idx * segment_length
            # ensure final segment reaches the end to avoid rounding gaps
            end = duration if idx == segment_count - 1 else min((idx + 1) * segment_length, duration)
            end = max(end, start)  # guard against negative or reversed ranges

            part_name = f"{base_name}_part{idx + 1}.mp4"
            out_path = out_dir / part_name

            # MoviePy v1/v2 compatibility for subclip
            if hasattr(clip, 'subclip'):
                subclip = clip.subclip(start, end)
            elif hasattr(clip, 'subclipped'):
                subclip = clip.subclipped(start, end)
            else:
                raise AttributeError("VideoFileClip has no subclip or subclipped method")

            # MoviePy v1/v2 compatibility for write_videofile
            try:
                # Try MoviePy v1 style first
                subclip.write_videofile(
                    str(out_path),
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=str(out_dir / f"{base_name}_temp_audio_{idx + 1}.m4a"),
                    remove_temp=True,
                    verbose=False,
                    logger=None,
                )
            except TypeError:
                try:
                    # Try without verbose/logger (MoviePy v2)
                    subclip.write_videofile(
                        str(out_path),
                        codec="libx264",
                        audio_codec="aac",
                    )
                except Exception:
                    # Minimal fallback
                    subclip.write_videofile(str(out_path))

            if hasattr(subclip, 'close'):
                subclip.close()
            written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a video into six equal parts. Omit flags to be prompted interactively."
    )
    parser.add_argument("--input", "-i", type=Path, default=None, help="Path to the source video.")
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None, help="Directory to write the parts (defaults to current dir)."
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default=None,
        help="Optional filename prefix for the outputs (defaults to input name).",
    )

    args = parser.parse_args()

    video_path: Path | None = args.input
    while video_path is None or not video_path.exists():
        prompt = "Enter path to the input video"
        if video_path is not None:
            print(f"File not found: {video_path}")
        response = input(f"{prompt}: ").strip().strip('\"\'')
        if not response:
            continue
        candidate = Path(response).expanduser()
        if candidate.exists():
            video_path = candidate
        else:
            print(f"File not found: {candidate}")

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir = output_dir.expanduser()

    written = split_into_six(video_path, output_dir, args.prefix)
    print("Wrote segments:")
    for path in written:
        print(f" - {path}")


if __name__ == "__main__":
    main()
