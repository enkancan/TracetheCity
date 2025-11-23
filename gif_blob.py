from __future__ import annotations

from pathlib import Path
import os
import sys

from main import render_tracked_effect
import tempfile
import numpy as np
try:
    import moviepy.editor as mpy
except Exception:
    from moviepy import ImageSequenceClip
    class _MPY:
        ImageSequenceClip = ImageSequenceClip
    mpy = _MPY()
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def ask(msg: str, default: str = "") -> str:
    try:
        return input(msg).strip()
    except EOFError:
        return default


def main():
    print("Apply blob-tracing effect on a GIF/video and output MP4.")
    src = ask("Enter GIF/video path: ")
    src = src.strip().strip("\"'")
    if not src:
        print("No input path provided.")
        return
    in_path = Path(os.path.expanduser(src))
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return

    out_str = ask("Output MP4 path (blank=gif_blob_output.mp4): ")
    out_path = Path(out_str) if out_str else Path.cwd() / "gif_blob_output.mp4"

    # Calmer but visible defaults when no audio: use spawn_interval
    spawn_interval_str = ask("Spawn interval seconds when no audio (blank=0.5): ")
    try:
        spawn_interval = float(spawn_interval_str) if spawn_interval_str else 0.5
    except ValueError:
        spawn_interval = 0.5

    # If input is a GIF, convert to a temporary MP4 first (some GIFs have no duration metadata)
    in_for_effect = in_path
    if in_path.suffix.lower() == ".gif":
        if imageio is None:
            print("imageio not available to read GIF; please install imageio.")
            return
        try:
            # Use streaming reader to avoid memtest limits
            reader = imageio.get_reader(str(in_path), mode='I', memtest=False)
            frames = []
            for i, fr in enumerate(reader):
                frames.append(np.asarray(fr))
                if i > 800:  # safety cap
                    break
            reader.close()
            if len(frames) < 1:
                print("GIF has no frames.")
                return
            # Build sequence clip; use hold from spawn_interval for smoothness
            fps = max(1, int(round(1.0 / max(0.05, spawn_interval))))
            clip = mpy.ImageSequenceClip([np.asarray(f) for f in frames], fps=fps)
            # Ensure even size
            w, h = getattr(clip, 'size', (None, None))
            if w is None or h is None:
                w, h = frames[0].shape[1], frames[0].shape[0]
            even_w = int(w) - (int(w) % 2)
            even_h = int(h) - (int(h) % 2)
            if hasattr(clip, 'resize'):
                clip = clip.resize((even_w, even_h))
            else:
                clip = clip.resized((even_w, even_h))
            # Write temp mp4
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_mp4 = tmp_dir / "gif_input.mp4"
            try:
                clip.write_videofile(str(tmp_mp4), codec="libx264", fps=fps, audio=False, ffmpeg_params=["-pix_fmt","yuv420p","-movflags","+faststart"]) 
            except TypeError:
                clip.write_videofile(str(tmp_mp4), codec="libx264", fps=fps, audio=False)
            in_for_effect = tmp_mp4
            print(f"Converted GIF to MP4: {in_for_effect}")
        except Exception as e:
            print(f"Failed to convert GIF: {e}")
            return

    # Effect intensity
    pts_per_beat = 30
    ambient_rate = 6.0
    jitter_px = 0.2
    life_frames = 24
    min_size = 100
    max_size = 400
    neighbor_links = 4

    render_tracked_effect(
        video_in=in_for_effect,
        video_out=out_path,
        fps=None,
        pts_per_beat=pts_per_beat,
        ambient_rate=ambient_rate,
        jitter_px=jitter_px,
        life_frames=life_frames,
        min_size=min_size,
        max_size=max_size,
        neighbor_links=neighbor_links,
        orb_fast_threshold=20,
        bell_width=4.0,
        seed=None,
        fill_video_path=None,
        random_spawn_frac=0.5,
        min_spawn_dist=24.0,
        spawn_interval=spawn_interval,
    )
    print(f"Done. Saved: {out_path}")


if __name__ == "__main__":
    main()
