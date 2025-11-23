from __future__ import annotations

import os
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# Import the core effect
from main import render_tracked_effect

# Optional helpers for GIF conversion
try:
    import moviepy.editor as mpy
    MPY_V2 = False
except ImportError:
    from moviepy import ImageSequenceClip
    class _MPY:
        ImageSequenceClip = ImageSequenceClip
    mpy = _MPY()
    MPY_V2 = True

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def convert_gif_to_mp4(gif_path: Path, fps_hint: float) -> Path:
    if imageio is None:
        raise RuntimeError("imageio not available to read GIFs")
    reader = imageio.get_reader(str(gif_path), mode='I', memtest=False)
    frames = []
    for i, fr in enumerate(reader):
        frames.append(np.asarray(fr))
        if i > 1000:  # safety cap
            break
    reader.close()
    if not frames:
        raise RuntimeError("GIF had no frames")
    fps = max(1, int(round(fps_hint)))
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    # Ensure even size for H.264
    w, h = getattr(clip, 'size', (frames[0].shape[1], frames[0].shape[0]))
    even_w = int(w) - (int(w) % 2)
    even_h = int(h) - (int(h) % 2)
    if hasattr(clip, 'resize'):
        clip = clip.resize((even_w, even_h))
    else:
        clip = clip.resized((even_w, even_h))
    tmp_mp4 = gif_path.with_suffix('').with_name(gif_path.stem + "_temp_mp4.mp4")
    try:
        clip.write_videofile(str(tmp_mp4), codec="libx264", fps=fps, audio=False, ffmpeg_params=["-pix_fmt","yuv420p","-movflags","+faststart"]) 
    except TypeError:
        clip.write_videofile(str(tmp_mp4), codec="libx264", fps=fps, audio=False)
    return tmp_mp4


class BlobUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Trace the City - UI")
        self.geometry("640x520")
        self.resizable(True, True)

        pad = {"padx": 6, "pady": 4}

        # Paths
        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar(value=str(Path.cwd() / "output.mp4"))
        self.fill_var = tk.StringVar()

        row = 0
        ttk.Label(self, text="Input GIF/Video").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.in_var, width=60).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self.browse_input).grid(row=row, column=2, **pad)

        row += 1
        ttk.Label(self, text="Output MP4").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.out_var, width=60).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Save As", command=self.browse_output).grid(row=row, column=2, **pad)

        row += 1
        ttk.Label(self, text="Fill Video (inside squares, optional)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.fill_var, width=60).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self.browse_fill).grid(row=row, column=2, **pad)

        # Params
        def add_param(label, var, col=0):
            nonlocal row
            ttk.Label(self, text=label).grid(row=row, column=col, sticky="w", **pad)
            e = ttk.Entry(self, textvariable=var, width=10)
            e.grid(row=row, column=col+1, sticky="w", **pad)
            row += 1

        self.spawn_interval = tk.StringVar(value="0.5")
        self.pts_per_beat = tk.StringVar(value="30")
        self.ambient_rate = tk.StringVar(value="6.0")
        self.jitter_px = tk.StringVar(value="0.2")
        self.life_frames = tk.StringVar(value="24")
        self.min_size = tk.StringVar(value="100")
        self.max_size = tk.StringVar(value="400")
        self.neighbor_links = tk.StringVar(value="4")
        self.random_spawn_frac = tk.StringVar(value="0.5")
        self.min_spawn_dist = tk.StringVar(value="24.0")
        self.use_blob = tk.BooleanVar(value=True)

        add_param("Spawn interval (s, no audio)", self.spawn_interval)
        add_param("Pts per beat", self.pts_per_beat)
        add_param("Ambient rate", self.ambient_rate)
        add_param("Jitter px", self.jitter_px)
        add_param("Life frames", self.life_frames)
        add_param("Min size", self.min_size)
        add_param("Max size", self.max_size)
        add_param("Neighbor links", self.neighbor_links)
        add_param("Random spawn frac", self.random_spawn_frac)
        add_param("Min spawn dist", self.min_spawn_dist)

        ttk.Checkbutton(self, text="Use blob detector (SimpleBlobDetector)", variable=self.use_blob).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        # Run
        self.status = tk.StringVar(value="Idle")
        ttk.Button(self, text="Run", command=self.run_effect).grid(row=row, column=0, **pad)
        ttk.Label(self, textvariable=self.status).grid(row=row, column=1, sticky="w", **pad)

        self.columnconfigure(1, weight=1)

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Media", "*.mp4;*.mov;*.mkv;*.avi;*.gif"), ("All", "*.*")])
        if path:
            self.in_var.set(path)

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4"), ("All", "*.*")])
        if path:
            self.out_var.set(path)

    def browse_fill(self):
        path = filedialog.askopenfilename(filetypes=[("Media", "*.mp4;*.mov;*.mkv;*.avi"), ("All", "*.*")])
        if path:
            self.fill_var.set(path)

    def run_effect(self):
        t = threading.Thread(target=self._run_effect_worker, daemon=True)
        t.start()

    def _run_effect_worker(self):
        try:
            self.status.set("Running…")
            in_path = Path(self.in_var.get().strip().strip("\"'"))
            out_path = Path(self.out_var.get().strip().strip("\"'"))
            fill_path_str = self.fill_var.get().strip().strip("\"'")
            fill_path = Path(fill_path_str) if fill_path_str else None
            if not in_path.exists():
                raise FileNotFoundError(f"Input not found: {in_path}")

            # Parse params
            spawn_interval = float(self.spawn_interval.get() or 0)
            pts_per_beat = int(self.pts_per_beat.get() or 30)
            ambient_rate = float(self.ambient_rate.get() or 0)
            jitter_px = float(self.jitter_px.get() or 0)
            life_frames = int(self.life_frames.get() or 10)
            min_size = int(self.min_size.get() or 10)
            max_size = int(self.max_size.get() or 40)
            neighbor_links = int(self.neighbor_links.get() or 2)
            random_spawn_frac = float(self.random_spawn_frac.get() or 0.5)
            min_spawn_dist = float(self.min_spawn_dist.get() or 24.0)
            use_blob = bool(self.use_blob.get())

            in_for_effect = in_path
            if in_path.suffix.lower() == ".gif":
                # Convert GIF to MP4 for robustness
                fps_hint = max(1.0, 1.0 / max(0.05, spawn_interval if spawn_interval > 0 else 0.5))
                in_for_effect = convert_gif_to_mp4(in_path, fps_hint=fps_hint)

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
                fill_video_path=fill_path,
                random_spawn_frac=random_spawn_frac,
                min_spawn_dist=min_spawn_dist,
                spawn_interval=(spawn_interval if spawn_interval > 0 else None),
                use_blob_detector=use_blob,
            )

            self.status.set("Done")
            messagebox.showinfo("Done", f"Saved: {out_path}")
        except Exception as e:
            self.status.set("Error")
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = BlobUI()
    app.mainloop()
