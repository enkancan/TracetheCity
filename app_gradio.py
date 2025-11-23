from __future__ import annotations

import os
import tempfile
from pathlib import Path

# FORCE ENGLISH - Must be set BEFORE importing Gradio
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LANGUAGE"] = "en_US:en"
os.environ["LC_ALL"] = "en_US.UTF-8"

import numpy as np

# Core effect
from main import render_tracked_effect

# Optional video helpers
try:
    import moviepy.editor as mpy  # type: ignore
    MPY_V2 = False
except ImportError:  # pragma: no cover
    from moviepy import ImageSequenceClip  # type: ignore
    class _MPY:  # minimal shim
        ImageSequenceClip = ImageSequenceClip
    mpy = _MPY()
    MPY_V2 = True

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None

import gradio as gr  # type: ignore


def _ensure_even_size_clip(clip):
    # Ensure H.264 friendly dims
    w, h = getattr(clip, "size", (None, None))
    if w is None or h is None:
        return clip
    ew = int(w) - (int(w) % 2)
    eh = int(h) - (int(h) % 2)
    if hasattr(clip, "resize"):
        return clip.resize((ew, eh))
    if hasattr(clip, "resized"):
        return clip.resized((ew, eh))
    return clip


def _gif_to_mp4(gif_path: Path, fps_hint: float = 10.0) -> Path:
    if imageio is None:
        raise RuntimeError("imageio not available to read GIFs")
    reader = imageio.get_reader(str(gif_path), mode="I", memtest=False)
    frames = []
    for i, fr in enumerate(reader):
        frames.append(np.asarray(fr))
        if i > 1000:  # safety cap
            break
    reader.close()
    if not frames:
        raise RuntimeError("GIF has no frames")
    fps = max(1, int(round(fps_hint)))
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip = _ensure_even_size_clip(clip)
    tmp_dir = Path(tempfile.mkdtemp())
    out = tmp_dir / "gif_input.mp4"
    try:
        clip.write_videofile(
            str(out), codec="libx264", fps=fps, audio=False,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    except TypeError:  # old moviepy
        clip.write_videofile(str(out), codec="libx264", fps=fps, audio=False)
    return out


def _save_upload(upload_obj, dst_dir: Path, prefix: str = "") -> Path | None:
    if not upload_obj:
        return None
    # tuple/list => (name, bytes)
    if isinstance(upload_obj, (tuple, list)) and len(upload_obj) >= 2 and isinstance(upload_obj[1], (bytes, bytearray)):
        name = Path(str(upload_obj[0])).name or f"{prefix}upload.bin"
        out = dst_dir / f"{prefix}{name}"
        with open(out, "wb") as f:
            f.write(upload_obj[1])
        return out
    # dict from newer Gradio: {name, data? , path?}
    if isinstance(upload_obj, dict):
        name = Path(str(upload_obj.get("orig_name") or upload_obj.get("name") or f"{prefix}upload.bin")).name
        out = dst_dir / f"{prefix}{name}"
        data = upload_obj.get("data", None)
        path = upload_obj.get("path", None)
        if data is not None:
            with open(out, "wb") as f:
                f.write(data)
            return out
        if path and os.path.exists(path):
            import shutil
            shutil.copy2(path, out)
            return out
        # try if name is a real path
        nm = upload_obj.get("name", "")
        if os.path.exists(nm):
            import shutil
            shutil.copy2(nm, out)
            return out
        raise gr.Error("Uploaded file object missing data/path.")
    # path-like string
    if isinstance(upload_obj, (str, os.PathLike)):
        src = Path(upload_obj)
        if not src.exists():
            raise gr.Error("Uploaded temp file not found.")
        import shutil
        out = dst_dir / (prefix + src.name)
        shutil.copy2(src, out)
        return out
    # object with .name attribute (tempfile)
    name = getattr(upload_obj, "name", None)
    if name and os.path.exists(name):
        import shutil
        src = Path(name)
        out = dst_dir / (prefix + src.name)
        shutil.copy2(src, out)
        return out
    # file-like object
    if hasattr(upload_obj, "read"):
        raw = upload_obj.read()
        out = dst_dir / f"{prefix}upload.bin"
        with open(out, "wb") as f:
            f.write(raw)
        return out
    raise gr.Error("Unsupported upload type from Gradio.")


def run_effect(
    in_file: tuple | None,
    fill_file: tuple | None,
    out_fps: float,
    pts_per_beat: int,
    ambient_rate: float,
    jitter_px: float,
    life_frames: int,
    min_size: int,
    max_size: int,
    neighbor_links: int,
    random_spawn_frac: float,
    min_spawn_dist: float,
    use_blob_detector: bool,
    spawn_interval: float,
    single_box_mode: bool,
    single_box_size: int,
    single_box_speed: float,
    motion_spawn_bias: bool,
    motion_percentile: float,
    motion_points: int,
    growth_amp: float,
    growth_speed: float,
    single_box_pulse_amp: float,
    single_box_pulse_speed: float,
    label_source_choice: str,
    overlap_mode_choice: str,
    growth_mode_choice: str,
    gpt_label_rate: float,
    gpt_label_max: int,
    gpt_model: str,
    gpt_api_key_input: str,
    gpt_prompt_input: str,
):
    if not in_file:
        raise gr.Error("Please upload a GIF or video file.")

    def _is_on(value) -> bool:
        text = str(value).strip().lower()
        return text in {"on", "true", "yes", "enable", "enabled", "1"}

    tmp_dir = Path(tempfile.mkdtemp())

    # Save uploads to temp paths (robust to different Gradio versions)
    in_path = _save_upload(in_file, tmp_dir, prefix="in_")
    fill_path = _save_upload(fill_file, tmp_dir, prefix="fill_") if fill_file else None

    # Convert GIF -> MP4 when needed
    in_for_effect = in_path
    if in_path.suffix.lower() == ".gif":
        fps_hint = max(1.0, 1.0 / max(0.05, spawn_interval if spawn_interval > 0 else 0.5))
        in_for_effect = _gif_to_mp4(in_path, fps_hint=fps_hint)

    # Output path in temp
    out_path = tmp_dir / "render.mp4"
    render_tracked_effect(
        video_in=in_for_effect,
        video_out=out_path,
        fps=float(out_fps) if out_fps and out_fps > 0 else None,
        pts_per_beat=int(pts_per_beat),
        ambient_rate=float(ambient_rate),
        jitter_px=float(jitter_px),
        life_frames=int(life_frames),
        min_size=int(min_size),
        max_size=int(max_size),
        neighbor_links=int(neighbor_links),
        orb_fast_threshold=20,
        bell_width=4.0,
        seed=None,
        fill_video_path=fill_path,
        random_spawn_frac=float(random_spawn_frac),
        min_spawn_dist=float(min_spawn_dist),
        spawn_interval=(float(spawn_interval) if spawn_interval > 0 else None),
        use_blob_detector=_is_on(use_blob_detector),
        avoid_overlap=(overlap_mode_choice != "Allow intersections"),
        use_dynamic_growth=(growth_mode_choice == "Dynamic pulse"),
        single_box_mode=bool(single_box_mode),
        single_box_size=int(single_box_size),
        single_box_speed=float(single_box_speed),
        motion_spawn_bias=_is_on(motion_spawn_bias),
        motion_percentile=float(motion_percentile),
        motion_points=int(motion_points),
        growth_amp=float(growth_amp),
        growth_speed=float(growth_speed),
        single_box_pulse_amp=float(single_box_pulse_amp),
        single_box_pulse_speed=float(single_box_pulse_speed),
        use_gpt_labels=(label_source_choice == "GPT"),
        gpt_label_rate=float(gpt_label_rate),
        gpt_label_max=int(gpt_label_max),
        gpt_model=str(gpt_model or "gpt-4o-mini"),
        gpt_api_key=gpt_api_key_input.strip() or None,
        gpt_prompt=gpt_prompt_input.strip() or None,
    )

    return str(out_path)


light = gr.themes.Soft()
dark = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="slate",
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    background_fill_primary="*neutral_900",
    background_fill_primary_dark="*neutral_900",
    background_fill_secondary="*neutral_800",
    background_fill_secondary_dark="*neutral_800",
)

light_css = """
.gradio-container, body { background-color: #ffffff !important; color: #111827 !important; }
.gradio-container * { color: #111827 !important; }
label, .gr-markdown, .gr-label { color: #111827 !important; }
.gr-button { background: #f3f4f6 !important; color: #111827 !important; border-color: #d1d5db !important; }
.gr-input, .gr-text-input, input, textarea, select { background: #ffffff !important; color: #111827 !important; border-color: #d1d5db !important; }
.gr-slider input { color: #111827 !important; }
input[type="checkbox"], .gr-checkbox input { accent-color: #111827 !important; }
video { background: #ffffff !important; }

/* Force any lingering dark/light utility classes to white bg + black text */
.bg-white, .bg-gray-50, .bg-slate-50, .bg-neutral-50,
[class*="bg-white"], [class*="bg-gray-50"], [class*="bg-slate-50"], [class*="bg-neutral-50"] {
  background-color: #ffffff !important;
  color: #111827 !important;
}

/* File upload dropzones and chips */
.file-wrap, .upload-box, .upload-preview, .preview, .wrp, .wrap, .file.svelte-* {
  background-color: #ffffff !important;
  color: #111827 !important;
  border-color: #d1d5db !important;
}

/* Placeholder text */
input::placeholder, textarea::placeholder { color: #6b7280 !important; }
"""

dark_css = """
/* Base dark theme */
.gradio-container, body {
  background-color: #0a0a0a !important;
  color: #e5e7eb !important;
}

/* Text colors */
.gradio-container *,
label,
.gr-markdown,
.gr-label,
.gr-text-label,
span,
p {
  color: #e5e7eb !important;
}

/* Buttons */
.gr-button,
button {
  background: #1f2937 !important;
  color: #e5e7eb !important;
  border-color: #374151 !important;
}

/* Inputs and textareas */
.gr-input,
.gr-text-input,
input:not([type="checkbox"]):not([type="radio"]),
textarea,
select {
  background: #1f2937 !important;
  color: #e5e7eb !important;
  border-color: #374151 !important;
}

/* Slider values and labels */
.gr-slider input,
.gr-slider .number,
input[type="number"] {
  background: #1f2937 !important;
  color: #e5e7eb !important;
  border-color: #374151 !important;
}

/* Radio and checkbox */
input[type="checkbox"],
.gr-checkbox input {
  accent-color: #60a5fa !important;
}

input[type="radio"] {
  accent-color: #60a5fa !important;
}

/* Video player */
video {
  background: #000000 !important;
}

/* Force dark utility classes */
.bg-white, .bg-gray-50, .bg-slate-50, .bg-neutral-50,
[class*="bg-white"], [class*="bg-gray-50"], [class*="bg-slate-50"], [class*="bg-neutral-50"] {
  background-color: #1f2937 !important;
  color: #e5e7eb !important;
}

/* File upload components - More aggressive selectors */
.file-wrap,
.upload-box,
.upload-preview,
.preview,
.wrp,
.wrap,
.file.svelte-*,
[class*="upload"],
[class*="file-"],
[data-testid="file-upload"],
.svelte-1aq8tno,
.svelte-1ee5vd7,
div[data-testid*="upload"] {
  background-color: #1f2937 !important;
  color: #e5e7eb !important;
  border-color: #374151 !important;
}

/* Override any white backgrounds in upload areas */
.file-wrap *,
.upload-box *,
[class*="upload"] * {
  background-color: #1f2937 !important;
}

/* Dropzone specific */
.dz-message,
.upload-text {
  color: #9ca3af !important;
  background-color: #1f2937 !important;
}

/* Group containers */
.gr-group,
.gr-box,
.gr-form,
.gr-panel {
  background-color: #111827 !important;
  border-color: #374151 !important;
}

/* Markdown text */
.gr-markdown h1,
.gr-markdown h2,
.gr-markdown h3,
.gr-markdown p,
.gr-markdown li {
  color: #e5e7eb !important;
}

/* Placeholder text */
input::placeholder,
textarea::placeholder {
  color: #9ca3af !important;
}
"""

with gr.Blocks(title="Trace the City", theme=light, css=light_css) as demo:
    gr.Markdown("""
    # Trace the City
    Upload a GIF or video. Optionally add a fill video. Tune effect parameters and render.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Use filepath mode for simpler backend handling
            in_file = gr.File(
                label="Input GIF/Video",
                file_types=[".gif", ".mp4", ".mov", ".mkv", ".avi"],
                type="filepath",
                file_count="single",
                elem_id="input_video"
            )
            fill_file = gr.File(
                label="Fill Video (optional)",
                file_types=[".mp4", ".mov", ".mkv", ".avi"],
                type="filepath",
                file_count="single",
                elem_id="fill_video"
            )
            out_fps = gr.Slider(10, 60, value=30, step=1, label="Output FPS")
            mode_radio = gr.Radio(["Multi-Blob", "Single Box"], value="Multi-Blob", label="Mode")

            with gr.Group(visible=True) as multi_group:
                pts_per_beat = gr.Slider(1, 80, value=30, step=1, label="Pts per beat")
                ambient_rate = gr.Slider(0, 20, value=6.0, step=0.5, label="Ambient spawn rate")
                jitter_px = gr.Slider(0, 2.0, value=0.2, step=0.05, label="Jitter (px)")
                life_frames = gr.Slider(4, 120, value=24, step=1, label="Life frames")
                min_size = gr.Slider(50, 1000, value=100, step=10, label="Min size (px)")
                max_size = gr.Slider(50, 1000, value=400, step=10, label="Max size (px)")
                neighbor_links = gr.Slider(0, 8, value=4, step=1, label="Neighbor links")
                random_spawn_frac = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Random spawn fraction")
                min_spawn_dist = gr.Slider(0.0, 60.0, value=24.0, step=1.0, label="Min spawn distance (px)")
                use_blob_detector = gr.Radio(
                    ["On", "Off"],
                    value="On",
                    label="Use blob detector (SimpleBlobDetector)",
                )
                spawn_interval = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="Spawn interval (s) if no audio (0=off)")
                label_source = gr.Radio(["Coordinates", "GPT"], value="Coordinates", label="Label source")
                overlap_mode = gr.Radio(
                    ["Prevent intersections", "Allow intersections"],
                    value="Prevent intersections",
                    label="Box overlap mode",
                )
                growth_mode = gr.Radio(
                    ["Static size", "Dynamic pulse"],
                    value="Static size",
                    label="Box size animation",
                )
                gr.Markdown("---")
                motion_spawn_bias = gr.Radio(
                    ["On", "Off"],
                    value="Off",
                    label="Bias spawns to high motion areas",
                )
                motion_percentile = gr.Slider(0.5, 0.99, value=0.9, step=0.01, label="Motion percentile threshold")
                motion_points = gr.Slider(50, 2000, value=600, step=50, label="Max motion candidate points")
                growth_amp = gr.Slider(0.0, 0.8, value=0.3, step=0.02, label="Dynamic growth amplitude")
                growth_speed = gr.Slider(0.01, 0.5, value=0.12, step=0.01, label="Dynamic growth speed")
                gr.Markdown(
                    "**Multi-box parameters**\n"
                    "- **Pts per beat**: Number of fresh boxes that appear on each detected music beat.\n"
                    "- **Ambient spawn rate**: Average extra boxes per second while the audio is quiet; raise it for a busier background.\n"
                    "- **Jitter (px)**: How much tracked points wobble between frames. Lower values keep boxes steadier.\n"
                    "- **Life frames**: Frame count each box stays alive before fading; higher values keep boxes on screen longer.\n"
                    "- **Min/Max size (px)**: Minimum and maximum side length for boxes; actual sizes are sampled inside this range.\n"
                    "- **Neighbor links**: Number of nearby boxes to connect with lines. Larger counts create denser link webs.\n"
                    "- **Random spawn fraction**: Portion of beat spawns that ignore motion and appear at random coordinates.\n"
                    "- **Min spawn distance (px)**: Required space to the closest box when spawning; increase to reduce overlaps.\n"
                    "- **Use blob detector (On/Off)**: When on, prefer bright regions via OpenCV SimpleBlobDetector; off falls back to ORB points only.\n"
                    "- **Spawn interval (s)**: Forces periodic spawns when no beats are detected; set to 0 to disable.\n"
                    "- **Label source**: Choose between simple coordinate labels or GPT captions.\n"
                    "- **Box overlap mode**: Decide whether boxes are allowed to intersect each other.\n"
                    "- **Box size animation**: Select static sizing or enable the dynamic pulse animation.\n"
                    "- **Motion spawn bias (On/Off)**: When on, prioritizes high-motion regions for new boxes.\n"
                    "- **Motion percentile threshold**: Motion magnitude quantile used to define \"high\" motion pixels.\n"
                    "- **Motion points**: Number of motion pixels sampled per frame; larger values improve accuracy at the cost of speed.\n"
                    "- **Dynamic growth amplitude**: Pulse strength applied when dynamic sizing is active.\n"
                    "- **Dynamic growth speed**: Controls how fast the size pulse cycles."
                )

            with gr.Group(visible=False) as single_group:
                single_box_mode = gr.Checkbox(value=False, label="Single Box Mode (one dynamic rectangle)", interactive=False)
                single_box_size = gr.Slider(20, 300, value=120, step=1, label="Single box size (px)")
                single_box_speed = gr.Slider(0.5, 10.0, value=3.0, step=0.1, label="Single box speed (px/frame)")
                single_box_pulse_amp = gr.Slider(0.0, 0.8, value=0.15, step=0.02, label="Single box pulse amplitude (0 = static)")
                single_box_pulse_speed = gr.Slider(0.005, 0.2, value=0.05, step=0.005, label="Single box pulse speed")
                gr.Markdown(
                    "**Single-box parameters**\n"
                    "- **Single box size (px)**: Target side length for the lone roaming rectangle; multi-box size sliders do not affect it.\n"
                    "- **Single box speed (px/frame)**: Travel speed across the frame. Lower values drift smoothly, higher values bounce faster.\n"
                    "- **Single box pulse amplitude (0 = static)**: Amount of size breathing. Set to 0 to keep the box completely static.\n"
                    "- **Single box pulse speed**: How quickly the pulse completes a cycle; small values breathe slowly, large values pulse rapidly."
                )

            with gr.Group(visible=False) as gpt_group:
                gr.Markdown(
                    "Set an `OPENAI_API_KEY` environment variable to enable GPT labels. Keys are never stored."
                )
                gpt_label_rate = gr.Slider(
                    0.0, 1.0, value=0.25, step=0.05,
                    label="GPT label rate (fraction of boxes to caption)",
                    info="0 disables captions; 1 captions every eligible box."
                )
                gpt_label_max = gr.Slider(
                    0, 400, value=50, step=1,
                    label="Max GPT labels",
                    info="Hard cap on caption requests per render; raise slowly to manage API cost."
                )
                gpt_model = gr.Textbox(value="gpt-4.1", label="GPT model", lines=1)
                gpt_api_key = gr.Textbox(value="", label="API key (optional override)", type="password")
                gpt_prompt = gr.Textbox(
                    value="",
                    label="Custom GPT prompt (optional)",
                    lines=3,
                    placeholder="Leave empty for auto: 'What's in this photo?' for single video, detailed comparison for dual video",
                    info="Customize the question sent to GPT-4 Vision. Leave blank to use smart defaults."
                )

            run_btn = gr.Button("Render", variant="primary")

        with gr.Column(scale=1):
            out_video = gr.Video(label="Output", interactive=False)
            out_path_text = gr.Textbox(label="Saved path", interactive=False)

    def _run_and_return(*args):
        path = run_effect(*args)
        return path, path

    # Toggle groups based on mode
    def _set_mode(mode, label_choice):
        single = (mode == "Single Box")
        gpt_visible = (not single) and (label_choice == "GPT")
        return (
            gr.update(value=single),
            gr.update(visible=not single),
            gr.update(visible=single),
            gr.update(visible=gpt_visible),
        )

    mode_radio.change(
        _set_mode,
        inputs=[mode_radio, label_source],
        outputs=[single_box_mode, multi_group, single_group, gpt_group],
    )

    def _toggle_label_source(choice, mode):
        single = (mode == "Single Box")
        return gr.update(visible=(choice == "GPT") and (not single))

    label_source.change(
        _toggle_label_source,
        inputs=[label_source, mode_radio],
        outputs=[gpt_group],
    )

    run_btn.click(
        _run_and_return,
        inputs=[
            in_file, fill_file,
            out_fps,
            pts_per_beat, ambient_rate, jitter_px, life_frames,
            min_size, max_size, neighbor_links,
            random_spawn_frac, min_spawn_dist, use_blob_detector,
            spawn_interval,
            single_box_mode, single_box_size, single_box_speed,
            motion_spawn_bias, motion_percentile, motion_points, growth_amp, growth_speed,
            single_box_pulse_amp, single_box_pulse_speed,
            label_source, overlap_mode, growth_mode,
            gpt_label_rate, gpt_label_max, gpt_model, gpt_api_key, gpt_prompt,
        ],
        outputs=[out_video, out_path_text],
    )


if __name__ == "__main__":
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    demo.launch(server_name="127.0.0.1", inbrowser=False)
