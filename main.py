from __future__ import annotations

import logging
import os
import random
import tempfile
import warnings
from pathlib import Path
import base64
import hashlib

import cv2
try:
    import librosa  # optional (for audio onset detection)
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
try:
    import moviepy.editor as mpy
except ImportError:
    # MoviePy v2 fallback
    from moviepy import VideoFileClip, VideoClip
    class _MPY:
        VideoFileClip = VideoFileClip
        VideoClip = VideoClip
    mpy = _MPY()
try:
    import requests  # optional for GPT labeling
except Exception:
    requests = None
import numpy as np
import ffmpeg

# Suppress only specific MoviePy warning
warnings.filterwarnings(
    "ignore",
    message="Warning: in file .* bytes wanted but 0 bytes read.*",
    category=UserWarning,
    module="moviepy.video.io.ffmpeg_reader"
)


def _extract_audio(video_path: Path, sr: int = 22050) -> Path | None:
    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "temp_audio.wav"
    clip = mpy.VideoFileClip(str(video_path))
    try:
        if getattr(clip, 'audio', None) is None:
            return None
        clip.audio.write_audiofile(str(wav_path), fps=sr, logger=None, verbose=False)
        return wav_path
    except Exception:
        return None


def _detect_onsets(wav_path: Path | None, sr: int = 22050) -> np.ndarray:
    if wav_path is None or not LIBROSA_AVAILABLE:
        return np.array([], dtype=np.float32)
    y, _ = librosa.load(str(wav_path), sr=sr)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time")


def get_rotation(path: Path) -> int:
    try:
        meta = ffmpeg.probe(str(path))
        for stream in meta["streams"]:
            if stream["codec_type"] == "video" and "tags" in stream and "rotate" in stream["tags"]:
                return int(stream["tags"]["rotate"])
    except Exception:
        return 0
    return 0


def _with_duration(clip, dur):
    if hasattr(clip, "set_duration"):
        return clip.set_duration(dur)
    if hasattr(clip, "with_duration"):
        return clip.with_duration(dur)
    return clip


def _with_resize(clip, size):
    if hasattr(clip, "resize"):
        return clip.resize(size)
    if hasattr(clip, "resized"):
        return clip.resized(size)
    return clip


def _with_fps(clip, fps):
    if hasattr(clip, "set_fps"):
        return clip.set_fps(fps)
    if hasattr(clip, "with_fps"):
        return clip.with_fps(fps)
    return clip


def _with_audio(clip, audio):
    if hasattr(clip, "set_audio"):
        return clip.set_audio(audio)
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio)
    return clip

class TrackedPoint:
    def __init__(self, pos: tuple[float, float], life: int, size: int, label: str | None = None, phase: float | None = None):
        self.pos = np.array(pos, dtype=np.float32)
        self.life = life
        self.size = size
        self.label = label
        self.phase = float(phase) if phase is not None else float(np.random.uniform(0, 2 * np.pi))


def _sample_size_bell(min_s: int, max_s: int, width_div: float = 6.0) -> int:
    mean = (min_s + max_s) / 2.0
    sigma = (max_s - min_s) / width_div
    for _ in range(10):
        val = np.random.normal(mean, sigma)
        if min_s <= val <= max_s:
            return int(val)
    return int(np.clip(val, min_s, max_s))


def render_tracked_effect(
    video_in: Path,
    video_out: Path,
    *,
    fps: float | None,
    pts_per_beat: int,
    ambient_rate: float,
    jitter_px: float,
    life_frames: int,
    min_size: int,
    max_size: int,
    neighbor_links: int,
    orb_fast_threshold: int,
    bell_width: float,
    seed: int | None,
    fill_video_path: Path | None = None,
    random_spawn_frac: float = 0.4,
    min_spawn_dist: float = 22.0,
    spawn_interval: float | None = None,
    max_active: int = 200,
    avoid_overlap: bool = True,
    min_gap_px: float = 6.0,
    # Motion-biased spawning and dynamic growth
    motion_spawn_bias: bool = False,
    motion_percentile: float = 0.9,
    motion_points: int = 600,
    growth_amp: float = 0.3,
    growth_speed: float = 0.12,
    use_dynamic_growth: bool = False,
    # Single-box pulse controls
    single_box_pulse_amp: float = 0.15,
    single_box_pulse_speed: float = 0.05,
    # Single-box mode (procedural dynamic rectangle)
    single_box_mode: bool = False,
    single_box_size: int = 120,
    single_box_speed: float = 3.0,
    single_box_thickness: int = 2,
    # GPT one-word labeling
    use_gpt_labels: bool = False,
    gpt_label_rate: float = 0.25,
    gpt_label_max: int = 50,
    gpt_model: str = "gpt-4.1",
    gpt_api_key: str | None = None,
    gpt_prompt: str | None = None,
    use_blob_detector: bool = True,
    blob_min_area: float = 30.0,
    blob_max_area: float = 5000.0,
    blob_min_threshold: float = 10.0,
    blob_max_threshold: float = 200.0,
    blob_threshold_step: float = 10.0,
):
    print(f"🎥 render_tracked_effect: Video girdi = {video_in}")
    print(f"📤 Çıktı yolu = {video_out}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print("🔄 Video dosyası yükleniyor...")
    rotation = get_rotation(video_in)
    print(f"🔃 Rotasyon = {rotation}")

    print("🎬 VideoFileClip oluşturuluyor...")
    clip = mpy.VideoFileClip(str(video_in))
    print(f"✅ Clip yüklendi: {clip.duration}s, {clip.fps} fps")

    if rotation == 90:
        clip = clip.rotate(-90)
    elif rotation == 270:
        clip = clip.rotate(90)
    elif rotation == 180:
        clip = clip.rotate(180)

    fps = fps or clip.fps
    clip = _with_duration(clip, clip.duration)
    frame_w, frame_h = clip.size

    overlay_clip = None
    overlay_duration = None
    if fill_video_path is not None and fill_video_path.exists():
        try:
            logging.info(f"Loading fill video: {fill_video_path}")
            overlay_clip = mpy.VideoFileClip(str(fill_video_path))
            o_rot = get_rotation(fill_video_path)
            if o_rot == 90:
                overlay_clip = overlay_clip.rotate(-90)
            elif o_rot == 270:
                overlay_clip = overlay_clip.rotate(90)
            elif o_rot == 180:
                overlay_clip = overlay_clip.rotate(180)
            overlay_clip = _with_resize(overlay_clip, clip.size)
            overlay_duration = overlay_clip.duration or clip.duration
            logging.info(
                "Fill video ready (size=%s, duration=%.2fs)",
                overlay_clip.size if hasattr(overlay_clip, 'size') else 'unknown',
                overlay_duration if overlay_duration else -1,
            )
        except Exception as e:
            logging.warning(f"Failed to load fill video: {e}")
            overlay_clip = None
            overlay_duration = None
    elif fill_video_path is not None:
        logging.warning(f"Fill video path not found: {fill_video_path}")

    # Skip audio/onset if in single-box mode (faster)
    print(f"🔊 Audio işleme başlıyor... (single_box_mode={single_box_mode})")
    if single_box_mode:
        onset_times = np.array([], dtype=np.float32)
        print("⏭️  Single box mode - audio atlandı")
    else:
        print("🎵 Audio extract ediliyor...")
        wav_path = _extract_audio(video_in)
        print(f"✅ Audio extract tamamlandı: {wav_path}")

        print("🎶 Onset detection başlıyor...")
        onset_times = _detect_onsets(wav_path)
        print(f"✅ {len(onset_times)} onset bulundu")

        if onset_times.size == 0 and spawn_interval and spawn_interval > 0:
            onset_times = np.arange(0.0, float(clip.duration or 0.0), float(spawn_interval), dtype=np.float32)
            print(f"⚠️  Onset bulunamadı, spawn_interval kullanılıyor: {len(onset_times)} nokta")
    logging.info("%d onsets detected", len(onset_times))

    print("🔍 ORB detector oluşturuluyor...")
    orb = cv2.ORB_create(nfeatures=300, fastThreshold=orb_fast_threshold)
    print("✅ ORB hazır (300 features max)")

    # Optional: SimpleBlobDetector for true blob-based points
    blob_detector = None
    print(f"🔵 Blob detector: {use_blob_detector}")
    if use_blob_detector:
        print("🔧 Blob detector parametreleri ayarlanıyor...")
        try:
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = blob_min_threshold
            params.maxThreshold = blob_max_threshold
            params.thresholdStep = blob_threshold_step
            params.filterByArea = True
            params.minArea = blob_min_area
            params.maxArea = blob_max_area
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False
            params.filterByColor = False
            if hasattr(cv2, 'SimpleBlobDetector_create'):
                blob_detector = cv2.SimpleBlobDetector_create(params)
            else:
                blob_detector = cv2.SimpleBlobDetector(params)
            print("✅ Blob detector hazır")
            logging.info("SimpleBlobDetector initialized (area %.1f-%.1f)", blob_min_area, blob_max_area)
        except Exception as e:
            print(f"⚠️ Blob detector hatası: {e}")
            logging.warning(f"Failed to init SimpleBlobDetector: {e}")
            blob_detector = None
    else:
        print("➡️ Blob detector kullanılmıyor")

    print("📝 Değişkenler başlatılıyor...")
    active: list[TrackedPoint] = []
    onset_idx = 0
    prev_gray: np.ndarray | None = None
    prev_frame_color: np.ndarray | None = None
    frame_counter = 0
    print("✅ Değişkenler hazır")
    # Calm movement: exponential smoothing of tracked positions (0=no smoothing)
    pos_smooth = 0.85

    # State for single-box mode
    sb_pos = None
    sb_vel = None
    sb_phase = 0.0
    gpt_calls = 0
    gpt_counter = 0
    gpt_cache: dict[str, str] = {}

    def _roi_signature(img: np.ndarray | None) -> str | None:
        if img is None or not img.size:
            return None
        try:
            return hashlib.sha1(img.tobytes()).hexdigest()
        except Exception:
            return None

    def _label_roi_with_gpt(
        current_roi: np.ndarray,
        prev_roi: np.ndarray | None = None,
        fill_roi: np.ndarray | None = None,
        cache_key: str | None = None,
    ) -> str | None:
        nonlocal gpt_calls, gpt_counter
        if not use_gpt_labels:
            return None
        if gpt_calls >= gpt_label_max:
            return None
        if cache_key and cache_key in gpt_cache:
            return gpt_cache[cache_key]
        api_key = (gpt_api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not api_key or requests is None:
            return None
        try:
            def _encode(img: np.ndarray) -> str | None:
                ok, enc = cv2.imencode('.jpg', img)
                if not ok:
                    return None
                return base64.b64encode(enc.tobytes()).decode('ascii')

            curr_b64 = _encode(current_roi)
            if not curr_b64:
                return None
            prev_b64 = _encode(prev_roi) if prev_roi is not None and prev_roi.size else None
            fill_b64 = _encode(fill_roi) if fill_roi is not None and fill_roi.size else None

            # Use custom prompt or default based on video mode
            if gpt_prompt:
                prompt_text = gpt_prompt
            elif fill_video_path is None:
                # Single video mode: simple English prompt
                prompt_text = "What's in this photo? Describe in 8 words or less."
            else:
                # Dual video mode: detailed comparison prompt
                prompt_text = (
                    "Image 1 = current frame from the input video.\n"
                    "Image 2 (if provided) = the same location captured in the fill video at another time.\n"
                    "Image 3 (if provided) = the previous frame from the input video for motion context.\n"
                    "In <=8 words, describe what stands out about the current moment versus the alternate-time look. "
                    "If only Image 1 is available, briefly describe that scene. Avoid mentioning different cities or places."
                )
            content: list[dict[str, object]] = [{"type": "text", "text": prompt_text}]
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{curr_b64}"}})
            if fill_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fill_b64}"}})
            if prev_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prev_b64}"}})

            payload = {
                "model": gpt_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.2,
                "max_tokens": 16,
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            logging.debug("GPT payload preview (truncated): %s", str(payload)[:200])
            r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=10)
            if r.status_code == 200:
                out = r.json()
                first_choice = (out.get("choices") or [{}])[0]
                msg = first_choice.get("message")

                def _content_to_text(raw) -> str:
                    if isinstance(raw, str):
                        return raw.strip()
                    if isinstance(raw, list):
                        parts: list[str] = []
                        for entry in raw:
                            if isinstance(entry, dict):
                                if "text" in entry:
                                    parts.append(entry.get("text", ""))
                                elif entry.get("type") == "reasoning":
                                    parts.append(entry.get("reasoning", ""))
                            else:
                                parts.append(str(entry))
                        return " ".join(p.strip() for p in parts if p).strip()
                    if raw is None:
                        return ""
                    return str(raw).strip()

                raw_content = None
                if isinstance(msg, dict):
                    raw_content = msg.get("content")
                elif msg is not None:
                    raw_content = msg
                if not raw_content:
                    raw_content = first_choice.get("content")

                txt = _content_to_text(raw_content)
                logging.debug("GPT raw response: %s", txt[:160])

                if txt:
                    gpt_calls += 1
                    gpt_counter += 1
                    desc = txt.splitlines()[0].strip()
                    if len(desc) > 80:
                        desc = desc[:77] + "..."
                    label_text = f"ID{gpt_counter:04d}: {desc}"
                    if cache_key:
                        gpt_cache[cache_key] = label_text
                    return label_text
        except Exception as e:
            logging.warning("GPT label error: %s", e)
            return None
        return None

    print("🎯 make_frame fonksiyonu tanımlanıyor...")

    def make_frame(t: float):
        nonlocal prev_gray, prev_frame_color, onset_idx, active, frame_counter
        frame_counter += 1
        if prev_gray is None:
            print(f"🎬 İLK FRAME İŞLENİYOR (t={t:.2f}s)")
            print("  ↳ Kaynak frame alınıyor...")
        elif frame_counter % 50 == 0:
            print(f"📹 Frame {frame_counter} işleniyor (t={t:.2f}s)")
        frame = clip.get_frame(t).copy()
        if prev_gray is None:
            print(f"  ↳ Frame alındı: {frame.shape}")
        overlay_frame = None
        if overlay_clip is not None:
            if prev_gray is None:
                print("  ↳ Overlay frame alınıyor...")
            try:
                tt = (t % overlay_duration) if (overlay_duration and overlay_duration > 0) else t
                overlay_frame = overlay_clip.get_frame(tt)
            except Exception as e:
                logging.warning(f"Overlay frame fetch failed; using invert. Reason: {e}")
                overlay_frame = None
        if prev_gray is None:
            print("  ↳ Grayscale'e çeviriliyor...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if prev_gray is None:
            print(f"  ↳ Gray hazır: {w}x{h}")

        # Fast-path: single dynamic rectangle only
        if single_box_mode:
            if prev_gray is None:
                print("  ↳ Single box mode aktif")
            nonlocal sb_pos, sb_vel, sb_phase
            if sb_pos is None:
                sb_pos = np.array([w / 2.0, h / 2.0], dtype=np.float32)
                ang = (random.random() * 2.0 - 1.0) * np.pi
                sb_vel = np.array([
                    np.cos(ang) * max(0.5, single_box_speed),
                    np.sin(ang) * max(0.5, single_box_speed),
                ], dtype=np.float32)
            # Lissajous-like wobble on velocity
            phase_step = float(max(0.0, single_box_pulse_speed))
            if phase_step > 0:
                sb_phase += phase_step
            wob = np.array([
                np.sin(sb_phase * 1.7) * 0.8,
                np.cos(sb_phase * 1.3) * 0.8,
            ], dtype=np.float32)
            vel = sb_vel + wob
            pulse_amp = float(max(0.0, single_box_pulse_amp))
            pulse_term = np.sin(sb_phase * 2.1) if pulse_amp > 0 else 0.0
            size = max(8, int(single_box_size * (1.0 + pulse_amp * pulse_term)))
            # Integrate and bounce on edges (keep box fully in frame)
            sb_pos += vel
            half = size // 2
            if sb_pos[0] - half < 0 or sb_pos[0] + half >= w:
                sb_vel[0] *= -1
                sb_pos[0] = np.clip(sb_pos[0], half, w - 1 - half)
            if sb_pos[1] - half < 0 or sb_pos[1] + half >= h:
                sb_vel[1] *= -1
                sb_pos[1] = np.clip(sb_pos[1], half, h - 1 - half)

            cx, cy = float(sb_pos[0]), float(sb_pos[1])
            tl = (max(0, int(cx - half)), max(0, int(cy - half)))
            br = (min(w - 1, int(cx + half)), min(h - 1, int(cy + half)))
            roi = frame[tl[1]:br[1], tl[0]:br[0]]
            if roi.size:
                if overlay_frame is not None:
                    of = overlay_frame
                    if of.dtype != np.uint8:
                        of = np.clip(of, 0, 255).astype(np.uint8)
                    if of.ndim == 2:
                        of = cv2.cvtColor(of, cv2.COLOR_GRAY2BGR)
                    elif of.shape[2] == 4:
                        of = of[:, :, :3]
                    if of.shape[0] != h or of.shape[1] != w:
                        of = cv2.resize(of, (w, h))
                    frame[tl[1]:br[1], tl[0]:br[0]] = of[tl[1]:br[1], tl[0]:br[0]]
                else:
                    frame[tl[1]:br[1], tl[0]:br[0]] = 255 - roi
            cv2.rectangle(frame, tl, br, (255, 255, 255), max(1, int(single_box_thickness)))
            prev_gray = gray
            prev_frame_color = frame.copy()
            return frame

        if prev_gray is not None and active:
            if frame_counter == 2:
                print("  ↳ Optical flow tracking başlıyor...")
            prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            if frame_counter == 2:
                print(f"  ↳ Optical flow tamamlandı: {len(active)} nokta")
            new_active: list[TrackedPoint] = []
            for tp, new_pt, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
                if not ok:
                    continue
                x, y = new_pt
                if 0 <= x < w and 0 <= y < h and tp.life > 0:
                    if 0.0 < pos_smooth < 1.0:
                        tp.pos = tp.pos * pos_smooth + new_pt * (1.0 - pos_smooth)
                    else:
                        tp.pos = new_pt
                    tp.life -= 1
                    if jitter_px > 0:
                        tp.pos += np.random.normal(0, jitter_px, size=2)
                        tp.pos[0] = np.clip(tp.pos[0], 0, w - 1)
                        tp.pos[1] = np.clip(tp.pos[1], 0, h - 1)
                    new_active.append(tp)
            active = new_active

        while onset_idx < len(onset_times) and t >= onset_times[onset_idx]:
            if prev_gray is None:
                print(f"  ↳ Onset {onset_idx} spawn başlıyor...")
            # Build a candidate list from ORB and (optionally) SimpleBlobDetector
            candidate_pts: list[tuple[float, float]] = []
            try:
                if prev_gray is None:
                    print("  ↳ ORB feature detection çalışıyor...")
                kps_orb = orb.detect(gray, None)
                if prev_gray is None:
                    print(f"  ↳ ORB tamamlandı: {len(kps_orb)} feature bulundu")
                    print("  ↳ ORB sıralaması yapılıyor...")
                kps_orb = sorted(kps_orb, key=lambda k: k.response, reverse=True)
                if prev_gray is None:
                    print("  ↳ Candidate points ekleniyor...")
                candidate_pts.extend([kp.pt for kp in kps_orb])
                if prev_gray is None:
                    print(f"  ↳ Toplam {len(candidate_pts)} candidate point")
            except Exception as e:
                if prev_gray is None:
                    print(f"  ↳ ORB hatası: {e}")
                pass
            if blob_detector is not None:
                if prev_gray is None:
                    print("  ↳ Blob detector kontrol ediliyor...")
                try:
                    kps_blob = blob_detector.detect(gray)
                    kps_blob = sorted(kps_blob, key=lambda k: getattr(k, 'size', 0), reverse=True)
                    candidate_pts.extend([kp.pt for kp in kps_blob])
                except Exception as e:
                    logging.debug(f"Blob detect failed: {e}")
            # Motion-biased candidates from frame differencing
            if motion_spawn_bias and prev_gray is not None:
                try:
                    diff = cv2.absdiff(gray, prev_gray)
                    q = motion_percentile if 0.0 <= motion_percentile <= 1.0 else 0.9
                    thr = float(np.quantile(diff, q))
                    ys, xs = np.where(diff >= thr)
                    if len(xs) > 0:
                        take = min(int(motion_points), len(xs))
                        sel = np.random.choice(len(xs), size=take, replace=False)
                        for idx in sel:
                            candidate_pts.append((float(xs[idx]), float(ys[idx])))
                except Exception as e:
                    logging.debug(f"motion candidates failed: {e}")
            # Spawn closer to the maximum requested to increase box count
            target_spawn = random.randint(max(1, pts_per_beat // 2), pts_per_beat)
            if prev_gray is None:
                print(f"  ↳ Spawn loop başlıyor (target={target_spawn}, candidates={len(candidate_pts)})...")
            spawned = 0
            kp_i = 0
            new_coords: list[tuple[float, float]] = []
            loop_iter = 0
            max_iterations = min(500, len(candidate_pts) + target_spawn * 3)
            while spawned < target_spawn and loop_iter < max_iterations:
                loop_iter += 1
                if prev_gray is None and loop_iter % 100 == 0:
                    print(f"  ↳ Loop iteration {loop_iter}, spawned={spawned}/{target_spawn}")
                if len(active) >= max_active:
                    break
                if kp_i >= len(candidate_pts) and loop_iter > target_spawn * 2:
                    break
                use_random = (random.random() < random_spawn_frac) or (kp_i >= len(candidate_pts))
                if use_random:
                    x = random.uniform(0, w)
                    y = random.uniform(0, h)
                else:
                    x, y = candidate_pts[kp_i]
                    kp_i += 1
                    # Nudge around feature point to avoid exact clustering
                    x += random.uniform(-8, 8)
                    y += random.uniform(-8, 8)
                # Enforce a minimum distance from existing and newly spawned points
                size = random.randint(min_size, max_size)
                if avoid_overlap:
                    too_close = False
                    for tp in active:
                        dyn = 0.5 * (tp.size + size) + min_gap_px
                        if np.linalg.norm(tp.pos - (x, y)) < max(min_spawn_dist, dyn):
                            too_close = True
                            break
                    if too_close:
                        continue
                    for c in new_coords:
                        if np.linalg.norm(np.array(c) - (x, y)) < (0.5 * size + min_gap_px):
                            too_close = True
                            break
                    if too_close:
                        continue
                else:
                    if any(np.linalg.norm(tp.pos - (x, y)) < min_spawn_dist for tp in active):
                        continue
                    if any(np.linalg.norm(np.array(c) - (x, y)) < min_spawn_dist for c in new_coords):
                        continue
                label = f"({random.randint(0, w - 1)}, {random.randint(0, h - 1)})"
                cache_key = None
                if use_gpt_labels and (random.random() < max(0.0, min(1.0, gpt_label_rate))) and gpt_calls < gpt_label_max:
                    half = size // 2
                    tlx = max(0, int(x - half)); tly = max(0, int(y - half))
                    brx = min(w - 1, int(x + half)); bry = min(h - 1, int(y + half))
                    roi2 = frame[tly:bry, tlx:brx]
                    prev_roi = None
                    if prev_frame_color is not None:
                        prev_roi = prev_frame_color[tly:bry, tlx:brx]
                        if prev_roi is not None and prev_roi.size == 0:
                            prev_roi = None
                    fill_roi = None
                    if overlay_frame is not None:
                        fill_roi = overlay_frame[tly:bry, tlx:brx]
                        if fill_roi is not None and fill_roi.size == 0:
                            fill_roi = None
                    if roi2.size:
                        cache_key = _roi_signature(roi2)
                        got = _label_roi_with_gpt(roi2, prev_roi=prev_roi, fill_roi=fill_roi, cache_key=cache_key)
                        if got:
                            label = got
                if len(active) < max_active:
                    active.append(TrackedPoint((x, y), life_frames, size, label))
                new_coords.append((x, y))
                spawned += 1
            if prev_gray is None:
                print(f"  ↳ Spawn loop tamamlandı: {spawned} nokta oluşturuldu ({loop_iter} iterasyon)")
            onset_idx += 1

        if ambient_rate > 0 and len(active) < max_active:
            noise_n = np.random.poisson(ambient_rate / fps)
            for _ in range(noise_n):
                if len(active) >= max_active:
                    break
                x = random.uniform(0, w)
                y = random.uniform(0, h)
                # Ambient spawns also use clearly varying sizes
                size = random.randint(min_size, max_size)
                label = f"({random.randint(0, w - 1)}, {random.randint(0, h - 1)})"
                if use_gpt_labels and (random.random() < max(0.0, min(1.0, gpt_label_rate))) and gpt_calls < gpt_label_max:
                    half = size // 2
                    tlx = max(0, int(x - half)); tly = max(0, int(y - half))
                    brx = min(w - 1, int(x + half)); bry = min(h - 1, int(y + half))
                    roi2 = frame[tly:bry, tlx:brx]
                    prev_roi = None
                    if prev_frame_color is not None:
                        prev_roi = prev_frame_color[tly:bry, tlx:brx]
                        if prev_roi is not None and prev_roi.size == 0:
                            prev_roi = None
                    fill_roi = None
                    if overlay_frame is not None:
                        fill_roi = overlay_frame[tly:bry, tlx:brx]
                        if fill_roi is not None and fill_roi.size == 0:
                            fill_roi = None
                    if roi2.size:
                        cache_key = _roi_signature(roi2)
                        got = _label_roi_with_gpt(roi2, prev_roi=prev_roi, fill_roi=fill_roi, cache_key=cache_key)
                        if got:
                            label = got
                if avoid_overlap:
                    ok = True
                    for tp in active:
                        dyn = 0.5 * (tp.size + size) + min_gap_px
                        if np.linalg.norm(tp.pos - (x, y)) < dyn:
                            ok = False
                            break
                    if ok and len(active) < max_active:
                        active.append(TrackedPoint((x, y), life_frames, size, label))
                else:
                    if len(active) < max_active:
                        active.append(TrackedPoint((x, y), life_frames, size, label))

        # Hard cap pruning if somehow exceeded
        if len(active) > max_active:
            active = sorted(active, key=lambda p: p.life, reverse=True)[:max_active]

        # Optional: apply a light separation pass to reduce overlaps after motion
        if avoid_overlap and len(active) > 1:
            idxs = list(range(len(active)))
            # Sort by x to limit comparisons
            idxs.sort(key=lambda i: active[i].pos[0])
            for ii in range(len(idxs)):
                i = idxs[ii]
                pi = active[i].pos.copy()
                si = active[i].size
                for jj in range(ii + 1, len(idxs)):
                    j = idxs[jj]
                    pj = active[j].pos.copy()
                    sj = active[j].size
                    # Early break if too far on x-axis
                    if pj[0] - pi[0] > (0.5 * (si + sj) + min_gap_px):
                        break
                    need = 0.5 * (si + sj) + min_gap_px
                    dvec = pj - pi
                    d = float(np.linalg.norm(dvec))
                    if d < need and d > 1e-3:
                        push = (need - d) * 0.5
                        dir_vec = dvec / d
                        # Move apart and clip to frame
                        new_pi = np.array([
                            np.clip(pi[0] - dir_vec[0] * push, 0, w - 1),
                            np.clip(pi[1] - dir_vec[1] * push, 0, h - 1),
                        ], dtype=np.float32)
                        new_pj = np.array([
                            np.clip(pj[0] + dir_vec[0] * push, 0, w - 1),
                            np.clip(pj[1] + dir_vec[1] * push, 0, h - 1),
                        ], dtype=np.float32)
                        active[i].pos = new_pi
                        active[j].pos = new_pj

        coords = [tp.pos for tp in active]
        # Helper to draw a simple quadratic Bezier curve between two points
        def _draw_curvy_link(img, p0, p1, color=(200, 200, 255), width=1):
            p0 = p0.astype(float)
            p1 = p1.astype(float)
            mid = (p0 + p1) / 2.0
            v = p1 - p0
            if np.linalg.norm(v) < 1e-3:
                cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), color, width)
                return
            # Perpendicular vector for curvature
            perp = np.array([-v[1], v[0]])
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 1e-6:
                perp /= perp_norm
            # Curvature magnitude relative to distance
            dist = np.linalg.norm(v)
            mag = random.uniform(0.1, 0.35) * dist
            ctrl = mid + perp * mag * (1 if random.random() < 0.5 else -1)

            # Sample quadratic Bezier points
            ts = np.linspace(0, 1, 24)
            pts = ((1 - ts)[:, None] ** 2) * p0 + 2 * (1 - ts)[:, None] * ts[:, None] * ctrl + (ts[:, None] ** 2) * p1
            pts = np.clip(pts, [0, 0], [w - 1, h - 1]).astype(int)
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=width)

        for i, p in enumerate(coords):
            dists = [(j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i]
            dists.sort(key=lambda x: x[1])
            for j, _ in dists[:neighbor_links]:
                if random.random() < 0.50:
                    _draw_curvy_link(frame, p, coords[j], (200, 200, 255), 1)
                    continue
                cv2.line(frame, tuple(p.astype(int)), tuple(coords[j].astype(int)), (200, 200, 255), 1)

        for tp in active:
            x, y = tp.pos
            if use_dynamic_growth:
                if hasattr(tp, 'phase'):
                    tp.phase += float(growth_speed)
                else:
                    tp.phase = float(np.random.uniform(0, 2 * np.pi))
                s = int(max(4, tp.size * (1.0 + float(growth_amp) * np.sin(tp.phase))))
            else:
                s = int(max(4, tp.size))
            tl = (max(0, int(x - s // 2)), max(0, int(y - s // 2)))
            br = (min(w - 1, int(x + s // 2)), min(h - 1, int(y + s // 2)))
            roi = frame[tl[1]:br[1], tl[0]:br[0]]
            if roi.size:
                if overlay_frame is not None:
                    try:
                        of = overlay_frame
                        if of.dtype != np.uint8:
                            of = np.clip(of, 0, 255).astype(np.uint8)
                        if of.ndim == 2:
                            of = cv2.cvtColor(of, cv2.COLOR_GRAY2BGR)
                        elif of.shape[2] == 4:
                            of = of[:, :, :3]
                        sub = of[tl[1]:br[1], tl[0]:br[0]]
                        if sub.shape != roi.shape:
                            resized = cv2.resize(of, (frame_w, frame_h))
                            sub = resized[tl[1]:br[1], tl[0]:br[0]]
                        frame[tl[1]:br[1], tl[0]:br[0]] = sub
                    except Exception as e:
                        logging.warning(f"Overlay paste failed; using invert. Reason: {e}")
                        frame[tl[1]:br[1], tl[0]:br[0]] = 255 - roi
                else:
                    frame[tl[1]:br[1], tl[0]:br[0]] = 255 - roi
            cv2.rectangle(frame, tl, br, (200, 200, 255), 1)

            # Draw a small coordinate-like random label near the rectangle with semi-transparent bg
            if tp.label:
                text = tp.label
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.45
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
                tx = tl[0]
                ty = tl[1] - 5
                if ty - th - baseline < 0:
                    ty = br[1] + th + baseline + 5
                # Background for readability with transparency
                bg_tl = (max(0, tx - 2), max(0, ty - th - baseline - 2))
                bg_br = (min(w - 1, tx + tw + 2), min(h - 1, ty + 2))
                x0, y0 = bg_tl
                x1, y1 = bg_br
                roi = frame[y0:y1, x0:x1]
                if roi.size:
                    overlay = roi.copy()
                    cv2.rectangle(overlay, (0, 0), (max(0, overlay.shape[1] - 1), max(0, overlay.shape[0] - 1)), (0, 0, 0), -1)
                    alpha = 0.45
                    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
                cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        prev_gray = gray
        prev_frame_color = frame.copy()
        if frame_counter == 1:
            print("  ↳ İlk frame tamamlandı, döndürülüyor")
        return frame

    print("✅ make_frame fonksiyonu tanımlandı")
    print(f"🎞️ VideoClip oluşturuluyor (duration={clip.duration}s)...")
    out_clip = mpy.VideoClip(make_frame, duration=clip.duration)
    print("✅ VideoClip oluşturuldu")
    print("🔊 Audio ekleniyor...")
    out_clip = _with_audio(out_clip, clip.audio)
    print("⏱️ FPS ayarlanıyor...")
    out_clip = _with_fps(out_clip, fps)
    print("📐 Boyut ayarlanıyor...")
    out_clip = _with_resize(out_clip, clip.size)
    print(f"💾 Video dosyası yazılıyor: {video_out}")
    print(f"📊 Frame sayısı: {int(clip.duration * fps)}")
    out_clip.write_videofile(str(video_out), codec="libx264", audio_codec="aac")
    print("✅ Render tamamlandı!")


def funny_loading_bar():
    import time
    import sys

    stages = [
        "Initializing...",
        "Summoning pixels...",
        "Whispering to the GPU...",
        "Feeding bits...",
        "Untangling frames...",
        "Convincing codecs...",
        "Drawing boxes like Picasso...",
        "Launching frame...",
        "Synchronizing beats...",
        "Finishing touches...",
    ]

    for i in range(0, 101):
        time.sleep(0.03 if i < 90 else 0.01)
        bar = ('█' * (i // 2)).ljust(50)
        sys.stdout.write(
            f"\r[{bar}] {i}%  {stages[i // 10 % len(stages)] if i % 10 == 0 else ''}   "
        )
        sys.stdout.flush()
    print("\nRender prep complete!")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = input("Enter the input video file path: ").strip()
    if not os.path.isfile(input_path):
        print("❌ Error: Input file does not exist.")
        return

    default_output = Path.home() / "Downloads" / "output.mp4"
    prompt = f"Enter output video file path (leave blank to save in Downloads as '{default_output.name}'): "
    output_path_str = input(prompt).strip()
    output_path = Path(output_path_str) if output_path_str else default_output

    fill_path_input = input("Optional: enter a FILL video path for square interiors (leave blank to use invert effect): ").strip()
    # Sanitize quotes and expand ~
    fill_path_clean = fill_path_input.strip().strip('"\'')
    fill_path_clean = os.path.expanduser(fill_path_clean)
    fill_path = Path(fill_path_clean) if fill_path_clean else None
    if fill_path is not None and not fill_path.exists():
        print(f"Fill video not found at: {fill_path}")
        fill_path = None


    print("\nStarting rendering process...\n")
    funny_loading_bar()

    render_tracked_effect(
        video_in=Path(input_path),
        video_out=output_path,
        fps=30.0,
        # Denser + connected look with arcs and larger squares
        pts_per_beat=35,
        ambient_rate=8.0,
        jitter_px=0.25,
        life_frames=24,
        # Larger sizes
        min_size=100,
        max_size=400,
        neighbor_links=4,
        orb_fast_threshold=20,
        bell_width=4.0,
        seed=None,
        fill_video_path=fill_path,
        random_spawn_frac=0.45,
        min_spawn_dist=24.0,
        max_active=150,
        avoid_overlap=True,
        min_gap_px=8.0,
        gpt_api_key=None,
    )


if __name__ == "__main__":
    main()
