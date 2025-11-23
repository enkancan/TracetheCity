from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
try:
    import moviepy.editor as mpy  # MoviePy v1 API
    from moviepy.editor import concatenate_videoclips as CONCAT
    MPY_V2 = False
except ImportError:
    # MoviePy v2 API fallback
    from moviepy import ImageSequenceClip, CompositeVideoClip, ImageClip, concatenate_videoclips as CONCAT
    class _MPY:
        ImageSequenceClip = ImageSequenceClip
        CompositeVideoClip = CompositeVideoClip
        ImageClip = ImageClip
    mpy = _MPY()
    MPY_V2 = True

# Default images directory for convenience
DEFAULT_IMG_DIR = Path(r"C:\Users\user\OneDrive\Desktop\Codes\images_galataport")


@dataclass
class ImgItem:
    path: Path
    id1: str
    angle: float
    angle_key: float
    date: datetime
    year: int
    lon: float | None = None
    lat: float | None = None


FNAME_RE = re.compile(r"^([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([0-9]{4}-[0-9]{2}-[0-9]{2})\.(jpg|jpeg|png)$",
                      re.IGNORECASE)


def parse_image_filename(p: Path, angle_round: int = 2, angle_field_index: int = 6) -> Optional[ImgItem]:
    m = FNAME_RE.match(p.name)
    if not m:
        return None
    id1 = m.group(1)
    # angle_field_index can be 4 or 6 depending on schema; default 6
    try:
        angle_str = m.group(angle_field_index)
        angle = float(angle_str)
    except Exception:
        return None
    # lon/lat if present (groups 4/5 as in the regex order)
    lon = None
    lat = None
    try:
        lon = float(m.group(4))
        lat = float(m.group(5))
    except Exception:
        pass
    date_str = m.group(8)
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None
    return ImgItem(
        path=p,
        id1=id1,
        angle=angle,
        angle_key=round(angle, angle_round),
        date=dt,
        year=dt.year,
        lon=lon,
        lat=lat,
    )


def build_groups(img_dir: Path, angle_round: int = 2, angle_field_index: int = 6) -> Dict[Tuple[str, float], List[ImgItem]]:
    groups: Dict[Tuple[str, float], List[ImgItem]] = {}
    for entry in img_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        item = parse_image_filename(entry, angle_round=angle_round, angle_field_index=angle_field_index)
        if not item:
            continue
        key = (item.id1, item.angle_key)
        groups.setdefault(key, []).append(item)
    for items in groups.values():
        items.sort(key=lambda it: it.date)
    return groups


# ---------------- Path-mode helpers (no shapely dependency) ----------------
def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    # Approx conversion at given latitude
    lat_rad = abs(lat_deg) * 3.141592653589793 / 180.0
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    return m_per_deg_lon, m_per_deg_lat


def _project_dist_m(lon: float, lat: float, s_lon: float, s_lat: float, e_lon: float, e_lat: float) -> tuple[float, float]:
    # Equirectangular projection to approximate projection and distance
    m_per_deg_lon, m_per_deg_lat = _meters_per_degree((s_lat + e_lat) / 2.0)
    x0 = (lon - s_lon) * m_per_deg_lon
    y0 = (lat - s_lat) * m_per_deg_lat
    vx = (e_lon - s_lon) * m_per_deg_lon
    vy = (e_lat - s_lat) * m_per_deg_lat
    v2 = vx * vx + vy * vy
    if v2 <= 1e-9:
        return 0.0, (x0 * x0 + y0 * y0) ** 0.5
    t = max(0.0, min(1.0, (x0 * vx + y0 * vy) / v2))
    px = t * vx
    py = t * vy
    dx = x0 - px
    dy = y0 - py
    dist = (dx * dx + dy * dy) ** 0.5
    return t, dist


def collect_path_mode(img_dir: Path, fg_year: int, bg_year: int, start_lon: float, start_lat: float, end_lon: float, end_lat: float, radius_m: float, angle_round: int, angle_field_index: int) -> tuple[list[ImgItem], list[ImgItem]]:
    # Parse all items in dir
    all_items: list[ImgItem] = []
    for entry in img_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        it = parse_image_filename(entry, angle_round=angle_round, angle_field_index=angle_field_index)
        if not it or it.lon is None or it.lat is None:
            continue
        t, d = _project_dist_m(it.lon, it.lat, start_lon, start_lat, end_lon, end_lat)
        if d <= radius_m:
            # store projection along the line in a temporary attribute via tuple
            all_items.append(ImgItem(path=it.path, id1=it.id1, angle=it.angle, angle_key=it.angle_key, date=it.date, year=it.year, lon=it.lon, lat=it.lat))
            all_items[-1].angle_key = t  # reuse angle_key temporarily to hold t

    # Separate by year and sort by projection (t) then date
    fg_items = [it for it in all_items if it.year == fg_year]
    bg_items = [it for it in all_items if it.year == bg_year]
    fg_items.sort(key=lambda it: (it.angle_key, it.date))
    bg_items.sort(key=lambda it: (it.angle_key, it.date))
    return fg_items, bg_items


def align_start_by_common_id(fg_items: list[ImgItem], bg_items: list[ImgItem]) -> tuple[list[ImgItem], list[ImgItem]]:
    fg_ids = [it.id1 for it in fg_items]
    bg_ids = [it.id1 for it in bg_items]
    common = set(fg_ids) & set(bg_ids)
    if not common:
        return fg_items, bg_items
    # pick first common id from fg order
    first_common = next((i for i in fg_ids if i in common), None)
    if first_common is None:
        return fg_items, bg_items
    # trim lists to start at the first occurrence of this id
    def trim(lst: list[ImgItem], target_id: str) -> list[ImgItem]:
        for idx, it in enumerate(lst):
            if it.id1 == target_id:
                return lst[idx:]
        return lst
    fg_trim = trim(fg_items, first_common)
    bg_trim = trim(bg_items, first_common)
    # equalize lengths
    n = min(len(fg_trim), len(bg_trim))
    return fg_trim[:n], bg_trim[:n]


def pair_common_ids_in_path_order(fg_items: list[ImgItem], bg_items: list[ImgItem]) -> tuple[list[ImgItem], list[ImgItem]]:
    # Build id -> (item, t_order) maps using the earliest-along-path item per id
    def build_map(items: list[ImgItem]) -> dict[str, tuple[ImgItem, float]]:
        m: dict[str, tuple[ImgItem, float]] = {}
        for it in items:
            # angle_key was reused to store along-line projection t in collect_path_mode
            t = float(getattr(it, 'angle_key', 0.0))
            if it.id1 not in m or t < m[it.id1][1]:
                m[it.id1] = (it, t)
        return m

    fg_map = build_map(fg_items)
    bg_map = build_map(bg_items)
    common_ids = set(fg_map.keys()) & set(bg_map.keys())
    if not common_ids:
        return fg_items, bg_items

    pairs: list[tuple[float, ImgItem, ImgItem]] = []
    for idv in common_ids:
        f_item, tf = fg_map[idv]
        b_item, tb = bg_map[idv]
        t_order = (tf + tb) / 2.0
        pairs.append((t_order, f_item, b_item))
    pairs.sort(key=lambda x: x[0])

    fg_sel = [f for _, f, _ in pairs]
    bg_sel = [b for _, _, b in pairs]
    return fg_sel, bg_sel


def collect_ids_only(img_dir: Path, fg_year: int, bg_year: int, angle_round: int, angle_field_index: int, prefer: str = "earliest") -> tuple[list[ImgItem], list[ImgItem]]:
    """Build FG/BG lists by ID only (ignore angle), using one image per year per ID.
    prefer can be 'earliest' or 'latest'. Order is taken from background year by date."""
    per_id: dict[str, dict[int, list[ImgItem]]] = {}
    for entry in img_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        it = parse_image_filename(entry, angle_round=angle_round, angle_field_index=angle_field_index)
        if not it:
            continue
        if it.year not in (fg_year, bg_year):
            continue
        per_id.setdefault(it.id1, {}).setdefault(it.year, []).append(it)

    fg_map: dict[str, ImgItem] = {}
    bg_map: dict[str, ImgItem] = {}
    for idv, year_map in per_id.items():
        if bg_year in year_map:
            year_map[bg_year].sort(key=lambda x: x.date)
            bg_map[idv] = year_map[bg_year][0 if prefer == "earliest" else -1]
        if fg_year in year_map:
            year_map[fg_year].sort(key=lambda x: x.date)
            fg_map[idv] = year_map[fg_year][0 if prefer == "earliest" else -1]

    common_ids = sorted([i for i in bg_map.keys() if i in fg_map.keys()], key=lambda k: bg_map[k].date)
    fg_list = [fg_map[i] for i in common_ids]
    bg_list = [bg_map[i] for i in common_ids]
    return fg_list, bg_list


def choose_group(groups: Dict[Tuple[str, float], List[ImgItem]], fg_year: int, bg_year: int) -> Optional[Tuple[Tuple[str, float], List[ImgItem]]]:
    # Filter groups that have both years present
    eligible: List[Tuple[Tuple[str, float], List[ImgItem]]] = []
    for key, items in groups.items():
        years = {it.year for it in items}
        if fg_year in years and bg_year in years:
            eligible.append((key, items))
    if not eligible:
        print("No groups contain both years.")
        return None

    # Display a compact list to choose from
    print("Found groups with both years (index: id1 | angle | counts fg/bg):")
    view = []
    for key, items in eligible:
        id1, angle_key = key
        fg_n = len([it for it in items if it.year == fg_year])
        bg_n = len([it for it in items if it.year == bg_year])
        view.append((key, items, fg_n, bg_n))
    # Sort by min(fg,bg) desc for convenience
    view.sort(key=lambda row: min(row[2], row[3]), reverse=True)
    for j, (key, items, fg_n, bg_n) in enumerate(view[:30]):
        id1, angle_key = key
        print(f"  {j:2d}: {id1} | angle~{angle_key} | {fg_n}/{bg_n}")

    sel = input("Select index to build video (blank = best match): ").strip()
    if sel.isdigit():
        j = int(sel)
    else:
        j = 0  # best match
    if not (0 <= j < len(view)):
        print("Invalid selection.")
        return None
    key, items, _, _ = view[j]
    return key, items


def _build_seq_clip(paths: List[Path], fps: int, hold_sec: float | None):
    # Build a clip from images: either as a slideshow with per-image duration
    # or as an image sequence at given fps
    if hold_sec and hold_sec > 0:
        clips = []
        for p in paths:
            ic = getattr(mpy, 'ImageClip', None)
            if ic is None:
                # Fallback: cannot build slideshow without ImageClip
                return mpy.ImageSequenceClip([str(pp) for pp in paths], fps=fps)
            c = ic(str(p))
            if hasattr(c, 'set_duration'):
                c = c.set_duration(hold_sec)
            else:
                c = c.with_duration(hold_sec)
            clips.append(c)
        # Prefer global CONCAT handle to avoid bound-method issues on shims
        concat = CONCAT if 'CONCAT' in globals() else None
        if concat is None:
            # Fallback to sequence if concatenate not available
            return mpy.ImageSequenceClip([str(pp) for pp in paths], fps=fps)
        # MoviePy v1/v2 compatibility: some versions don't accept method kw
        try:
            return concat(clips, method="chain")
        except TypeError:
            return concat(clips)
    else:
        return mpy.ImageSequenceClip([str(p) for p in paths], fps=fps)


def _ensure_even_size(clip):
    bw, bh = getattr(clip, 'size', (None, None))
    if bw is None or bh is None:
        bw, bh = 1920, 1080
    even_w = int(bw) - (int(bw) % 2)
    even_h = int(bh) - (int(bh) % 2)
    target_size = (max(2, even_w), max(2, even_h))
    if hasattr(clip, 'resize'):
        return clip.resize(target_size)
    return clip.resized(target_size)


def _write_clip(clip, out_path: Path, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(codec="libx264", audio=False, fps=fps)
    try:
        clip.write_videofile(str(out_path), ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"], **extra)
    except TypeError:
        clip.write_videofile(str(out_path), **extra)


def make_separate_videos(items: List[ImgItem], fg_year: int, bg_year: int, out_fg: Path, out_bg: Path, fps: int = 2, opacity: float = 0.65, hold_sec: float | None = 2.0):
    fg_list = [it.path for it in items if it.year == fg_year]
    bg_list = [it.path for it in items if it.year == bg_year]
    if not fg_list or not bg_list:
        raise ValueError("Selected group does not contain both years.")

    # Align counts (trim to min to keep durations comparable)
    n = min(len(fg_list), len(bg_list))
    fg_list = fg_list[:n]
    bg_list = bg_list[:n]

    fg_clip = _build_seq_clip(fg_list, fps=fps, hold_sec=hold_sec)
    bg_clip = _build_seq_clip(bg_list, fps=fps, hold_sec=hold_sec)

    # Ensure both have even sizes; match foreground to background size
    bg_clip = _ensure_even_size(bg_clip)
    if hasattr(fg_clip, 'resize'):
        fg_clip = fg_clip.resize(bg_clip.size)
    else:
        fg_clip = fg_clip.resized(bg_clip.size)

    _write_clip(bg_clip, out_bg, fps=fps)
    _write_clip(fg_clip, out_fg, fps=fps)


def make_composite_video(items: List[ImgItem], fg_year: int, bg_year: int, out_path: Path, fps: int = 2, opacity: float = 0.65, hold_sec: float | None = 2.0):
    fg_list = [it.path for it in items if it.year == fg_year]
    bg_list = [it.path for it in items if it.year == bg_year]
    if not fg_list or not bg_list:
        raise ValueError("Selected group does not contain both years.")

    # Align by index in sorted order; trim to min length
    n = min(len(fg_list), len(bg_list))
    fg_list = fg_list[:n]
    bg_list = bg_list[:n]

    # Build background and foreground videos (slideshow-friendly)
    bg_clip = _build_seq_clip(bg_list, fps=fps, hold_sec=hold_sec)
    fg_clip = _build_seq_clip(fg_list, fps=fps, hold_sec=hold_sec)

    # Ensure dimensions are even for H.264 yuv420p compatibility
    bg_clip = _ensure_even_size(bg_clip)

    if hasattr(fg_clip, 'resize'):
        fg_clip = fg_clip.resize(target_size)
        if hasattr(fg_clip, 'set_opacity'):
            fg_clip = fg_clip.set_opacity(opacity)
        else:
            fg_clip = fg_clip.with_opacity(opacity)
    else:
        fg_clip = fg_clip.resized(target_size).with_opacity(opacity)

    comp = mpy.CompositeVideoClip([bg_clip, fg_clip])
    dur = min(bg_clip.duration, fg_clip.duration)
    if hasattr(comp, 'set_duration'):
        comp = comp.set_duration(dur)
    else:
        comp = comp.with_duration(dur)

    _write_clip(comp, out_path, fps=fps)


def main():
    print("Composite year layers (foreground vs background) from images.")
    img_dir_str = input(f"Enter images folder path (blank = {DEFAULT_IMG_DIR}): ").strip()
    img_dir_str = img_dir_str.strip().strip('"\'')
    if not img_dir_str:
        img_dir = DEFAULT_IMG_DIR
    else:
        img_dir = Path(os.path.expanduser(img_dir_str))
    if not img_dir.exists() or not img_dir.is_dir():
        print("Images folder not found or not a directory.")
        return

    fg_year_str = input("Foreground year (default 2024): ").strip()
    bg_year_str = input("Background year (default 2018): ").strip()
    try:
        fg_year = int(fg_year_str) if fg_year_str else 2024
        bg_year = int(bg_year_str) if bg_year_str else 2018
    except ValueError:
        print("Invalid year input.")
        return

    angle_round_str = input("Angle rounding decimals (default 2): ").strip()
    try:
        angle_round = int(angle_round_str) if angle_round_str else 2
    except ValueError:
        angle_round = 2

    angle_field_str = input("Angle field index (4 or 6, default 6): ").strip()
    try:
        angle_field_index = int(angle_field_str) if angle_field_str else 6
    except ValueError:
        angle_field_index = 6
    if angle_field_index not in (4, 6):
        angle_field_index = 6

    path_mode = input("Use path mode (filter by line and order along it)? (y/N): ").strip().lower().startswith('y')
    ids_only = False
    if not path_mode:
        ids_only = input("IDs-only pairing (ignore angle)? (y/N): ").strip().lower().startswith('y')

    if not path_mode and not ids_only:
        groups = build_groups(img_dir, angle_round=angle_round, angle_field_index=angle_field_index)
        chosen = choose_group(groups, fg_year=fg_year, bg_year=bg_year)
        if not chosen:
            return
        key, items = chosen
        id1, angle_key = key
        print(f"Building composite for id={id1}, angle~{angle_key}")
    
    fps_str = input("FPS (frames per second, default 4): ").strip()
    opacity_str = input("Foreground opacity 0..1 (default 0.65): ").strip()
    try:
        fps = int(fps_str) if fps_str else 4
    except ValueError:
        fps = 2
    try:
        opacity = float(opacity_str) if opacity_str else 0.65
        opacity = max(0.0, min(1.0, opacity))
    except ValueError:
        opacity = 0.65

    mode = input("Output mode: separate (s) or composite (c)? [s]: ").strip().lower()
    sep_mode = (mode != 'c')

    hold_str = input("Seconds per image (blank=1.0, 0 = use FPS only): ").strip()
    try:
        hold_sec = float(hold_str) if hold_str else 1.0
    except ValueError:
        hold_sec = 1.0

    if path_mode:
        # Ask for line and radius (defaults from your notebook)
        def _parse_float_pair(s: str, default: tuple[float,float]) -> tuple[float,float]:
            try:
                parts = [p.strip() for p in s.split(',')]
                if len(parts) == 2:
                    return float(parts[0]), float(parts[1])
            except Exception:
                pass
            return default

        s_default = (28.981057, 41.026975)
        e_default = (28.987375, 41.029066)
        r_default = 25.0
        s_in = input(f"Start lon,lat (blank={s_default[0]},{s_default[1]}): ").strip()
        e_in = input(f"End lon,lat (blank={e_default[0]},{e_default[1]}): ").strip()
        r_in = input(f"Radius meters (blank={r_default}): ").strip()
        s_lon, s_lat = _parse_float_pair(s_in, s_default)
        e_lon, e_lat = _parse_float_pair(e_in, e_default)
        try:
            radius_m = float(r_in) if r_in else r_default
        except ValueError:
            radius_m = r_default

        fg_items, bg_items = collect_path_mode(img_dir, fg_year, bg_year, s_lon, s_lat, e_lon, e_lat, radius_m, angle_round, angle_field_index)
        if not fg_items or not bg_items:
            print("No images found near the path for one or both years.")
            return
        # Build matched sequences using all ids common to both years, ordered along path
        fg_items, bg_items = pair_common_ids_in_path_order(fg_items, bg_items)
        if len(fg_items) < 2 or len(bg_items) < 2:
            print("Too few images after alignment. Try increasing radius.")
            return
        # Build paths lists
        fg_paths = [it.path for it in fg_items]
        bg_paths = [it.path for it in bg_items]
        # Write separate or composite
        if sep_mode:
            out_bg = img_dir / f"bg_path_{bg_year}.mp4"
            out_fg = img_dir / f"fg_path_{fg_year}.mp4"
            out_bg_str = input(f"Background video path (blank='{out_bg.name}'): ").strip()
            out_fg_str = input(f"Foreground video path (blank='{out_fg.name}'): ").strip()
            out_bg = Path(out_bg_str) if out_bg_str else out_bg
            out_fg = Path(out_fg_str) if out_fg_str else out_fg
            # Build clips directly
            bg_clip = _build_seq_clip(bg_paths, fps=fps, hold_sec=hold_sec)
            fg_clip = _build_seq_clip(fg_paths, fps=fps, hold_sec=hold_sec)
            bg_clip = _ensure_even_size(bg_clip)
            if hasattr(fg_clip, 'resize'):
                fg_clip = fg_clip.resize(bg_clip.size)
            else:
                fg_clip = fg_clip.resized(bg_clip.size)
            _write_clip(bg_clip, out_bg, fps=fps)
            _write_clip(fg_clip, out_fg, fps=fps)
            print(f"Done. Saved background: {out_bg}")
            print(f"Done. Saved foreground: {out_fg}")
        else:
            default_out = img_dir / f"composite_path_{bg_year}_bg_{fg_year}_fg.mp4"
            out_str = input(f"Output composite path (blank='{default_out.name}'): ").strip()
            out_path = Path(out_str) if out_str else default_out
            bg_clip = _build_seq_clip(bg_paths, fps=fps, hold_sec=hold_sec)
            fg_clip = _build_seq_clip(fg_paths, fps=fps, hold_sec=hold_sec)
            bg_clip = _ensure_even_size(bg_clip)
            if hasattr(fg_clip, 'resize'):
                fg_clip = fg_clip.resize(bg_clip.size)
                if hasattr(fg_clip, 'set_opacity'):
                    fg_clip = fg_clip.set_opacity(opacity)
                else:
                    fg_clip = fg_clip.with_opacity(opacity)
            else:
                fg_clip = fg_clip.resized(bg_clip.size).with_opacity(opacity)
            comp = mpy.CompositeVideoClip([bg_clip, fg_clip])
            dur = min(bg_clip.duration, fg_clip.duration)
            if hasattr(comp, 'set_duration'):
                comp = comp.set_duration(dur)
            else:
                comp = comp.with_duration(dur)
            _write_clip(comp, out_path, fps=fps)
            print(f"Done. Saved composite: {out_path}")
        return

    if ids_only:
        # Build sequences by ID only
        fg_items, bg_items = collect_ids_only(img_dir, fg_year, bg_year, angle_round, angle_field_index, prefer="earliest")
        if len(fg_items) < 2 or len(bg_items) < 2:
            print("Too few common IDs (need >=2).")
            return
        fg_paths = [it.path for it in fg_items]
        bg_paths = [it.path for it in bg_items]
        if sep_mode:
            out_bg = img_dir / f"bg_ids_{bg_year}.mp4"
            out_fg = img_dir / f"fg_ids_{fg_year}.mp4"
            out_bg_str = input(f"Background video path (blank='{out_bg.name}'): ").strip()
            out_fg_str = input(f"Foreground video path (blank='{out_fg.name}'): ").strip()
            out_bg = Path(out_bg_str) if out_bg_str else out_bg
            out_fg = Path(out_fg_str) if out_fg_str else out_fg
            bg_clip = _build_seq_clip(bg_paths, fps=fps, hold_sec=hold_sec)
            fg_clip = _build_seq_clip(fg_paths, fps=fps, hold_sec=hold_sec)
            bg_clip = _ensure_even_size(bg_clip)
            if hasattr(fg_clip, 'resize'):
                fg_clip = fg_clip.resize(bg_clip.size)
            else:
                fg_clip = fg_clip.resized(bg_clip.size)
            _write_clip(bg_clip, out_bg, fps=fps)
            _write_clip(fg_clip, out_fg, fps=fps)
            print(f"Done. Saved background: {out_bg}")
            print(f"Done. Saved foreground: {out_fg}")
        else:
            default_out = img_dir / f"composite_ids_{bg_year}_bg_{fg_year}_fg.mp4"
            out_str = input(f"Output composite path (blank='{default_out.name}'): ").strip()
            out_path = Path(out_str) if out_str else default_out
            bg_clip = _build_seq_clip(bg_paths, fps=fps, hold_sec=hold_sec)
            fg_clip = _build_seq_clip(fg_paths, fps=fps, hold_sec=hold_sec)
            bg_clip = _ensure_even_size(bg_clip)
            if hasattr(fg_clip, 'resize'):
                fg_clip = fg_clip.resize(bg_clip.size)
                if hasattr(fg_clip, 'set_opacity'):
                    fg_clip = fg_clip.set_opacity(opacity)
                else:
                    fg_clip = fg_clip.with_opacity(opacity)
            else:
                fg_clip = fg_clip.resized(bg_clip.size).with_opacity(opacity)
            comp = mpy.CompositeVideoClip([bg_clip, fg_clip])
            dur = min(bg_clip.duration, fg_clip.duration)
            if hasattr(comp, 'set_duration'):
                comp = comp.set_duration(dur)
            else:
                comp = comp.with_duration(dur)
            _write_clip(comp, out_path, fps=fps)
            print(f"Done. Saved composite: {out_path}")
        return

    # Original (non-path) flow
    if sep_mode:
        out_bg = img_dir / f"bg_{id1}_angle{angle_key}_{bg_year}.mp4"
        out_fg = img_dir / f"fg_{id1}_angle{angle_key}_{fg_year}.mp4"
        out_bg_str = input(f"Background video path (blank='{out_bg.name}'): ").strip()
        out_fg_str = input(f"Foreground video path (blank='{out_fg.name}'): ").strip()
        out_bg = Path(out_bg_str) if out_bg_str else out_bg
        out_fg = Path(out_fg_str) if out_fg_str else out_fg
        make_separate_videos(items, fg_year=fg_year, bg_year=bg_year, out_fg=out_fg, out_bg=out_bg, fps=fps, opacity=opacity, hold_sec=hold_sec)
        print(f"Done. Saved background: {out_bg}")
        print(f"Done. Saved foreground: {out_fg}")
    else:
        default_out = img_dir / f"composite_{id1}_angle{angle_key}_{bg_year}_bg_{fg_year}_fg.mp4"
        out_str = input(f"Output composite path (blank='{default_out.name}'): ").strip()
        out_path = Path(out_str) if out_str else default_out
        make_composite_video(items, fg_year=fg_year, bg_year=bg_year, out_path=out_path, fps=fps, opacity=opacity, hold_sec=hold_sec)
        print(f"Done. Saved composite: {out_path}")


if __name__ == "__main__":
    main()
