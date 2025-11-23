# Trace the City

A Python-based blob tracking video effect system that creates artistic visualizations by tracking and highlighting motion in videos. Creates dynamic rectangular boxes that follow movement, with optional audio-reactive spawning and custom fill videos.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## Features

- **Multi-blob tracking**: Tracks multiple features simultaneously using optical flow
- **Audio-reactive spawning**: Boxes spawn in sync with audio beats
- **Motion-biased tracking**: Prioritizes high-motion areas for more dynamic effects
- **Custom fill videos**: Replace box interiors with custom video content
- **Multiple UI options**: CLI, Gradio web interface, or Tkinter desktop app
- **GIF support**: Direct GIF processing with automatic conversion
- **No GPU required**: Runs smoothly on CPU-only systems

## Demo





https://github.com/user-attachments/assets/7e71ebc2-5401-4354-8ad5-2d3bc70e2bdb





## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (installation instructions in [ffmpeg-installation.txt](ffmpeg-installation.txt))

### Setup

```bash
# Clone the repository
git clone https://github.com/enkancan/TracetheCity.git
cd TracetheCity

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Gradio Web UI (Recommended)
<img width="1426" height="726" alt="Ekran görüntüsü 2025-11-22 164121" src="https://github.com/user-attachments/assets/22caddf0-b9f3-42cd-9c0b-bb19f49037d5" />


Launch the interactive web interface with full parameter controls:

```bash
python app_gradio.py
```

Then open http://127.0.0.1:7860 in your browser.

### CLI Interface

Quick processing with optimized defaults:

```bash
python main.py
```

You'll be prompted for:
- Input video path
- Output path
- Optional fill video path

### Tkinter Desktop UI

Native desktop interface:

```bash
python ui_blobs.py
```

### GIF Processing

Simplified interface for GIF files:

```bash
python gif_blob.py
```

## Key Parameters

- **pts_per_beat**: Boxes spawned per audio beat (default: 30)
- **ambient_rate**: Background spawn rate per second (default: 8.0)
- **life_frames**: Box lifetime in frames (default: 24)
- **min_size / max_size**: Box size range in pixels (40-160)
- **neighbor_links**: Number of connecting lines between boxes (default: 4)
- **motion_spawn_bias**: Prioritize high-motion areas for spawning
- **single_box_mode**: Single roaming rectangle instead of multi-blob tracking

## Advanced Features

### Audio-Reactive Spawning

The system uses librosa to detect audio onsets and synchronize box spawning with beats for music videos.

### GPT-4 Labeling (Experimental)

Enable semantic labeling of tracked boxes using GPT-4 Vision:

```python
# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Enable in Gradio UI or pass use_gpt_labels=True
```

### Fill Video Effects

Replace box interiors with content from another video for creative composite effects.

### Motion-Biased Spawning

Focus tracking on high-motion regions by enabling motion bias in the Gradio interface.

## Utility Scripts

### Video Splitting

```bash
python split_video_six.py --input video.mp4 --output-dir out/
```

### Year-Layer Composition

For time-lapse comparisons:

```bash
python compose_year_layers.py
# or
python compose_ids_quick.py --dir path/to/images --fg 2024 --bg 2018 --composite
```

## Architecture

Built on OpenCV for computer vision, with:
- ORB and SimpleBlobDetector for feature detection
- Lucas-Kanade optical flow for tracking
- MoviePy for video I/O (v1 and v2 compatible)
- Librosa for audio analysis
- Gradio for web UI


## Credits

Developed from [Blob-Track-Lite](https://github.com/Code-X-Sakthi/Blob-Track-Lite)

## License

MIT License - feel free to use in your own projects!

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

## Support

For questions or issues, please open a GitHub issue.
