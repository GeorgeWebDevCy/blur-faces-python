# blur-faces-python

Small Python CLI for blurring faces in video files.

## Install

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```bash
python blur_faces.py "input-video.mp4"
```

This creates `input-video.blurred.mp4` and keeps the original audio when possible.

You can also choose an output path:

```bash
python blur_faces.py "input-video.mp4" --output "output-video.mp4" --overwrite
```

To blur only one specific person, pass a reference photo of that person:

```bash
python blur_faces.py "input-video.mp4" --reference-image "person.jpg" --overwrite
```

Single-person mode uses a stronger DNN face detector plus short tracking, so the blur is less likely to drop out during movement.

Useful options:

- `--blur-strength 99` controls how strong the face blur is.
- `--padding 0.2` adds extra blur area around each detected face.
- `--min-face-size 40` ignores tiny detections.
- `--reference-image person.jpg` only blurs the matching face.
- `--match-threshold 0.363` adjusts how strictly the face must match.
- `--skip-audio` writes a video-only file.
- `--max-frames 200` is handy for quick test runs.
