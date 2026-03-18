from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from urllib.request import urlopen

import cv2
import imageio_ffmpeg


DNN_FACE_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
DNN_FACE_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20180205_fp16/"
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


@dataclass
class BlurConfig:
    blur_strength: int
    padding: float
    scale_factor: float
    min_neighbors: int
    min_face_size: int
    max_frames: int | None
    preserve_audio: bool
    reference_image: Path | None
    match_threshold: float
    models_dir: Path


@dataclass
class FaceCandidate:
    box: tuple[int, int, int, int]
    similarity: float


@dataclass
class TrackedFaceState:
    box: tuple[int, int, int, int]
    velocity: tuple[float, float]
    missed_frames: int


@dataclass
class DnnFaceDetector:
    net: object
    input_size: tuple[int, int]
    confidence_threshold: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect faces in a video and blur them."
    )
    parser.add_argument("input", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the output video. Defaults to '<input>.blurred.mp4'.",
    )
    parser.add_argument(
        "--blur-strength",
        type=int,
        default=99,
        help="Gaussian blur kernel size. Higher values create a stronger blur.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.20,
        help="Extra padding around each face as a ratio of face size.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.10,
        help="Face detection pyramid scale factor for the Haar cascade.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="How many neighboring detections are required to confirm a face.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Minimum face size in pixels for detection.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional limit for the number of frames to process.",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio remuxing and write video only.",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="Only blur faces that match the person in this reference image.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.363,
        help="Cosine similarity threshold for reference-image matching.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="Directory used to cache downloaded OpenCV face models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.blurred.mp4")


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, BlurConfig]:
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(input_path)
    )

    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    if args.blur_strength < 3:
        raise ValueError("--blur-strength must be at least 3.")
    if args.scale_factor <= 1.0:
        raise ValueError("--scale-factor must be greater than 1.0.")
    if args.min_neighbors < 1:
        raise ValueError("--min-neighbors must be at least 1.")
    if args.min_face_size < 1:
        raise ValueError("--min-face-size must be at least 1.")
    if args.padding < 0:
        raise ValueError("--padding cannot be negative.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1 when provided.")
    if args.match_threshold <= 0:
        raise ValueError("--match-threshold must be greater than 0.")

    reference_image = None
    if args.reference_image is not None:
        reference_image = args.reference_image.expanduser().resolve()
        if not reference_image.exists():
            raise FileNotFoundError(
                f"Reference image does not exist: {reference_image}"
            )

    models_dir = (
        args.models_dir.expanduser().resolve()
        if args.models_dir is not None
        else Path(__file__).resolve().parent / ".models"
    )

    config = BlurConfig(
        blur_strength=args.blur_strength,
        padding=args.padding,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_face_size=args.min_face_size,
        max_frames=args.max_frames,
        preserve_audio=not args.skip_audio,
        reference_image=reference_image,
        match_threshold=args.match_threshold,
        models_dir=models_dir,
    )
    return input_path, output_path, config


def load_haar_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load face detector from {cascade_path}")
    return detector


def ensure_model_downloaded(destination: Path, url: str) -> Path:
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model: {destination.name}")
    with urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    return destination


def ensure_face_matching_model(models_dir: Path) -> Path:
    return ensure_model_downloaded(
        models_dir / "face_recognition_sface_2021dec.onnx",
        SFACE_MODEL_URL,
    )


def ensure_dnn_face_detector_models(models_dir: Path) -> tuple[Path, Path]:
    prototxt_path = ensure_model_downloaded(
        models_dir / "face_detector_deploy.prototxt",
        DNN_FACE_PROTO_URL,
    )
    model_path = ensure_model_downloaded(
        models_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel",
        DNN_FACE_MODEL_URL,
    )
    return prototxt_path, model_path


def load_dnn_face_detector(models_dir: Path) -> DnnFaceDetector:
    prototxt_path, model_path = ensure_dnn_face_detector_models(models_dir)
    net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return DnnFaceDetector(
        net=net,
        input_size=(300, 300),
        confidence_threshold=0.45,
    )


def load_sface_recognizer(model_path: Path):
    recognizer = cv2.FaceRecognizerSF_create(str(model_path), "")
    if recognizer is None:
        raise RuntimeError(
            f"Failed to load SFace face recognizer from {model_path}"
        )
    return recognizer


def detect_faces_with_haar(
    image,
    detector: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
    min_face_size: int,
):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return detector.detectMultiScale(
        grayscale,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
    )


def detect_faces_with_dnn(
    image,
    detector: DnnFaceDetector,
    min_face_size: int,
) -> list[tuple[int, int, int, int]]:
    frame_height, frame_width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=detector.input_size,
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    detector.net.setInput(blob)
    detections = detector.net.forward()

    boxes: list[list[int]] = []
    scores: list[float] = []
    for detection_index in range(detections.shape[2]):
        confidence = float(detections[0, 0, detection_index, 2])
        if confidence < detector.confidence_threshold:
            continue

        left = int(round(detections[0, 0, detection_index, 3] * frame_width))
        top = int(round(detections[0, 0, detection_index, 4] * frame_height))
        right = int(round(detections[0, 0, detection_index, 5] * frame_width))
        bottom = int(round(detections[0, 0, detection_index, 6] * frame_height))

        left = max(0, min(left, frame_width - 1))
        top = max(0, min(top, frame_height - 1))
        right = max(left + 1, min(right, frame_width))
        bottom = max(top + 1, min(bottom, frame_height))

        width = right - left
        height = bottom - top
        if width < min_face_size or height < min_face_size:
            continue

        boxes.append([left, top, width, height])
        scores.append(confidence)

    if not boxes:
        return []

    kept_boxes: list[tuple[int, int, int, int]] = []
    kept_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=detector.confidence_threshold,
        nms_threshold=0.35,
    )
    if len(kept_indices) == 0:
        return []

    for kept_index in kept_indices:
        index = int(kept_index[0] if hasattr(kept_index, "__len__") else kept_index)
        kept_boxes.append(tuple(int(value) for value in boxes[index]))
    return kept_boxes


def detect_faces_for_reference_mode(
    image,
    dnn_detector: DnnFaceDetector,
    haar_detector: cv2.CascadeClassifier,
    min_face_size: int,
) -> list[tuple[int, int, int, int]]:
    dnn_faces = detect_faces_with_dnn(
        image=image,
        detector=dnn_detector,
        min_face_size=min_face_size,
    )
    if dnn_faces:
        return dnn_faces
    return [
        tuple(int(value) for value in face)
        for face in detect_faces_with_haar(
            image=image,
            detector=haar_detector,
            scale_factor=1.10,
            min_neighbors=4,
            min_face_size=min_face_size,
        )
    ]


def choose_largest_face(faces: list):
    return max(faces, key=lambda face: int(face[2] * face[3]))


def build_face_crop(image, face) -> tuple[tuple[int, int, int, int], object]:
    x, y, w, h = (int(value) for value in face[:4])
    pad_x = int(w * 0.35)
    pad_y = int(h * 0.45)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("Failed to crop face region from image.")

    normalized = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_CUBIC)
    return (x, y, w, h), normalized


def compute_face_feature(face_crop, recognizer):
    return recognizer.feature(face_crop)


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + (w / 2.0), y + (h / 2.0))


def box_area(box: tuple[int, int, int, int]) -> int:
    _, _, w, h = box
    return max(0, w) * max(0, h)


def box_iou(
    first_box: tuple[int, int, int, int],
    second_box: tuple[int, int, int, int],
) -> float:
    first_x, first_y, first_w, first_h = first_box
    second_x, second_y, second_w, second_h = second_box

    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_w, second_x + second_w)
    bottom = min(first_y + first_h, second_y + second_h)

    if right <= left or bottom <= top:
        return 0.0

    intersection_area = (right - left) * (bottom - top)
    union_area = box_area(first_box) + box_area(second_box) - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def normalized_center_distance(
    first_box: tuple[int, int, int, int],
    second_box: tuple[int, int, int, int],
) -> float:
    first_center_x, first_center_y = box_center(first_box)
    second_center_x, second_center_y = box_center(second_box)
    reference_diagonal = max(1.0, hypot(first_box[2], first_box[3]))
    return hypot(
        first_center_x - second_center_x,
        first_center_y - second_center_y,
    ) / reference_diagonal


def tracking_proximity_score(
    track_box: tuple[int, int, int, int],
    candidate_box: tuple[int, int, int, int],
) -> float:
    overlap_score = box_iou(track_box, candidate_box)
    distance_score = max(0.0, 1.0 - (normalized_center_distance(track_box, candidate_box) / 1.75))
    size_score = min(box_area(track_box), box_area(candidate_box)) / max(
        1,
        max(box_area(track_box), box_area(candidate_box)),
    )
    return (overlap_score * 0.5) + (distance_score * 0.35) + (size_score * 0.15)


def is_candidate_near_track(
    track_box: tuple[int, int, int, int],
    candidate_box: tuple[int, int, int, int],
) -> bool:
    return (
        box_iou(track_box, candidate_box) >= 0.05
        or normalized_center_distance(track_box, candidate_box) <= 1.25
    )


def choose_reference_candidate(
    candidates: list[FaceCandidate],
    tracked_face: TrackedFaceState | None,
    match_threshold: float,
) -> FaceCandidate | None:
    if not candidates:
        return None

    def candidate_score(candidate: FaceCandidate) -> float:
        proximity_bonus = 0.0
        if tracked_face is not None:
            proximity_bonus = tracking_proximity_score(
                tracked_face.box,
                candidate.box,
            )
        return candidate.similarity + (proximity_bonus * 0.20)

    primary_matches = [
        candidate
        for candidate in candidates
        if candidate.similarity >= match_threshold
    ]
    if primary_matches:
        return max(primary_matches, key=candidate_score)

    if tracked_face is None:
        return None

    recovery_threshold = max(0.30, match_threshold - 0.10)
    recovery_matches = [
        candidate
        for candidate in candidates
        if candidate.similarity >= recovery_threshold
        and is_candidate_near_track(tracked_face.box, candidate.box)
    ]
    if recovery_matches:
        return max(recovery_matches, key=candidate_score)
    return None


def clamp_box_to_frame(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    w = max(1, min(w, frame_width))
    h = max(1, min(h, frame_height))
    x = max(0, min(x, frame_width - w))
    y = max(0, min(y, frame_height - h))
    return (x, y, w, h)


def update_tracked_face(
    tracked_face: TrackedFaceState | None,
    new_box: tuple[int, int, int, int],
) -> TrackedFaceState:
    if tracked_face is None:
        return TrackedFaceState(
            box=new_box,
            velocity=(0.0, 0.0),
            missed_frames=0,
        )

    previous_center_x, previous_center_y = box_center(tracked_face.box)
    current_center_x, current_center_y = box_center(new_box)
    observed_velocity = (
        current_center_x - previous_center_x,
        current_center_y - previous_center_y,
    )
    smoothed_velocity = (
        (tracked_face.velocity[0] * 0.55) + (observed_velocity[0] * 0.45),
        (tracked_face.velocity[1] * 0.55) + (observed_velocity[1] * 0.45),
    )
    return TrackedFaceState(
        box=new_box,
        velocity=smoothed_velocity,
        missed_frames=0,
    )


def predict_tracked_box(
    tracked_face: TrackedFaceState,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    center_x, center_y = box_center(tracked_face.box)
    predicted_box = (
        int(round(center_x + tracked_face.velocity[0] - (tracked_face.box[2] / 2.0))),
        int(round(center_y + tracked_face.velocity[1] - (tracked_face.box[3] / 2.0))),
        tracked_face.box[2],
        tracked_face.box[3],
    )
    return clamp_box_to_frame(predicted_box, frame_width, frame_height)


def advance_tracked_face(
    tracked_face: TrackedFaceState,
    frame_width: int,
    frame_height: int,
) -> TrackedFaceState:
    return TrackedFaceState(
        box=predict_tracked_box(tracked_face, frame_width, frame_height),
        velocity=(tracked_face.velocity[0] * 0.85, tracked_face.velocity[1] * 0.85),
        missed_frames=tracked_face.missed_frames + 1,
    )


def load_reference_feature(
    reference_image: Path,
    models_dir: Path,
    dnn_detector: DnnFaceDetector,
    haar_detector: cv2.CascadeClassifier,
    min_face_size: int,
):
    sface_model = ensure_face_matching_model(models_dir)
    recognizer = load_sface_recognizer(sface_model)

    reference_frame = cv2.imread(str(reference_image))
    if reference_frame is None:
        raise RuntimeError(f"Failed to read reference image: {reference_image}")

    reference_faces = detect_faces_for_reference_mode(
        image=reference_frame,
        dnn_detector=dnn_detector,
        haar_detector=haar_detector,
        min_face_size=min_face_size,
    )
    if len(reference_faces) == 0:
        reference_crop = cv2.resize(
            reference_frame,
            (112, 112),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        reference_face = choose_largest_face(reference_faces)
        _, reference_crop = build_face_crop(reference_frame, reference_face)
    reference_feature = compute_face_feature(reference_crop, recognizer)
    return recognizer, reference_feature


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def blur_faces_in_frame(
    frame,
    faces,
    blur_strength: int,
    padding_ratio: float,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    requested_kernel = ensure_odd(blur_strength)

    for (x, y, w, h) in faces:
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_width, x + w + pad_x)
        y2 = min(frame_height, y + h + pad_y)

        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            continue

        smallest_dimension = min(region.shape[0], region.shape[1])
        if smallest_dimension < 3:
            continue

        max_kernel = smallest_dimension if smallest_dimension % 2 == 1 else smallest_dimension - 1
        kernel_size = min(requested_kernel, max_kernel)
        if kernel_size < 3:
            continue

        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)


def write_progress(processed_frames: int, total_frames: int) -> None:
    if total_frames > 0:
        percent = processed_frames / total_frames * 100
        print(
            f"\rProcessed {processed_frames}/{total_frames} frames ({percent:5.1f}%)",
            end="",
            flush=True,
        )
    else:
        print(f"\rProcessed {processed_frames} frames", end="", flush=True)


def remux_original_audio(
    blurred_video_path: Path,
    original_video_path: Path,
    output_path: Path,
) -> None:
    ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(blurred_video_path),
        "-i",
        str(original_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg audio remux failed")


def process_video(input_path: Path, output_path: Path, config: BlurConfig) -> None:
    haar_detector = load_haar_face_detector()
    dnn_detector = None
    recognizer = None
    reference_feature = None
    tracked_face = None

    if config.reference_image is not None:
        print(f"Using reference image: {config.reference_image}")
        dnn_detector = load_dnn_face_detector(config.models_dir)
        recognizer, reference_feature = load_reference_feature(
            config.reference_image,
            config.models_dir,
            dnn_detector,
            haar_detector,
            config.min_face_size,
        )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    display_total = (
        min(total_frames, config.max_frames)
        if config.max_frames is not None and total_frames > 0
        else total_frames
    )
    tracking_grace_frames = max(4, int(round(fps * 0.5)))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_frames = 0
    with tempfile.TemporaryDirectory(prefix="blur-faces-") as temp_dir:
        temp_video_path = Path(temp_dir) / "blurred_video.mp4"
        writer = cv2.VideoWriter(
            str(temp_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Failed to create the output video writer.")

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if reference_feature is not None:
                    detected_faces = detect_faces_for_reference_mode(
                        image=frame,
                        dnn_detector=dnn_detector,
                        haar_detector=haar_detector,
                        min_face_size=config.min_face_size,
                    )
                    candidates: list[FaceCandidate] = []
                    for face in detected_faces:
                        try:
                            face_box, face_crop = build_face_crop(frame, face)
                            candidate_feature = compute_face_feature(
                                face_crop,
                                recognizer,
                            )
                        except (RuntimeError, cv2.error):
                            continue

                        similarity = recognizer.match(
                            reference_feature,
                            candidate_feature,
                            cv2.FaceRecognizerSF_FR_COSINE,
                        )
                        candidates.append(
                            FaceCandidate(
                                box=face_box,
                                similarity=float(similarity),
                            )
                        )

                    chosen_candidate = choose_reference_candidate(
                        candidates=candidates,
                        tracked_face=tracked_face,
                        match_threshold=config.match_threshold,
                    )
                    if chosen_candidate is not None:
                        tracked_face = update_tracked_face(
                            tracked_face,
                            chosen_candidate.box,
                        )
                        faces = [tracked_face.box]
                    elif tracked_face is not None and tracked_face.missed_frames < tracking_grace_frames:
                        tracked_face = advance_tracked_face(
                            tracked_face,
                            frame_width,
                            frame_height,
                        )
                        faces = [tracked_face.box]
                    else:
                        tracked_face = None
                        faces = []
                else:
                    detected_faces = detect_faces_with_haar(
                        image=frame,
                        detector=haar_detector,
                        scale_factor=config.scale_factor,
                        min_neighbors=config.min_neighbors,
                        min_face_size=config.min_face_size,
                    )
                    faces = detected_faces
                blur_faces_in_frame(
                    frame=frame,
                    faces=faces,
                    blur_strength=config.blur_strength,
                    padding_ratio=config.padding,
                )

                writer.write(frame)
                processed_frames += 1

                if processed_frames == 1 or processed_frames % 25 == 0:
                    write_progress(processed_frames, display_total)

                if config.max_frames is not None and processed_frames >= config.max_frames:
                    break
        finally:
            capture.release()
            writer.release()

        if processed_frames == 0:
            raise RuntimeError("No frames were processed. Check that the input video is valid.")

        write_progress(processed_frames, display_total)
        print()

        if config.preserve_audio:
            try:
                remux_original_audio(temp_video_path, input_path, output_path)
                return
            except Exception as exc:
                print(f"Audio remux failed, writing video without audio instead: {exc}")

        if output_path.exists():
            output_path.unlink()
        shutil.move(str(temp_video_path), str(output_path))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    input_path, output_path, config = validate_args(args)
    process_video(input_path, output_path, config)
    print(f"Blurred video saved to: {output_path}")


if __name__ == "__main__":
    main()
