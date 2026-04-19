from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms

from asl_model import ASLVideoMAEClassifier
from config import build_config, load_labels


CFG = build_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
LABELS = load_labels(CFG.labels_path)
NUM_CLASSES = len(LABELS)

if not CFG.checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CFG.checkpoint_path}. Run training.ipynb first or set ASL_CHECKPOINT."
    )

model = ASLVideoMAEClassifier(num_classes=NUM_CLASSES)
state = torch.load(CFG.checkpoint_path, map_location="cpu")
model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
model.to(DEVICE).eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def _ensure_hand_landmarker_model(path: Path) -> Path | None:
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(HAND_LANDMARKER_URL, str(path))
        return path
    except Exception as e:
        print(f"[app] Could not download hand_landmarker.task: {e}")
        return None


def _init_mediapipe_detector() -> tuple[Any | None, str, str]:
    if not hasattr(mp, "tasks") or not hasattr(mp.tasks, "vision"):
        return None, "fallback-full-frame", "MediaPipe Tasks API unavailable"

    model_path = _ensure_hand_landmarker_model(HAND_LANDMARKER_PATH)
    if model_path is None:
        return None, "fallback-full-frame", "HandLandmarker model unavailable"

    try:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        return detector, "tasks-hand-landmarker", ""
    except Exception as e:
        return None, "fallback-full-frame", f"HandLandmarker init failed: {e}"


hand_detector, MEDIAPIPE_MODE, mediapipe_note = _init_mediapipe_detector()
print(f"[app] MediaPipe mode: {MEDIAPIPE_MODE}")
if mediapipe_note:
    print(f"[app] MediaPipe note: {mediapipe_note}")

frame_buffer: deque[np.ndarray] = deque(maxlen=CFG.num_frames)
prob_history: deque[np.ndarray] = deque(maxlen=CFG.prob_smoothing_window)
frame_counter = 0
last_prediction: tuple[str, float, list[tuple[str, float]], int] = ("Collecting frames...", 0.0, [], -1)
prev_bbox: tuple[int, int, int, int] | None = None
frames_since_detect = 0
prev_motion_gray: np.ndarray | None = None


def _bbox_from_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    margin_ratio: float = 0.15,
) -> tuple[int, int, int, int] | None:
    if not points:
        return None

    xs = np.array([p[0] for p in points], dtype=np.float32) * width
    ys = np.array([p[1] for p in points], dtype=np.float32) * height

    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = bw * margin_ratio
    my = bh * margin_ratio

    x1 = int(max(0, np.floor(x1 - mx)))
    y1 = int(max(0, np.floor(y1 - my)))
    x2 = int(min(width, np.ceil(x2 + mx)))
    y2 = int(min(height, np.ceil(y2 + my)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _smooth_bbox(
    new_bbox: tuple[int, int, int, int],
    old_bbox: tuple[int, int, int, int],
    alpha: float = 0.35,
) -> tuple[int, int, int, int]:
    return tuple(int(round(alpha * n + (1.0 - alpha) * o)) for n, o in zip(new_bbox, old_bbox))


def _detect_hand_points(frame_rgb: np.ndarray) -> list[tuple[float, float]]:
    if hand_detector is None:
        return []

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = hand_detector.detect(mp_image)

    points: list[tuple[float, float]] = []
    if result and result.hand_landmarks:
        for hand in result.hand_landmarks:
            for lm in hand:
                points.append((float(lm.x), float(lm.y)))
    return points


def dynamic_crop(frame_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    global prev_bbox, frames_since_detect

    h, w = frame_rgb.shape[:2]
    should_detect = (
        hand_detector is not None
        and (frame_counter % CFG.detect_every_n == 0 or prev_bbox is None)
    )
    points = _detect_hand_points(frame_rgb) if should_detect else []
    bbox = _bbox_from_points(points, w, h)

    if bbox is not None:
        if prev_bbox is not None:
            bbox = _smooth_bbox(bbox, prev_bbox)
        prev_bbox = bbox
        frames_since_detect = 0
    elif prev_bbox is not None and frames_since_detect < CFG.bbox_hold_frames:
        bbox = prev_bbox
        frames_since_detect += 1
    else:
        prev_bbox = None
        frames_since_detect = 0

    if bbox is None:
        crop = frame_rgb
    else:
        x1, y1, x2, y2 = bbox
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame_rgb

    resized = cv2.resize(crop, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_AREA)
    return resized, bbox


@torch.inference_mode()
def predict_from_buffer() -> np.ndarray | None:
    if len(frame_buffer) < CFG.min_ready_frames:
        return None

    clip_list = list(frame_buffer)
    if len(clip_list) < CFG.num_frames:
        clip_list.extend([clip_list[-1]] * (CFG.num_frames - len(clip_list)))

    clip = np.stack(clip_list[-CFG.num_frames :], axis=0)
    frames = torch.stack([transform(frame) for frame in clip], dim=0)
    frames = frames.unsqueeze(0).to(DEVICE, non_blocking=True)

    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
        logits = model(frames)
        probs = torch.softmax(logits.float(), dim=-1)
    return probs[0].detach().cpu().numpy().astype(np.float32)


def stream_predict(frame: np.ndarray) -> tuple[str, str, np.ndarray, str]:
    global frame_counter, last_prediction, prev_motion_gray

    if frame is None:
        return "Waiting for webcam...", "", np.zeros((CFG.img_size, CFG.img_size, 3), dtype=np.uint8), ""

    frame_counter += 1
    crop, bbox = dynamic_crop(frame)
    frame_buffer.append(crop)

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    if prev_motion_gray is None:
        motion_score = 999.0
    else:
        motion_score = float(cv2.absdiff(crop_gray, prev_motion_gray).mean())
    prev_motion_gray = crop_gray

    motion_ok = (not CFG.motion_gate_enabled) or (motion_score >= CFG.motion_gate_threshold)

    should_infer = (frame_counter % CFG.infer_every_n == 0 or len(frame_buffer) < CFG.min_ready_frames)
    if should_infer and motion_ok:
        probs_vec = predict_from_buffer()
        if probs_vec is not None:
            prob_history.append(probs_vec)

    if len(prob_history) > 0:
        smooth_probs = np.mean(np.stack(list(prob_history), axis=0), axis=0)
        topk = min(3, smooth_probs.shape[0])
        top_idxs = np.argsort(smooth_probs)[-topk:][::-1]
        top_scores = smooth_probs[top_idxs]
        pred_idx = int(top_idxs[0])
        conf = float(top_scores[0])
        top_preds = [(LABELS[int(i)], float(s)) for i, s in zip(top_idxs, top_scores)]
        pred = LABELS[pred_idx]
        last_prediction = (pred, conf, top_preds, pred_idx)
    else:
        pred, conf, top_preds, pred_idx = last_prediction

    motion_state = "moving" if motion_ok else "idle(gated)"
    status = (
        f"Buffer: {len(frame_buffer)}/{CFG.num_frames} | "
        f"Ready at: {CFG.min_ready_frames} | "
        f"Infer every: {CFG.infer_every_n} frame(s) | "
        f"Detect every: {CFG.detect_every_n} | "
        f"Motion: {motion_score:.2f} ({motion_state}) | "
        f"Prob window: {len(prob_history)}/{CFG.prob_smoothing_window}"
    )

    annotated = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (46, 204, 113), 2)

    topk_text = " | ".join([f"{name}: {score:.1%}" for name, score in top_preds])

    if pred == "Collecting frames...":
        return pred, status, annotated, topk_text

    if conf < CFG.min_confidence:
        return f"Uncertain (max {conf:.2%})", status, annotated, topk_text

    return f"Prediction: {pred} ({conf:.2%})", status, annotated, topk_text


with gr.Blocks(title="Real-time ASL Translator (VideoMAE + MediaPipe)") as demo:
    gr.Markdown("## Real-time ASL Translator")
    gr.Markdown(
        f"Checkpoint: {Path(CFG.checkpoint_path).name} | "
        f"Classes: {NUM_CLASSES} | "
        f"MediaPipe mode: {MEDIAPIPE_MODE}"
    )

    gr.Markdown(
        "Best demo words to start with: inform, green, hair, cat, cry, class, door, have, different, check"
    )

    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"],
            streaming=True,
            type="numpy",
            label="Webcam",
            width=CFG.webcam_width,
            height=CFG.webcam_height,
        )
        annotated_frame = gr.Image(
            type="numpy",
            label="Annotated Frame",
            width=CFG.annotated_width,
            height=CFG.annotated_height,
        )

    with gr.Row():
        pred_text = gr.Textbox(label="Predicted ASL Word")
        status_text = gr.Textbox(label="Status")
        topk_text = gr.Textbox(label="Top-3 Predictions")

    webcam.stream(
        fn=stream_predict,
        inputs=webcam,
        outputs=[pred_text, status_text, annotated_frame, topk_text],
    )

if __name__ == "__main__":
    demo.launch(share=True)
