from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

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

if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
    )
    MEDIAPIPE_MODE = "holistic"
else:
    holistic = None
    MEDIAPIPE_MODE = "fallback-full-frame"

frame_buffer: deque[np.ndarray] = deque(maxlen=CFG.num_frames)
pred_history: deque[int] = deque(maxlen=CFG.smoothing_window)
frame_counter = 0
last_prediction: tuple[str, float, list[tuple[str, float]], int] = ("Collecting frames...", 0.0, [], -1)


def _compute_bbox(results: Any, width: int, height: int, margin_ratio: float = 0.15) -> tuple[int, int, int, int] | None:
    points: list[tuple[float, float]] = []

    if results and results.pose_landmarks:
        points.extend((lm.x, lm.y) for lm in results.pose_landmarks.landmark)
    if results and results.left_hand_landmarks:
        points.extend((lm.x, lm.y) for lm in results.left_hand_landmarks.landmark)
    if results and results.right_hand_landmarks:
        points.extend((lm.x, lm.y) for lm in results.right_hand_landmarks.landmark)

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


def dynamic_crop(frame_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    if holistic is None:
        resized = cv2.resize(frame_rgb, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_AREA)
        return resized, None

    h, w = frame_rgb.shape[:2]
    results = holistic.process(frame_rgb)
    bbox = _compute_bbox(results, w, h)

    if bbox is None:
        crop = frame_rgb
    else:
        x1, y1, x2, y2 = bbox
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame_rgb

    resized = cv2.resize(crop, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_AREA)
    return resized, bbox


@torch.no_grad()
def predict_from_buffer() -> tuple[str, float, list[tuple[str, float]], int]:
    if len(frame_buffer) < CFG.min_ready_frames:
        return "Collecting frames...", 0.0, [], -1

    clip_list = list(frame_buffer)
    if len(clip_list) < CFG.num_frames:
        clip_list.extend([clip_list[-1]] * (CFG.num_frames - len(clip_list)))

    clip = np.stack(clip_list[-CFG.num_frames :], axis=0)
    frames = torch.stack([transform(frame) for frame in clip], dim=0)
    frames = frames.unsqueeze(0).to(DEVICE, non_blocking=True)

    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
        logits = model(frames)
        probs = torch.softmax(logits.float(), dim=-1)

    conf, pred = probs.max(dim=-1)
    idx = int(pred.item())

    topk = min(3, probs.shape[-1])
    top_vals, top_idxs = torch.topk(probs[0], k=topk)
    top_predictions = [(LABELS[int(i.item())], float(v.item())) for v, i in zip(top_vals, top_idxs)]

    return LABELS[idx], float(conf.item()), top_predictions, idx


def stream_predict(frame: np.ndarray) -> tuple[str, str, np.ndarray, str]:
    global frame_counter, last_prediction

    if frame is None:
        return "Waiting for webcam...", "", np.zeros((CFG.img_size, CFG.img_size, 3), dtype=np.uint8), ""

    crop, bbox = dynamic_crop(frame)
    frame_buffer.append(crop)
    frame_counter += 1

    if frame_counter % CFG.infer_every_n == 0 or len(frame_buffer) < CFG.min_ready_frames:
        last_prediction = predict_from_buffer()

    pred, conf, top_preds, pred_idx = last_prediction

    if pred_idx >= 0:
        pred_history.append(pred_idx)
        voted_idx = max(set(pred_history), key=list(pred_history).count)
        pred = LABELS[voted_idx]

    status = (
        f"Buffer: {len(frame_buffer)}/{CFG.num_frames} | "
        f"Ready at: {CFG.min_ready_frames} | "
        f"Infer every: {CFG.infer_every_n} frame(s)"
    )

    annotated = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (46, 204, 113), 2)

    topk_text = " | ".join([f"{name}: {score:.1%}" for name, score in top_preds])

    if pred == "Collecting frames...":
        return pred, status, annotated, topk_text

    return f"Prediction: {pred} ({conf:.2%})", status, annotated, topk_text


with gr.Blocks(title="Real-time ASL Translator (VideoMAE + MediaPipe)") as demo:
    gr.Markdown("## Real-time ASL Translator")
    gr.Markdown(
        f"Checkpoint: {Path(CFG.checkpoint_path).name} | "
        f"Classes: {NUM_CLASSES} | "
        f"MediaPipe mode: {MEDIAPIPE_MODE}"
    )

    webcam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
    pred_text = gr.Textbox(label="Predicted ASL Word")
    status_text = gr.Textbox(label="Status")
    annotated_frame = gr.Image(type="numpy", label="Annotated Frame")
    topk_text = gr.Textbox(label="Top-3 Predictions")

    webcam.stream(
        fn=stream_predict,
        inputs=webcam,
        outputs=[pred_text, status_text, annotated_frame, topk_text],
    )

if __name__ == "__main__":
    demo.launch(share=True)
