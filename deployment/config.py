from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DeploymentConfig:
    checkpoint_path: Path
    labels_path: Path
    num_frames: int
    min_ready_frames: int
    infer_every_n: int
    smoothing_window: int
    prob_smoothing_window: int
    img_size: int
    detect_every_n: int
    bbox_hold_frames: int
    min_confidence: float
    motion_gate_enabled: bool
    motion_gate_threshold: float
    webcam_width: int
    webcam_height: int
    annotated_width: int
    annotated_height: int


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}



def _resolve_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]



def _default_checkpoint_path() -> Path:
    env_path = os.getenv("ASL_CHECKPOINT")
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    candidates = [
        PROJECT_ROOT / "checkpoints/wlasl_top100/best.pt",
        PROJECT_ROOT / "checkpoints/wlasl_top100/last.pt",
    ]
    return _resolve_existing(candidates)



def _default_labels_path() -> Path:
    env_path = os.getenv("ASL_LABELS")
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    candidates = [
        PROJECT_ROOT / "processed/wlasl_top100/labels_top100.json",
        PROJECT_ROOT / "wlasl_full_top_100_dataset/labels_top100.json",
    ]
    return _resolve_existing(candidates)



def load_labels(labels_path: Path) -> list[str]:
    with labels_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "labels" in payload:
        return [str(x) for x in payload["labels"]]
    if isinstance(payload, list):
        return [str(x) for x in payload]

    raise ValueError(f"Unsupported labels format in: {labels_path}")



def build_config() -> DeploymentConfig:
    num_frames = int(os.getenv("ASL_NUM_FRAMES", "16"))
    min_ready = int(os.getenv("ASL_MIN_READY_FRAMES", "8"))
    return DeploymentConfig(
        checkpoint_path=_default_checkpoint_path(),
        labels_path=_default_labels_path(),
        num_frames=num_frames,
        min_ready_frames=max(1, min(num_frames, min_ready)),
        infer_every_n=max(1, int(os.getenv("ASL_INFER_EVERY_N", "2"))),
        smoothing_window=max(1, int(os.getenv("ASL_SMOOTHING_WINDOW", "4"))),
        prob_smoothing_window=max(1, int(os.getenv("ASL_PROB_SMOOTHING_WINDOW", "5"))),
        img_size=int(os.getenv("ASL_IMG_SIZE", "224")),
        detect_every_n=max(1, int(os.getenv("ASL_DETECT_EVERY_N", "2"))),
        bbox_hold_frames=max(1, int(os.getenv("ASL_BBOX_HOLD_FRAMES", "10"))),
        min_confidence=float(os.getenv("ASL_MIN_CONFIDENCE", "0.25")),
        motion_gate_enabled=_env_bool("ASL_MOTION_GATE_ENABLED", True),
        motion_gate_threshold=float(os.getenv("ASL_MOTION_GATE_THRESHOLD", "4.5")),
        webcam_width=int(os.getenv("ASL_WEBCAM_WIDTH", "480")),
        webcam_height=int(os.getenv("ASL_WEBCAM_HEIGHT", "320")),
        annotated_width=int(os.getenv("ASL_ANNOTATED_WIDTH", "480")),
        annotated_height=int(os.getenv("ASL_ANNOTATED_HEIGHT", "320")),
    )
