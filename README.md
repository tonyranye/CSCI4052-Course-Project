# CSCI4052-Course-Project

## WLASL Top-100 Pipeline

Run the notebooks in this order:

1. `01_dataAnalysis.ipynb`
2. `preprocessing.ipynb`
3. `training.ipynb`

### 1) Build Top-100 Dataset

Open `01_dataAnalysis.ipynb` and run all cells.

This creates:
- `wlasl_full_top_100_dataset/dataset`
- `wlasl_full_top_100_dataset/class_statistics.csv`
- `wlasl_full_top_100_dataset/class_statistics.txt`
- `wlasl_full_top_100_dataset/labels_top100.json`
- `wlasl_full_top_100_dataset/id_to_gloss_top100.json`

### 2) Preprocess Videos (MediaPipe Dynamic Crop)

Open `preprocessing.ipynb` and run all cells.

This creates:
- `processed/wlasl_top100/clips/.../*.npy`
- `processed/wlasl_top100/metadata.jsonl`
- `processed/wlasl_top100/train_metadata.jsonl`
- `processed/wlasl_top100/val_metadata.jsonl`
- `processed/wlasl_top100/test_metadata.jsonl`
- `processed/wlasl_top100/labels_top100.json`

### 3) Train VideoMAE on Top-100

Open `training.ipynb` and run all cells.

This creates checkpoints:
- `checkpoints/wlasl_top100/best.pt`
- `checkpoints/wlasl_top100/last.pt`

## Deployment (Gradio)

Deployment files are in `deployment/`:
- `deployment/app.py`
- `deployment/asl_model.py`
- `deployment/config.py`

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the real-time app:

```bash
cd deployment
python app.py
```

Optional environment variables:
- `ASL_CHECKPOINT`
- `ASL_LABELS`
- `ASL_NUM_FRAMES`
- `ASL_MIN_READY_FRAMES`
- `ASL_INFER_EVERY_N`
- `ASL_SMOOTHING_WINDOW`
- `ASL_IMG_SIZE`