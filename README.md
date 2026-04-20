# ASL Sign Language Interpreter CSCI 4052U Final Project

**Team:** Tony Akinniranye, Lukas Fenkam, Jaathavan Ranjanathan

**Course:** CSCI 4052U - Machine Learning II, Ontario Tech University

---

## Overview

This project is an end-to-end AI application that recognizes American Sign Language (ASL) gestures from short video clips. Given an input video (or webcam stream) of a person signing, the system classifies the gesture into one of **100 ASL word categories** using a fine-tuned VideoMAE transformer model trained on WLASL.

The current repository includes both training notebooks and a deployment app (`deployment/app.py`) that performs real-time webcam inference with MediaPipe-assisted dynamic cropping via Gradio.

### Problem Formulation: Why Modern Neural Models Are Needed

Traditional computer-vision pipelines struggle with sign-language understanding because ASL semantics are strongly temporal. A detector such as YOLO is excellent for spatial object localization in single frames, but it does not model how handshape, trajectory, and timing evolve across a clip.

Before modern video transformers, many systems relied on static hand snapshots or letter-by-letter fingerspelling workflows, which are inconvenient for natural word-level signing. Our approach instead predicts **whole-word glosses** directly from short video clips by learning spatial and temporal context jointly.

---

## Dataset

We use the **WLASL (Word-Level American Sign Language)** dataset, loaded from Hugging Face (`Voxel51/WLASL`) and processed into a Top-100 subset.

### Top 100 subset

- Classes: **100**
- Samples per class: **100**
- Total samples: **10,000**
- Processed labels file: `processed/wlasl_top100/labels_top100.json`

### Stratified split

| Split | Count |
|---|---:|
| Train (70%) | 7,000 |
| Validation (15%) | 1,500 |
| Test (15%) | 1,500 |

Stratified splitting preserves class balance across all splits.

### Supported signs

The deployed model predicts 100 gloss labels:

`book`, `drink`, `deaf`, `visit`, `wait`, `water`, `wife`, `yellow`, `backpack`, `bar`, `brother`, `cat`, `check`, `fine`, `class`, `cry`, `different`, `door`, `green`, `hair`, `have`, `headache`, `inform`, `help`, `no`, `thin`, `walk`, `year`, `yes`, `black`, `computer`, `cool`, `finish`, `hot`, `like`, `many`, `mother`, `now`, `orange`, `table`, `thanksgiving`, `before`, `what`, `woman`, `bed`, `blue`, `bowling`, `can`, `dog`, `family`, `fish`, `graduate`, `chair`, `hat`, `hearing`, `kiss`, `language`, `later`, `man`, `shirt`, `study`, `tall`, `white`, `go`, `wrong`, `accident`, `apple`, `bird`, `change`, `color`, `corn`, `cow`, `dance`, `dark`, `clothes`, `doctor`, `eat`, `enjoy`, `forget`, `give`, `last`, `meet`, `pink`, `pizza`, `play`, `who`, `school`, `secretary`, `short`, `time`, `want`, `work`, `africa`, `basketball`, `birthday`, `brown`, `candy`, `but`, `cheat`, `city`

---

## Theoretical and Software Engineering Discussion

### 1. Neural Network Design (ASLVideoMAE)
Our core model, `ASLVideoMAE`, is built upon the `MCG-NJU/videomae-base` architecture loaded via HuggingFace Transformers. 
* **Theoretical Justification:** We opted for a Video Masked Autoencoder (VideoMAE) rather than a standard 3D-CNN because Vision Transformers (ViTs) excel at capturing long-range dependencies in sequence data. VideoMAE treats video as a sequence of tubelets, allowing the self-attention mechanism to heavily weight the relationships between spatial hand positions across different temporal frames.
* **Training Details:** The model was trained in PyTorch using CrossEntropyLoss and an `AdamW` optimizer (with weight decay) paired with a `CosineAnnealingLR` scheduler to ensure smooth convergence over 100 epochs.

### 2. Software Components & Pipeline
Our end-to-end system is broken down into modular Jupyter Notebooks:
* **Data Ingestion (`01_dataAnalysis.ipynb`):** We use the Voxel51/FiftyOne library to programmatically fetch and filter the WLASL (Word-Level ASL) dataset from HuggingFace, narrowing it down to the top 100 classes.
* **Data Preprocessing (`preprocessing.ipynb`):** Videos are processed using OpenCV (`cv2`) to extract exactly 16 uniform frames per video, resized to 224x224. We integrated **Google MediaPipe** to detect hand landmarks, ensuring the network emphasizes hand articulation.
* **Modeling & Inference (`models.ipynb` & `training.ipynb`):** The PyTorch inference engine takes the processed `(16, 3, 224, 224)` tensor and outputs the class probabilities. 

---
## Model and Pipeline

### Neural Network Design

Our core model is `ASLVideoMAEClassifier`, based on the Hugging Face `MCG-NJU/videomae-base` model.

- Backbone: VideoMAE transformer encoder
- Head: `LayerNorm -> Dropout -> Linear(hidden_size, 100)`
- Loss: cross-entropy
- Optimizer: AdamW (with weight decay)
- Scheduler: cosine annealing

### Fine-Tuning and Training Loop (Top100)

- Train/val/test metadata: `7000 / 1500 / 1500` samples
- Clip length: 16 frames per sample
- Batch size: 2
- Epochs: up to 100
- Early stopping warmup: 25 minimum epochs before stopping checks
- Learning rate: `3e-5`
- Weight decay: `0.01`
- Loss: `nn.CrossEntropyLoss()`
- Optimizer: `AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)`
- Scheduler: `CosineAnnealingLR(optimizer, T_max=100)`
- Data loading: shuffled training loader, deterministic validation loader, `num_workers=4`, `pin_memory=torch.cuda.is_available()`

### End-to-End Flow

```text
Raw .mp4 video or webcam frame stream
      ->
Frame extraction / buffering (16 frames)
      ->
MediaPipe-based dynamic crop (fallback to full frame)
      ->
Resize + normalize -> (16, 3, 224, 224)
      ->
VideoMAE forward pass
      ->
Token mean pooling
      ->
Classifier head -> (100,) logits
      ->
Softmax + argmax (+ temporal smoothing in deployment)
      ->
Predicted ASL gloss
```

### Tensor Encoding and Model Interface

- Input interface: application frames are converted to RGB, dynamically cropped with MediaPipe, resized to `224x224`, normalized, and stacked into `(T, C, H, W)`.
- Model tensor: deployment passes `pixel_values` with shape `(B, T, C, H, W) = (1, 16, 3, 224, 224)`.
- Output tensor: classifier head returns logits of shape `(1, 100)`.
- Decoding: `softmax -> top-k -> argmax` maps logits to gloss labels loaded from `labels_top100.json`.
- Runtime integration: `deployment/app.py` streams frames and calls `ASLVideoMAEClassifier` from `deployment/asl_model.py`, with file/path controls in `deployment/config.py`.

---

## Software Components

- `01_dataAnalysis.ipynb`: dataset loading, filtering, and export
- `preprocessing.ipynb`: frame extraction and preprocessing pipeline
- `models.ipynb`: model architecture definitions and checks
- `training.ipynb`: training loop, validation, checkpoints, metrics
- `deployment/config.py`: deployment configuration and artifact path resolution
- `deployment/asl_model.py`: deployable VideoMAE classifier module
- `deployment/app.py`: Gradio real-time webcam inference app

---

## Repository Structure

```text
.
├── 01_dataAnalysis.ipynb
├── preprocessing.ipynb
├── models.ipynb
├── training.ipynb
├── deployment/
│   ├── app.py
│   ├── asl_model.py
│   └── config.py
├── checkpoints/
│   └── wlasl_top100/
│       ├── best.pt
│       └── last.pt
├── processed/
│   └── wlasl_top100/
│       ├── labels_top100.json
│       ├── train_metadata.jsonl
│       ├── val_metadata.jsonl
│       ├── test_metadata.jsonl
│       └── clips/                   # Not included as it is too large to push to GitHub
├── wlasl_full_top_100_dataset/
│   ├── labels_top100.json
│   └── class_statistics.txt
├── requirements.txt
└── README.md
```

`wlasl_top20/` is retained as an earlier experimental subset and is not the current training/deployment target.

---

## Setup and Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU recommended for training (CPU inference is supported)

### Installation

```bash
git clone https://github.com/tonyranye/CSCI4052-Course-Project
cd CSCI4052-Course-Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Project

### Notebook workflow

Run notebooks in this order:

1. `01_dataAnalysis.ipynb` - load/filter WLASL data
2. `preprocessing.ipynb` - generate processed Top100 clips and metadata
3. `models.ipynb` - verify model architecture
4. `training.ipynb` - train and validate model, save checkpoints

### Deployment

To set up the Gradio link run:

```bash
python deployment/app.py
```

which will generate a public Gradio URL for the application.

### Demo Video

https://github.com/user-attachments/assets/bcf290dc-7ed1-4af7-856f-86cc0634af25


## Dependencies

Main dependencies (see `requirements.txt` for full list):

- torch
- torchvision
- transformers
- mediapipe
- opencv-python
- numpy
- scikit-learn
- matplotlib
- fiftyone
- datasets
- av
- pandas
- tqdm
- gradio
- torchcodec

---

## References

1. VideoMAE (pretrained backbone): https://arxiv.org/abs/2203.12602
2. Hugging Face model card (`MCG-NJU/videomae-base`): https://huggingface.co/MCG-NJU/videomae-base
3. WLASL dataset paper: https://arxiv.org/abs/1910.11006
4. WLASL source used in this project (`Voxel51/WLASL`): https://huggingface.co/datasets/Voxel51/WLASL

---

## Presentation and Video

- Slides: *https://docs.google.com/presentation/d/17DZnsnPuLS3JhVJToF3YoACcxvVZ--xMSbrb241gj6Q/edit?slide=id.p#slide=id.p*
- Full Video with Presentation and Demo: *https://www.youtube.com/watch?v=0FKazTIN4EM*

---
 
