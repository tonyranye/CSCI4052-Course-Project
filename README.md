# ASL Sign Language Interpreter — CSCI 4052U Final Project
 
**Team:** Tony Akinniranye, Lukas Fenkam, Jaathavan  
**Course:** CSCI 4052U — Machine Learning II, Ontario Tech University
 
---
 
## Overview
 
This project is an end-to-end AI application that recognizes American Sign Language (ASL) gestures from short video clips. Given an input video of a person signing, the system classifies the gesture into one of 20 ASL word categories using a fine-tuned VideoMAE transformer model trained on the WLASL dataset.
 
**The 20 supported signs:**
`accident`, `bar`, `bed`, `before`, `bowling`, `call`, `candy`, `champion`, `change`, `check`, `cold`, `computer`, `cool`, `cousin`, `deaf`, `delay`, `drink`, `far`, `go`, `help`
 
---
## Dataset
 
We use the **WLASL (Word-Level American Sign Language)** dataset, loaded via FiftyOne from the Hugging Face Hub (`Voxel51/WLASL`). From the full 11,980-sample dataset, we work with the top 20 most frequent gloss classes, yielding 227 video clips.
 
**Dataset split (stratified):**
 
| Split | Count |
|---|---|
| Train (70%) | 158 |
| Validation (15%) | 34 |
| Test (15%) | 35 |
 
Stratified splitting ensures each class is proportionally represented across all three splits.
 
---

## Theoretical and Software Engineering Discussion

### 1. Neural Network Design (ASLVideoMAE)
Our core model, `ASLVideoMAE`, is built upon the `MCG-NJU/videomae-base` architecture loaded via HuggingFace Transformers. 
* **Theoretical Justification:** We opted for a Video Masked Autoencoder (VideoMAE) rather than a standard 3D-CNN because Vision Transformers (ViTs) excel at capturing long-range dependencies in sequence data. VideoMAE treats video as a sequence of tubelets, allowing the self-attention mechanism to heavily weight the relationships between spatial hand positions across different temporal frames.
* **Training Details:** The model was trained in PyTorch using CrossEntropyLoss and an `AdamW` optimizer (with weight decay) paired with a `CosineAnnealingLR` scheduler to ensure smooth convergence over 20 epochs.

### 2. Software Components & Pipeline
Our end-to-end system is broken down into modular Jupyter Notebooks:
* **Data Ingestion (`01_dataAnalysis.ipynb`):** We use the Voxel51/FiftyOne library to programmatically fetch and filter the WLASL (Word-Level ASL) dataset from HuggingFace, narrowing it down to the top 20 classes.
* **Data Preprocessing (`preprocessing.ipynb`):** Videos are processed using OpenCV (`cv2`) to extract exactly 16 uniform frames per video, resized to 224x224. We integrated **Google MediaPipe** to detect hand landmarks, ensuring the network emphasizes hand articulation.
* **Modeling & Inference (`models.ipynb` & `training.ipynb`):** The PyTorch inference engine takes the processed `(16, 3, 224, 224)` tensor and outputs the class probabilities. 

---
## End-to-End Application Pipeline
 
```
Raw .mp4 video
      ↓
Frame extraction (cv2) — 16 uniformly sampled frames
      ↓
Hand crop (MediaPipe) — isolate hand region per frame
      ↓
Resize + Normalize (torchvision transforms) → (16, 3, 224, 224) tensor
      ↓
ASLVideoMAE model forward pass
      ↓
Mean pool over patch tokens → (768,) feature vector
      ↓
MLP classifier → (20,) logits
      ↓
argmax → predicted gloss label
```
 
The `ASLDataset` PyTorch `Dataset` class handles per-sample loading and transformation. `DataLoader` handles batching and shuffling. The model is trained end-to-end with cross-entropy loss, AdamW optimizer, and cosine annealing LR scheduling.
 
---
 
## Repository Structure
 
```
├── 01_dataAnalysis.ipynb   # Dataset loading, exploration, and export via FiftyOne
├── preprocessing.ipynb      # Frame extraction, MediaPipe hand crop, dataset splits
├── models.ipynb             # ASLVideoMAE model definition
├── training.ipynb           # Training loop, validation, metrics
├── requirements.txt         # Python dependencies
└── README.md
```
 
---

## Setup and Installation
 
### Prerequisites
 
- Python 3.11+
- A CUDA-capable GPU is strongly recommended (the model was tested on CPU but training is very slow without GPU)
### Installation
 
```bash
git clone https://github.com/tonyranye/CSCI4052-Course-Project
cd CSCI4052-Course-Project
pip install -r requirements.txt
```
 
### Dataset Setup
 
The dataset is loaded from Hugging Face via FiftyOne. Run `01_dataAnalysis.ipynb` to download and export the top-20 WLASL classes to a local folder:
 
```
data/
└── wlasl_top20/
    ├── accident/
    ├── bar/
    ├── ...
    └── help/
```
 
> **Note:** FiftyOne's Hugging Face integration caps at 5,000 samples per download. The full WLASL dataset has 11,980 samples but the top-20 subset is well within this limit.
 
### Running the Notebooks
 
Run notebooks in this order:
 
1. `01_dataAnalysis.ipynb` — download and export dataset
2. `preprocessing.ipynb` — verify frame extraction and dataset splits
3. `models.ipynb` — verify model architecture with a dummy forward pass
4. `training.ipynb` — run the full training loop
> **Known issue:** MediaPipe's `mp.solutions` API requires `mediapipe >= 0.10`. If you encounter `AttributeError: module 'mediapipe' has no attribute 'solutions'`, upgrade mediapipe: `pip install --upgrade mediapipe`
 
---
 
## Dependencies
 
```
torch
torchvision
transformers
mediapipe
opencv-python
numpy
scikit-learn
matplotlib
fiftyone
```
 
---
 
## Deployment
 
The model runs locally. A GPU with at least 8GB VRAM is recommended for training. For inference only, a CPU is sufficient but slow. A Gradio or Streamlit demo interface is planned for the final submission to allow live webcam or video upload inference.
 
---
 
## Presentation and Video
 
- Slides: *(link to be added)*
- Demo video: *(YouTube link to be added)*
 
