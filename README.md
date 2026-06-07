# YOLO-SAM Segmentation Pipeline

An interactive computer vision pipeline combining **YOLO (You Only Look Once)** object detection with **SAM (Segment Anything Model)** for high-quality instance segmentation. Built for Google Colab with an interactive `ipywidgets` interface supporting both image and video processing.

---

## Overview

This project implements a two-stage vision pipeline:
1. **YOLOv11** detects objects and generates bounding boxes
2. **SAM** uses those boxes as prompts to produce precise segmentation masks

The notebook provides a complete GUI within Google Colab for uploading custom models, processing media files, visualizing results, and downloading annotated outputs — all without writing additional code.

---

## Features

| Capability | Description |
|------------|-------------|
| **Dual-Model Pipeline** | YOLO detection → SAM segmentation in a single workflow |
| **Model Flexibility** | Upload custom `.pt` weights for both YOLO and SAM via UI |
| **Image Processing** | Single-frame inference with mask overlay visualization |
| **Video Processing** | Frame-by-frame segmentation with MP4 output generation |
| **Interactive UI** | `ipywidgets`-based controls: file upload, model selection, download |
| **Colab-Native** | Leverages `cv2_imshow`, `files.upload()`, and GPU acceleration |
| **Result Export** | Download processed images/videos directly from the notebook |

---

## Installation

### Requirements
- Python 3.10+
- CUDA-enabled environment (recommended)
- Google Colab (primary target) or local Jupyter with widget support

### Quick Setup (Colab)
```bash
# Install core dependencies
!pip install -q ultralytics ipywidgets

# Enable widget extension (if needed)
!jupyter nbextension enable --py widgetsnbextension
```

### Local Installation
```bash
git clone https://github.com/yourusername/yolo-sam-pipeline.git
cd yolo-sam-pipeline
pip install -r requirements.txt
```

**`requirements.txt`**
```text
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
ipywidgets>=8.0.0
```

> **Note**: For local video I/O, ensure `ffmpeg` is installed (`apt-get install ffmpeg` or `brew install ffmpeg`).

---

## Usage

### 1. Launch the Notebook
Open `yolo_sam_pipeline.ipynb` in Google Colab or JupyterLab.

### 2. Start the UI
Execute the final cell to render the interface:
```python
create_ui()
```

### 3. Upload Models (Optional)
- Click **"Upload YOLO Model"** → select a `.pt` file (e.g., `yolov8n.pt`, `yolov8s-seg.pt`)
- Click **"Upload SAM Model"** → select a `.pt` file (e.g., `sam_b.pt`, `sam_l.pt`)
- Defaults to `yolov8n.pt` and `sam_b.pt` from Ultralytics Hub if skipped

### 4. Process Media
| Mode | Action |
|------|--------|
| **Image** | Click **"Upload Image"** → select file → results display automatically |
| **Video** | Click **"Upload Video"** → select file → processing runs frame-by-frame → download link appears |


