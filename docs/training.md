# Technical Documentation: YOLO-SAM Hybrid Inference Pipeline

## System Architecture Overview
This notebook implements a **zero-shot inference pipeline** fusing **Ultralytics YOLOv8** (object detection) with **Segment Anything Model (SAM)** (instance segmentation). The architecture follows a cascaded "Detect-then-Segment" paradigm: YOLO generates axis-aligned bounding boxes (`xyxy`) and class confidences; SAM consumes these boxes as prompts to produce high-fidelity binary masks. No model training, fine-tuning, or hyperparameter optimization occurs in this artifact; the workflow exclusively loads pre-trained weights (`.pt` files) for deployment.

## Core Components & Data Flow

### 1. `ImageProcessor` Class (Pipeline Orchestrator)
**Initialization (`__init__`)**
```python
def __init__(self, yolo_model_path: str, sam_model_path: str):
    self.yolo = YOLO(yolo_model_path)  # Loads YOLOv8 detection weights
    self.sam = SAM(sam_model_path)     # Loads SAM encoder/decoder weights
```
*   **Dependency**: `ultralytics` library handles weight loading, device placement (auto CUDA/MPS/CPU), and model compilation.
*   **State**: Stateless inference engines; no optimizer states, schedulers, or gradient buffers allocated.

### 2. Inference Pipeline (`process_image`)
**Input**: `np.ndarray` (BGR/HWC, OpenCV standard).
**Execution Flow**:
1.  **YOLO Forward Pass**: `results_yolo = self.yolo(image, verbose=False)`
    *   Output: `ultralytics.engine.results.Results` object containing `boxes