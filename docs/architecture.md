# Technical Documentation: YOLO-SAM Segmentation Pipeline

## System Architecture

The application follows a **modular pipeline architecture** deployed as a Google Colab notebook with an interactive `ipywidgets` frontend. The core design separates **model orchestration** (`ImageProcessor` class) from **I/O handling** and **UI event binding**, enabling hot-swapping of model weights without restructuring inference logic.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  UI Layer       │────▶│  Controller      │────▶│  Processing     │
│  (ipywidgets)   │     │  (Callbacks)     │     │  Core           │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                           ┌──────────────────────────────┼──────────────────┐
                           ▼                              ▼                  ▼
                    ┌───────────────┐            ┌───────────────┐  ┌───────────────┐
                    │  YOLOv8       │            │  SAM          │  │  Visualizer   │
                    │  (Detection)  │───────────▶│  (Segmentation)│  │  (OpenCV)     │
                    └───────────────┘  boxes     └───────────────┘  └───────────────┘
```

## Core Components

### `ImageProcessor` (Dual-Model Orchestrator)
- **`__init__(yolo_model_path, sam_model_path)`**: Lazy-loads `ultralytics.YOLO` and `ultralytics.SAM` instances. Accepts local paths or Ultralytics Hub identifiers.
- **`process_image