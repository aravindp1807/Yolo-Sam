# Technical Documentation: YOLO-SAM Object Detection & Segmentation Pipeline

## Environment Setup

### Dependencies
```bash
pip install ultralytics opencv-python ipywidgets numpy
```

### Google Colab Specific Requirements
- `google-colab` package for file upload/download widgets
- `cv2_imshow` patch for OpenCV display in Colab
- GPU runtime recommended (Runtime → Change runtime type → GPU)

### Model Weights
Download and place in working directory:
- **YOLOv8**: `yolov8n.pt` (nano) or `yolov8s.pt` (small) for detection
- **SAM**: `sam_b.pt` (base) or `sam_l.pt` (large) for segmentation

---

## Architecture Overview

### Core Class: `ImageProcessor`
```python
class ImageProcessor:
    def __init__(self, yolo_model_path: str, sam_model_path: str):
        self.yolo = YOLO(yolo_model_path)
        self.sam = SAM(sam_model_path)
    
    def process_image(self, image: np.ndarray) -> tuple:
        # YOLO inference → bounding boxes
        results_yolo = self.yolo(image)
        boxes_yolo = results_yolo[0].boxes.xyxy.cpu().numpy()
        
        # SAM inference → segmentation masks using YOLO boxes as prompts
        results_sam = self.sam(image, bboxes=boxes_yolo)
        return results_yolo, results_sam, boxes_yolo
    
    def process_video(self, video_path: str):
        # Frame-by-frame processing with cv2.VideoCapture
        # Outputs annotated video via cv2.VideoWriter
```

---

## API Usage

### Programmatic Interface
```python
processor = ImageProcessor("yolov8n.pt",