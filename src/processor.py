import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import SAM
import tempfile
import os

class ImageProcessor:
    """
    ImageProcessor handles object detection and instance segmentation pipeline
    using YOLO (detect) and SAM (segment). It is framework-agnostic.
    """
    def __init__(self, yolo_model_path: str, sam_model_path: str):
        self.yolo_model = YOLO(yolo_model_path)
        self.sam_model = SAM(sam_model_path)
        self.output_path = None

    def process_image(self, image: np.ndarray, conf_threshold: float = 0.25):
        """
        Runs object detection on the image using YOLO, then feeds the resulting bounding
        boxes to SAM to perform high-precision instance segmentation.
        """
        # Run YOLO inference
        results_yolo = self.yolo_model(image, conf=conf_threshold)
        boxes_yolo = []
        for result in results_yolo:
            class_ids = result.boxes.cls.int().tolist()
            if class_ids:
                boxes_yolo.extend(result.boxes.xyxy.tolist())

        results_sam = None
        if boxes_yolo:
            # Run SAM inference using detected bounding boxes as prompts
            results_sam = self.sam_model(image, bboxes=boxes_yolo, verbose=False, save=False, device=0)

        return results_yolo, results_sam, boxes_yolo

    def visualize_results(self, image: np.ndarray, results_yolo, results_sam, boxes_yolo):
        """
        Applies bounding box drawings and segmentation mask overlays.
        Returns the annotated image as a numpy array.
        """
        # Plot YOLO annotated bounding boxes and labels
        yolo_annotated = results_yolo[0].plot()

        if results_sam and isinstance(results_sam, list) and results_sam[0].masks is not None:
            # Convert masks to numpy format
            masks = results_sam[0].masks.data.cpu().numpy()
            combined_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

            # Accumulate all instance masks
            for mask in masks:
                combined_mask = np.maximum(combined_mask, (mask * 255).astype(np.uint8))

            # Apply semi-transparent green mask overlay on the original image
            masked_image = image.copy()
            masked_image[combined_mask > 0] = [0, 255, 0]  # Green overlay
            combined_output = cv2.addWeighted(yolo_annotated, 0.7, masked_image, 0.3, 0)
            return combined_output
        else:
            return yolo_annotated

    def process_video(self, video_path: str, output_path: str = None, conf_threshold: float = 0.25, progress_callback=None):
        """
        Processes video frame-by-frame, applying detection + segmentation and exporting as MP4.
        Optional progress_callback is called with (current_frame, total_frames).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            self.output_path = temp_out.name
        else:
            self.output_path = output_path

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results_yolo, results_sam, boxes_yolo = self.process_image(frame, conf_threshold=conf_threshold)
                frame_plot = results_yolo[0].plot()

                if results_sam and isinstance(results_sam, list) and results_sam[0].masks is not None:
                    masks = results_sam[0].masks.data.cpu().numpy()
                    combined_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
                    for mask in masks:
                        combined_mask = np.maximum(combined_mask, (mask * 255).astype(np.uint8))

                    masked_frame = frame.copy()
                    masked_frame[combined_mask > 0] = [0, 255, 0]
                    frame_plot = cv2.addWeighted(frame_plot, 0.7, masked_frame, 0.3, 0)

                out.write(frame_plot)
                frame_count += 1
                if progress_callback:
                    progress_callback(frame_count, total_frames)
        finally:
            cap.release()
            out.release()

        return self.output_path
