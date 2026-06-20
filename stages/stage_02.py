# Stage 2: feat: add YOLO model loading and basic inference
# ==================================================

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import SAM
from google.colab.patches import cv2_imshow
from IPython.display import display
import tempfile
from google.colab import files
import ipywidgets

# Upload SAM model manually here
MODEL_SAM_PATH = "sam2_b.pt"  # Replace with your SAM path if required

# Store model paths for each mode
MODEL_PATHS = {
    "Army": None,
    "Navy": None,
    "Airforce": None
}

# Dropdown for selecting mode
selected_mode = ipywidgets.Dropdown(
    options=["Army", "Navy", "Airforce"],
    value="Airforce",
    description="Select Mode:"
)

# Upload widgets for each model
army_upload = ipywidgets.FileUpload(accept=".pt", multiple=False)
navy_upload = ipywidgets.FileUpload(accept=".pt", multiple=False)
airforce_upload = ipywidgets.FileUpload(accept=".pt", multiple=False)

def handle_model_upload(change, mode):
    upload_widget = change['owner']
    if upload_widget.value:
        uploaded_model = list(upload_widget.value.values())[0]
        model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        model_file.write(uploaded_model['content'])
        model_file.close()
        MODEL_PATHS[mode] = model_file.name
        print(f"{mode} model uploaded and saved to: {model_file.name}")

army_upload.observe(lambda change: handle_model_upload(change, "Army"), names='value')
navy_upload.observe(lambda change: handle_model_upload(change, "Navy"), names='value')
airforce_upload.observe(lambda change: handle_model_upload(change, "Airforce"), names='value')

# ImageProcessor class
class ImageProcessor:
    def __init__(self, yolo_model_path: str, sam_model_path: str):
        self.yolo_model = YOLO(yolo_model_path)
        self.sam_model = SAM(sam_model_path)
        self.output_path = None

    def process_image(self, image: np.ndarray):
        results_yolo = self.yolo_model(image)
        boxes_yolo = []
        for result in results_yolo:
            class_ids = result.boxes.cls.int().tolist()
            if class_ids:
                boxes_yolo.extend(result.boxes.xyxy.tolist())

        results_sam = None
        if boxes_yolo:
            results_sam = self.sam_model(image, bboxes=boxes_yolo, verbose=False, save=False, device=0)

        return results_yolo, results_sam, boxes_yolo

    def visualize_results(self, image, results_yolo, results_sam, boxes_yolo):
        yolo_annotated = results_yolo[0].plot()
        cv2_imshow(yolo_annotated)

        if results_sam and isinstance(results_sam, list) and results_sam[0].masks is not None:
            masks = results_sam[0].masks.data.cpu().numpy()
            combined_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

            for mask in masks:
                combined_mask = np.maximum(combined_mask, (mask * 255).astype(np.uint8))

            cv2_imshow(combined_mask)

            masked_image = image.copy()
            masked_image[combined_mask > 0] = [0, 255, 0]
            combined_output = cv2.addWeighted(yolo_annotated, 0.7, masked_image, 0.3, 0)
            cv2_imshow(combined_output)
        else:
            print("No SAM masks available.")

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return

        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        self.output_path = temp_out.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results_yolo, results_sam, boxes_yolo = self.process_image(frame)
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

        cap.release()
        out.release()
        print("Video processing complete.")

# Upload widgets (visible only when user clicks buttons)
image_upload = ipywidgets.FileUpload(accept="image/*", multiple=False)
video_upload = ipywidgets.FileUpload(accept="video/*", multiple=False)
image_upload.layout.display = 'none'
video_upload.layout.display = 'none'

# Helper functions
def process_image_file(image_file: dict):
    img_bytes = image_file['content']
    img_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    selected_model_path = MODEL_PATHS[selected_mode.value]
    if selected_model_path is None:
        print(f"Please upload the {selected_mode.value} model before processing.")
        return

    image_processor = ImageProcessor(selected_model_path, MODEL_SAM_PATH)
    results_yolo, results_sam, boxes_yolo = image_processor.process_image(image)
    image_processor.visualize_results(image, results_yolo, results_sam, boxes_yolo)
    print("Image processing complete.")
    image_upload.layout.display = 'flex'
    video_upload.layout.display = 'flex'

def process_video_file(video_file: dict):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_video.name, "wb") as f:
        f.write(video_file['content'])

    selected_model_path = MODEL_PATHS[selected_mode.value]
    if selected_model_path is None:
        print(f"Please upload the {selected_mode.value} model before processing.")
        return

    image_processor = ImageProcessor(selected_model_path, MODEL_SAM_PATH)
    image_processor.process_video(temp_video.name)

    download_button = ipywidgets.Button(description="Download Processed Video", button_style='success')

    def on_download_clicked(b):
        files.download(image_processor.output_path)
        print("Download started. You can now upload another image or video.")
        image_upload.layout.display = 'flex'
        video_upload.layout.display = 'flex'

    download_button.on_click(on_download_clicked)
    display(download_button)

# Main UI function
def create_ui():
    print("Please upload the weights for each mode before uploading images or videos.")

    image_button = ipywidgets.Button(description="Upload Image", button_style='info')
    video_button = ipywidgets.Button(description="Upload Video", button_style='info')

    def on_image_upload(change):
        if image_upload.value:
            uploaded_image = list(image_upload.value.values())[0]
            process_image_file(uploaded_image)

    def on_video_upload(change):
        if video_upload.value:
            uploaded_video = list(video_upload.value.values())[0]
            process_video_file(uploaded_video)

    image_button.on_click(lambda x: setattr(image_upload.layout, 'display', 'flex'))
    video_button.on_click(lambda x: setattr(video_upload.layout, 'display', 'flex'))

    image_upload.observe(on_image_upload, names='value')
    video_upload.observe(on_video_upload, names='value')

    display(ipywidgets.VBox([
        ipywidgets.HTML(value="<h3>Upload YOLOv11 Weights for Each Mode</h3>"),
        ipywidgets.HBox([ipywidgets.Label("Army:"), army_upload]),
        ipywidgets.HBox([ipywidgets.Label("Navy:"), navy_upload]),
        ipywidgets.HBox([ipywidgets.Label("Airforce:"), airforce_upload]),
        ipywidgets.HTML(value="<hr><h3>Select Mode & Upload Media</h3>"),
        selected_mode,
        ipywidgets.HBox([image_button, video_button]),
        image_upload,
        video_upload
    ]))

# Run the UI
create_ui()