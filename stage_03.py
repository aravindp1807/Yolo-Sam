# Stage 3: feat: integrate SAM model for segmentation masks
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

# Go buttons
go_image_button = ipywidgets.Button(description="Go (Process Image)", button_style='success')
go_video_button = ipywidgets.Button(description="Go (Process Video)", button_style='success')
go_image_button.layout.display = 'none'
go_video_button.layout.display = 'none'


# Helper functions
def process_image_file_go(image_file: dict):
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
    go_image_button.layout.display = 'none' # Hide go button after processing
    image_upload.value = {} # Clear the upload widget content
    image_upload.layout.display = 'none' # Hide the upload widget

def process_video_file_go(video_file: dict):
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
        go_video_button.layout.display = 'none' # Hide go button after processing
        video_upload.value = {} # Clear the upload widget content
        video_upload.layout.display = 'none' # Hide the upload widget


    download_button.on_click(on_download_clicked)
    display(download_button)


# Main UI function
def create_ui():
    print("Please upload the weights for each mode before uploading images or videos.")

    image_button = ipywidgets.Button(description="Upload Image", button_style='info')
    video_button = ipywidgets.Button(description="Upload Video", button_style='info')

    # Function to handle image file upload and display Go button
    def on_image_upload_change(change):
        if image_upload.value:
            # Only display the Go button if a file has been selected
            go_image_button.layout.display = 'flex'
            print("Image uploaded. Click 'Go (Process Image)' to start analysis.")
        else:
            go_image_button.layout.display = 'none'

    # Function to handle video file upload and display Go button
    def on_video_upload_change(change):
        if video_upload.value:
            # Only display the Go button if a file has been selected
            go_video_button.layout.display = 'flex'
            print("Video uploaded. Click 'Go (Process Video)' to start analysis.")
        else:
            go_video_button.layout.display = 'none'

    image_upload.observe(on_image_upload_change, names='value')
    video_upload.observe(on_video_upload_change, names='value')

    def on_image_button_clicked(b):
        image_upload.layout.display = 'flex'
        video_upload.layout.display = 'none' # Hide video upload if image is chosen
        go_video_button.layout.display = 'none' # Hide video go button
        go_image_button.layout.display = 'none' # Hide image go button until file uploaded

    def on_video_button_clicked(b):
        video_upload.layout.display = 'flex'
        image_upload.layout.display = 'none' # Hide image upload if video is chosen
        go_image_button.layout.display = 'none' # Hide image go button
        go_video_button.layout.display = 'none' # Hide video go button until file uploaded


    image_button.on_click(on_image_button_clicked)
    video_button.on_click(on_video_button_clicked)

    # Attach the Go button click handlers
    go_image_button.on_click(lambda x: process_image_file_go(list(image_upload.value.values())[0]))
    go_video_button.on_click(lambda x: process_video_file_go(list(video_upload.value.values())[0]))


    display(ipywidgets.VBox([
        ipywidgets.HTML(value="<h3>Upload YOLOv11 Weights for Each Mode</h3>"),
        ipywidgets.HBox([ipywidgets.Label("Army:"), army_upload]),
        ipywidgets.HBox([ipywidgets.Label("Navy:"), navy_upload]),
        ipywidgets.HBox([ipywidgets.Label("Airforce:"), airforce_upload]),
        ipywidgets.HTML(value="<hr><h3>Select Mode & Upload Media</h3>"),
        selected_mode,
        ipywidgets.HBox([image_button, video_button]),
        image_upload,
        go_image_button, # Display the "Go" button for image
        video_upload,
        go_video_button # Display the "Go" button for video
    ]))

# Run the UI
create_ui()