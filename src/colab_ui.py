import cv2
import numpy as np
import tempfile
import ipywidgets
from IPython.display import display

# Try importing google.colab specific elements
try:
    from google.colab.patches import cv2_imshow
    from google.colab import files
    HAS_COLAB = True
except ImportError:
    HAS_COLAB = False

from src.processor import ImageProcessor

# Store model paths for each mode
MODEL_PATHS = {
    "Army": None,
    "Navy": None,
    "Airforce": None
}

MODEL_SAM_PATH = "sam2_b.pt"  # Default SAM model path

# UI Widgets
selected_mode = ipywidgets.Dropdown(
    options=["Army", "Navy", "Airforce"],
    value="Airforce",
    description="Select Mode:"
)

# Upload widgets for models
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

# Media upload widgets
image_upload = ipywidgets.FileUpload(accept="image/*", multiple=False)
video_upload = ipywidgets.FileUpload(accept="video/*", multiple=False)
image_upload.layout.display = 'none'
video_upload.layout.display = 'none'

# Go buttons
go_image_button = ipywidgets.Button(description="Go (Process Image)", button_style='success')
go_video_button = ipywidgets.Button(description="Go (Process Video)", button_style='success')
go_image_button.layout.display = 'none'
go_video_button.layout.display = 'none'

def show_image(image):
    if HAS_COLAB:
        cv2_imshow(image)
    else:
        # Fallback to saving and displaying via matplotlib if run locally
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        except ImportError:
            # If matplotlib is not installed, write to file
            cv2.imwrite("output.png", image)
            print("Output saved to output.png (matplotlib not installed for inline visualization)")

def process_image_file_go(image_file: dict):
    img_bytes = image_file['content']
    img_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    selected_model_path = MODEL_PATHS[selected_mode.value]
    if selected_model_path is None:
        print(f"Please upload the {selected_mode.value} model before processing.")
        return

    # Use the modular ImageProcessor
    processor = ImageProcessor(selected_model_path, MODEL_SAM_PATH)
    results_yolo, results_sam, boxes_yolo = processor.process_image(image)
    
    # Generate visualization array and show it
    annotated = processor.visualize_results(image, results_yolo, results_sam, boxes_yolo)
    show_image(annotated)
    
    print("Image processing complete.")
    go_image_button.layout.display = 'none'
    image_upload.value = {}
    image_upload.layout.display = 'none'

def process_video_file_go(video_file: dict):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_video.name, "wb") as f:
        f.write(video_file['content'])

    selected_model_path = MODEL_PATHS[selected_mode.value]
    if selected_model_path is None:
        print(f"Please upload the {selected_mode.value} model before processing.")
        return

    processor = ImageProcessor(selected_model_path, MODEL_SAM_PATH)
    
    # Track progress
    def progress_callback(curr, total):
        if curr % max(1, total // 10) == 0 or curr == total:
            print(f"Processing progress: {curr}/{total} frames ({(curr/total)*100:.1f}%)")

    processor.process_video(temp_video.name, progress_callback=progress_callback)

    download_button = ipywidgets.Button(description="Download Processed Video", button_style='success')

    def on_download_clicked(b):
        if HAS_COLAB:
            files.download(processor.output_path)
            print("Download started. You can now upload another image or video.")
        else:
            print(f"Video saved locally at: {processor.output_path}")
        go_video_button.layout.display = 'none'
        video_upload.value = {}
        video_upload.layout.display = 'none'

    download_button.on_click(on_download_clicked)
    display(download_button)

def create_ui():
    print("Please upload the weights for each mode before uploading images or videos.")

    image_button = ipywidgets.Button(description="Upload Image", button_style='info')
    video_button = ipywidgets.Button(description="Upload Video", button_style='info')

    def on_image_upload_change(change):
        if image_upload.value:
            go_image_button.layout.display = 'flex'
            print("Image uploaded. Click 'Go (Process Image)' to start analysis.")
        else:
            go_image_button.layout.display = 'none'

    def on_video_upload_change(change):
        if video_upload.value:
            go_video_button.layout.display = 'flex'
            print("Video uploaded. Click 'Go (Process Video)' to start analysis.")
        else:
            go_video_button.layout.display = 'none'

    image_upload.observe(on_image_upload_change, names='value')
    video_upload.observe(on_video_upload_change, names='value')

    def on_image_button_clicked(b):
        image_upload.layout.display = 'flex'
        video_upload.layout.display = 'none'
        go_video_button.layout.display = 'none'
        go_image_button.layout.display = 'none'

    def on_video_button_clicked(b):
        video_upload.layout.display = 'flex'
        image_upload.layout.display = 'none'
        go_image_button.layout.display = 'none'
        go_video_button.layout.display = 'none'

    image_button.on_click(on_image_button_clicked)
    video_button.on_click(on_video_button_clicked)

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
        go_image_button,
        video_upload,
        go_video_button
    ]))
