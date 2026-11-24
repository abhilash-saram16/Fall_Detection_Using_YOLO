import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Fall Detection | YOLO",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Fall Detection System")
st.caption("YOLO-powered detection of fall, human, and stand classes.")

MODEL_PATH = Path(__file__).parent / "best.pt"


# -------------------------------------------------
# Load trained YOLO model once
# -------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Place your trained best.pt file in the same "
        "folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the YOLO model: {exc}")
    st.stop()


# -------------------------------------------------
# Sidebar settings
# -------------------------------------------------
st.sidebar.header("Detection settings")
confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Higher values show only more confident detections.",
)
image_size = st.sidebar.select_slider(
    "Inference image size",
    options=[320, 416, 512, 640, 768, 960],
    value=640,
)
st.sidebar.info(
    "Classes in the supplied data.yaml: fall, human, stand."
)


# -------------------------------------------------
# Image detection
# -------------------------------------------------
def detect_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    results = model.predict(
        source=image_array,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            class_name = model.names.get(class_id, str(class_id))
            xyxy = [round(float(value), 1) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "Class": class_name,
                    "Confidence": f"{score:.1%}",
                    "Bounding box (x1, y1, x2, y2)": str(xyxy),
                }
            )

    return image, annotated_rgb, detections


# -------------------------------------------------
# Video detection
# -------------------------------------------------
def detect_video(uploaded_file, progress_bar, status_text):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path = input_temp.name
    output_path = output_temp.name
    input_temp.close()
    output_temp.close()

    try:
        with open(input_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise RuntimeError("Could not open the uploaded video.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 25.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise RuntimeError("The uploaded video has invalid dimensions.")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create the output video.")

        frame_number = 0
        fall_frames = 0
        detection_count = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(
                source=frame,
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )[0]

            annotated_frame = result.plot()
            writer.write(annotated_frame)
            frame_number += 1

            if result.boxes is not None and len(result.boxes) > 0:
                detection_count += len(result.boxes)
                for class_id in result.boxes.cls.tolist():
                    if model.names.get(int(class_id), "").lower() == "fall":
                        fall_frames += 1

            if total_frames > 0:
                progress_bar.progress(
                    min(frame_number / total_frames, 1.0)
                )
            status_text.write(f"Processed {frame_number} frames...")

        capture.release()
        writer.release()
        progress_bar.progress(1.0)
        status_text.success(f"Finished processing {frame_number} frames.")

        return output_path, {
            "Frames processed": frame_number,
            "Frames containing a fall detection": fall_frames,
            "Total detected objects across frames": detection_count,
        }

    except Exception:
        # Remove partial output if processing fails.
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


# -------------------------------------------------
# Main interface
# -------------------------------------------------
image_tab, video_tab, about_tab = st.tabs(
    ["🖼️ Image Detection", "🎥 Video Detection", "ℹ️ About"]
)

with image_tab:
    st.subheader("Upload an image")
    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
    )

    if image_file is not None:
        if st.button("Run image detection", type="primary"):
            with st.spinner("Running YOLO detection..."):
                try:
                    original, annotated, detections = detect_image(image_file)

                    left, right = st.columns(2)
                    with left:
                        st.markdown("**Original image**")
                        st.image(original, use_container_width=True)
                    with right:
                        st.markdown("**Detection result**")
                        st.image(annotated, use_container_width=True)

                    st.subheader("Detection summary")
                    if detections:
                        st.dataframe(
                            detections,
                            use_container_width=True,
                            hide_index=True,
                        )
                        result_image = Image.fromarray(annotated)
                        from io import BytesIO
                        buffer = BytesIO()
                        result_image.save(buffer, format="PNG")
                        st.download_button(
                            "⬇️ Download detected image",
                            data=buffer.getvalue(),
                            file_name="fall_detection_result.png",
                            mime="image/png",
                        )
                    else:
                        st.info("No objects were detected above the selected confidence threshold.")

                except Exception as exc:
                    st.error(f"Image detection failed: {exc}")

with video_tab:
    st.subheader("Upload a video")
    st.caption("Video processing runs frame by frame and may take time for long videos.")
    video_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    if video_file is not None:
        st.video(video_file)
        if st.button("Run video detection", type="primary"):
            progress = st.progress(0)
            status = st.empty()

            try:
                with st.spinner("Processing video frames..."):
                    output_path, summary = detect_video(
                        video_file, progress, status
                    )

                st.subheader("Processed video")
                with open(output_path, "rb") as output_file:
                    video_bytes = output_file.read()

                st.video(video_bytes)
                st.download_button(
                    "⬇️ Download processed video",
                    data=video_bytes,
                    file_name="fall_detection_result.mp4",
                    mime="video/mp4",
                )

                st.subheader("Video summary")
                st.dataframe(
                    [{"Metric": key, "Value": value} for key, value in summary.items()],
                    use_container_width=True,
                    hide_index=True,
                )

                os.remove(output_path)

            except Exception as exc:
                st.error(f"Video detection failed: {exc}")
                st.caption(
                    "If this happens, try a smaller MP4 video or check that "
                    "the video codec is supported."
                )

with about_tab:
    st.subheader("About this project")
    st.write(
        "This application uses your trained YOLO weights (best.pt) to detect "
        "three classes from your dataset."
    )
    st.markdown(
        """
        - **fall** — a person detected in a falling posture
        - **human** — a person detected by the model
        - **stand** — a person detected in a standing posture
        """
    )
    st.warning(
        "This is a computer-vision aid, not a certified emergency or medical "
        "monitoring system. Validate the model in real-world conditions before "
        "using it for safety-critical decisions."
    )

st.divider()
st.caption("Built with Streamlit and Ultralytics YOLO.")
