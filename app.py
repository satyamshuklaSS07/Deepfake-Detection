
"""
Deepfake Detector Pro
Run: streamlit run app.py
"""

import json
import os
import tempfile
import time
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import detector as D


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detector Pro",
    page_icon="🕵️",
    layout="wide",
)

st.title("🕵️ Deepfake Detector Pro")
st.caption("CNN Fake Score | Grad-CAM | ELA | Video Analysis | Audit Log")


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------
st.sidebar.header("Settings")

model_path = st.sidebar.text_input("Model file", D.MODEL_PATH)

max_seconds = st.sidebar.slider(
    "Video ke pehle kitne seconds analyze karein",
    5, 120, 30
)

cnn_per_sec = st.sidebar.slider(
    "CNN frames per second", 1, 6, 3
)

fake_thr = st.sidebar.slider(
    "FAKE threshold", 0.50, 0.95, 0.65, 0.01
)

real_thr = st.sidebar.slider(
    "REAL threshold", 0.05, 0.50, 0.35, 0.01
)

st.sidebar.info(
    "REAL aur FAKE thresholds ke beech score aane par "
    "result UNCERTAIN ho sakta hai."
)


# --------------------------------------------------
# IMAGE DISPLAY HELPERS
# --------------------------------------------------
def to_uint8(image):
    """Convert image data into a valid uint8 array."""
    arr = np.asarray(image)

    if arr.size == 0:
        raise ValueError("Image array empty hai.")

    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr)

        if arr.min() >= 0 and arr.max() <= 1:
            arr = arr * 255

        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def image_to_pil(image, color_format="BGR"):
    """Convert NumPy image arrays into PIL images."""
    arr = to_uint8(image)

    if arr.ndim == 2:
        return Image.fromarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.shape[2] == 4:
        if color_format.upper() == "BGR":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(np.ascontiguousarray(arr))

    if arr.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if color_format.upper() == "BGR":
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(np.ascontiguousarray(arr))


def show_image(container, image, caption, color_format="BGR"):
    """Safely convert and display an image in Streamlit."""
    if image is None:
        container.warning(f"{caption}: image available nahi hai.")
        return

    try:
        pil_image = image_to_pil(image, color_format)

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        if not png_bytes:
            raise ValueError("PNG image create nahi hui.")

        container.image(
            png_bytes,
            caption=caption,
            use_container_width=True,
        )

    except Exception as error:
        container.error(f"{caption} display error: {error}")


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
if not os.path.isfile(model_path):
    st.error(
        f"Model file nahi mili: {model_path}\n\n"
        "Sidebar mein correct model path set karo."
    )
    st.stop()


@st.cache_resource
def load_model(path):
    return D.DeepfakeModel(path)


try:
    model = load_model(model_path)
except Exception as error:
    st.error("Model load nahi ho paya.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# RESULT BANNER
# --------------------------------------------------
def show_result(result):
    label = result.get("label", "UNCERTAIN")

    if label == "NO FACE":
        st.warning("Image/video mein face detect nahi hua.")
        return

    score = float(result.get("score", 0))
    fake_fraction = float(result.get("frac_fake", 0))

    message = (
        f"**{label}** | Fake score: {score:.2f} | "
        f"Fake frames: {fake_fraction * 100:.0f}%"
    )

    if label == "REAL":
        st.success(message)
    elif label == "FAKE":
        st.error(message)
    else:
        st.warning(message)

    for note in result.get("notes", []):
        st.caption(str(note))


# --------------------------------------------------
# TABS
# --------------------------------------------------
tab_video, tab_image, tab_info = st.tabs(
    ["🎬 Video Analysis", "🖼️ Image Analysis", "ℹ️ Information"]
)


# --------------------------------------------------
# VIDEO ANALYSIS
# --------------------------------------------------
with tab_video:
    uploaded_video = st.file_uploader(
        "Video upload karo",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        video_bytes = uploaded_video.getvalue()
        st.video(video_bytes)

        if st.button("Analyze Video", type="primary"):
            temp_path = None
            progress_bar = None

            try:
                suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as temp_file:
                    temp_file.write(video_bytes)
                    temp_path = temp_file.name

                progress_bar = st.progress(
                    0.0,
                    text="Video analyze ho raha hai..."
                )

                def update_progress(value):
                    value = max(0.0, min(1.0, float(value)))
                    progress_bar.progress(
                        value,
                        text=f"Analyzing video: {value * 100:.0f}%"
                    )

                result = D.analyze_video(
                    model,
                    temp_path,
                    max_seconds,
                    cnn_per_sec,
                    progress=update_progress,
                )

                probabilities = result.get("probs", [])

                if len(probabilities) == 0:
                    st.warning(
                        "Video mein koi face/frame analyze nahi hua. "
                        "Clear face wala doosra video try karo."
                    )

                else:
                    verdict = D.decide(
                        probabilities,
                        result.get("blink"),
                        fake_thr,
                        real_thr,
                    )

                    file_hash = D.sha256_bytes(video_bytes)

                    D.log_analysis(
                        uploaded_video.name,
                        file_hash,
                        verdict["label"],
                        verdict["score"],
                    )

                    show_result(verdict)

                    times = result.get("times", [])

                    if len(times) == len(probabilities):
                        frame_data = pd.DataFrame({
                            "time_s": np.round(times, 2),
                            "fake_probability": np.round(
                                probabilities, 4
                            ),
                        })

                        st.subheader("Fake Score Timeline")
                        st.line_chart(
                            frame_data.set_index("time_s")[
                                "fake_probability"
                            ]
                        )

                    else:
                        frame_data = pd.DataFrame({
                            "fake_probability": np.round(
                                probabilities, 4
                            )
                        })

                    fps = result.get("fps", 0)
                    ear_values = result.get("ear", [])

                    if fps and fps > 0 and len(ear_values) > 0:
                        ear_data = pd.DataFrame({
                            "time_s": np.arange(
                                len(ear_values)
                            ) / fps,
                            "EAR": [
                                np.nan if value is None else value
                                for value in ear_values
                            ],
                        })

                        st.subheader("Eye Aspect Ratio")
                        st.line_chart(
                            ear_data.set_index("time_s")["EAR"]
                        )

                    # Grad-CAM frames
                    st.subheader("Suspicious Frames + Grad-CAM")
                    st.caption(
                        "Grad-CAM model ke focus regions dikhata hai. "
                        "Ye akela deepfake hone ka proof nahi hai."
                    )

                    faces = result.get("faces", [])
                    probabilities_array = np.asarray(
                        probabilities, dtype=float
                    )

                    top_indices = np.argsort(
                        probabilities_array
                    )[::-1][:4]

                    columns = st.columns(max(1, len(top_indices)))

                    for column, index in zip(columns, top_indices):
                        if index >= len(faces):
                            column.warning("Frame image unavailable.")
                            continue

                        try:
                            explanation = model.explain(faces[index])

                            if explanation is None:
                                column.warning(
                                    "Grad-CAM available nahi hai."
                                )
                                continue

                            show_image(
                                column,
                                explanation,
                                (
                                    f"Time: {times[index]:.1f}s | "
                                    f"Fake score: "
                                    f"{probabilities_array[index]:.2f}"
                                ),
                                color_format="BGR",
                            )

                        except Exception as image_error:
                            column.warning(
                                f"Frame display error: {image_error}"
                            )

                    report = {
                        "file": uploaded_video.name,
                        "sha256": file_hash,
                        "time": time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "verdict": verdict,
                        "blink": result.get("blink"),
                        "settings": {
                            "max_seconds": max_seconds,
                            "cnn_per_sec": cnn_per_sec,
                            "fake_threshold": fake_thr,
                            "real_threshold": real_thr,
                        },
                    }

                    csv_column, json_column = st.columns(2)

                    csv_column.download_button(
                        "⬇️ Download CSV Report",
                        frame_data.to_csv(index=False).encode("utf-8"),
                        file_name="deepfake_frames.csv",
                        mime="text/csv",
                    )

                    json_column.download_button(
                        "⬇️ Download JSON Report",
                        json.dumps(
                            report,
                            indent=2,
                            default=str
                        ),
                        file_name="deepfake_report.json",
                        mime="application/json",
                    )

                    with st.expander("View Frame-wise Data"):
                        st.dataframe(
                            frame_data,
                            use_container_width=True,
                        )

            except Exception as error:
                st.error("Video analysis failed.")
                st.exception(error)

            finally:
                if progress_bar is not None:
                    progress_bar.empty()

                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------
with tab_image:
    uploaded_image = st.file_uploader(
        "Image upload karo",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()

        image_array = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image_array is None:
            st.error(
                "Image read nahi ho payi. Valid JPG ya PNG upload karo."
            )

        else:
            st.subheader("Uploaded Image")

            show_image(
                st,
                image_array,
                "Original Image",
                color_format="BGR",
            )

            try:
                image_result = D.analyze_image(
                    model,
                    image_array,
                )

                if not image_result.get("face_found", False):
                    show_result({"label": "NO FACE"})

                    st.info(
                        "Face detect nahi hua. Clear, front-facing "
                        "image try karo."
                    )

                else:
                    verdict = D.decide(
                        [float(image_result["fake_prob"])],
                        None,
                        fake_thr,
                        real_thr,
                    )

                    D.log_analysis(
                        uploaded_image.name,
                        D.sha256_bytes(image_bytes),
                        verdict["label"],
                        verdict["score"],
                    )

                    show_result(verdict)

                    st.subheader("Image Analysis Results")

                    face_column, cam_column, ela_column = st.columns(3)

                    with face_column:
                        st.markdown("**Detected Face**")
                        show_image(
                            st,
                            image_result.get("face"),
                            "Detected Face",
                            color_format="BGR",
                        )

                    with cam_column:
                        st.markdown("**Grad-CAM**")
                        show_image(
                            st,
                            image_result.get("cam"),
                            "Grad-CAM",
                            color_format="BGR",
                        )

                    with ela_column:
                        st.markdown("**Error Level Analysis**")
                        show_image(
                            st,
                            image_result.get("ela"),
                            "ELA",
                            color_format="RGB",
                        )

                    st.caption(
                        "Grad-CAM aur ELA supportive visual tools hain. "
                        "Ye akelay manipulation ka final proof nahi hain."
                    )

            except Exception as error:
                st.error("Image analysis failed.")
                st.exception(error)


# --------------------------------------------------
# INFORMATION TAB
# --------------------------------------------------
with tab_info:
    st.markdown(
        """
## How It Works

1. MediaPipe face detector face aur landmarks detect karta hai.
2. Face crop trained CNN model ko diya jata hai.
3. Video ke selected frames analyze kiye jate hain.
4. Grad-CAM model ke focus regions ko visualize karta hai.
5. ELA image compression differences ko highlight karne mein madad karta hai.
6. SHA-256 hash analysis file ko identify karne mein madad karta hai.

## Important Limitations

- Accuracy trained model aur dataset ki quality par depend karti hai.
- Poor lighting, compression aur side-face images results ko affect kar sakte hain.
- Blink detection akela deepfake identify nahi kar sakta.
- ELA akela image manipulation ka proof nahi hai.
- Results ko supporting evidence samjhein, final proof nahi.
"""
    )
