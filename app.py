"""
Deepfake Detector Pro - SINGLE FILE (engine + Streamlit UI)
Is file ke saath alag detector.py ki zarurat nahi hai.
Run: streamlit run app.py
"""
import csv
import hashlib
import io
import os
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageChops
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("DF_MODEL", "deepfake_model.pt")
LOG_DIR = "logs"

LANDMARKER_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# FaceLandmarker landmark indices (EAR ke liye)
R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]


# ------------------------------------------------------------------ model
def build_net(pretrained=False):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
    return net.to(DEVICE)


def prep(bgr):
    return TFM(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


# ------------------------------------------------------------------ face
def ensure_landmarker():
    """face_landmarker.task file na ho to apne aap download kar leta hai."""
    if LANDMARKER_PATH.exists():
        return str(LANDMARKER_PATH)
    LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(LANDMARKER_URL, str(LANDMARKER_PATH))
    except Exception as e:
        raise FileNotFoundError(
            f"face_landmarker.task nahi mili aur download bhi fail hua ({e}). "
            f"Is file ko {LANDMARKER_PATH} par rakho. Link: {LANDMARKER_URL}"
        )
    return str(LANDMARKER_PATH)


class FaceTracker:
    """process() -> (N,2) pixel landmarks ya None."""

    def __init__(self, static=False):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_landmarker()),
            running_mode=vision.RunningMode.IMAGE if static else vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mesh = vision.FaceLandmarker.create_from_options(options)
        self.static = static
        self._timestamp_ms = 0

    def process(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.static:
            result = self.mesh.detect(image)
        else:
            self._timestamp_ms += 33  # timestamp badhta rehna zaruri hai
            result = self.mesh.detect_for_video(image, self._timestamp_ms)

        if not result.face_landmarks:
            return None

        h, w = bgr.shape[:2]
        points = result.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in points], dtype=np.float32)

    def close(self):
        self.mesh.close()


def crop_from_pts(bgr, pts, margin=0.25):
    """Landmarks se face crop (margin ke saath, kinare deepfake ke liye important hain)."""
    h, w = bgr.shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    bw, bh = x1 - x0, y1 - y0
    x0 = int(max(x0 - margin * bw, 0))
    y0 = int(max(y0 - margin * bh, 0))
    x1 = int(min(x1 + margin * bw, w))
    y1 = int(min(y1 + margin * bh, h))
    return bgr[y0:y1, x0:x1], (x0, y0, x1, y1)


def eye_aspect_ratio(eye):
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    return float((a + b) / (2.0 * c + 1e-8))


def mean_ear(pts):
    return (eye_aspect_ratio(pts[R_EYE]) + eye_aspect_ratio(pts[L_EYE])) / 2.0


# ------------------------------------------------------------------ gradcam
class GradCAM:
    def __init__(self, net, layer):
        self.net = net
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, cls_idx):
        self.net.zero_grad()
        with torch.enable_grad():
            out = self.net(x)
            out[0, cls_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1)).squeeze(0).detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam


def overlay_cam(face_bgr, cam):
    face = cv2.resize(face_bgr, (224, 224))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(face, 0.55, heat, 0.45, 0)


class DeepfakeModel:
    def __init__(self, path=MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file nahi mili: {path}")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        self.net = build_net(pretrained=False)
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        self.fake_idx = ckpt["classes"]["fake"]
        self.cam = GradCAM(self.net, self.net.features[-1])

    def predict(self, faces, batch=16):
        """faces: list of BGR crops -> list of fake probabilities (0..1)."""
        out = []
        with torch.no_grad():
            for i in range(0, len(faces), batch):
                x = torch.stack([prep(f) for f in faces[i:i + batch]]).to(DEVICE)
                p = torch.softmax(self.net(x), 1)[:, self.fake_idx]
                out += p.cpu().tolist()
        return out

    def explain(self, face_bgr):
        """Grad-CAM heatmap overlay: model ne face ka kaun sa hissa 'fake' maana."""
        x = prep(face_bgr).unsqueeze(0).to(DEVICE)
        cam = self.cam(x, self.fake_idx)
        return overlay_cam(face_bgr, cam)


# ------------------------------------------------------------------ blink
def count_blinks(ear_series, fps):
    e = np.array([np.nan if v is None else v for v in ear_series], dtype=float)
    valid = e[~np.isnan(e)]
    duration = len(valid) / fps if fps else 0.0
    if len(valid) < 10:
        return {"blinks": 0, "duration_s": duration, "rate_per_min": 0.0,
                "threshold": None, "no_blink_flag": False}

    thr = 0.75 * float(np.median(valid))          # har insaan ke liye adaptive
    min_run = max(2, int(0.05 * fps))             # >= ~50ms band aankh
    max_run = max(min_run + 1, int(0.6 * fps))    # >0.6s = aankh band, blink nahi
    blinks, run = 0, 0
    for v in e:
        if not np.isnan(v) and v < thr:
            run += 1
        else:
            if min_run <= run <= max_run:
                blinks += 1
            run = 0
    rate = blinks / duration * 60 if duration > 0 else 0.0
    return {
        "blinks": blinks,
        "duration_s": round(duration, 1),
        "rate_per_min": round(rate, 1),
        "threshold": round(thr, 3),
        "no_blink_flag": bool(duration >= 8 and blinks == 0),
    }


# ------------------------------------------------------------------ video
def analyze_video(model, path, max_seconds=30, cnn_per_sec=3, progress=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(max_seconds * fps)
    limit = min(total, int(max_seconds * fps))
    step = max(int(fps / max(cnn_per_sec, 1)), 1)

    tracker = FaceTracker()
    ear_series, faces, times = [], [], []
    i = 0
    while i < limit:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = 640.0 / w if w > 640 else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        pts = tracker.process(small)
        if pts is None:
            ear_series.append(None)
        else:
            ear_series.append(mean_ear(pts))
            if i % step == 0:
                face, _ = crop_from_pts(frame, pts / scale)
                if face.size and min(face.shape[:2]) >= 40:
                    faces.append(face)
                    times.append(i / fps)
        i += 1
        if progress and i % 5 == 0:
            progress(min(i / max(limit, 1), 1.0) * 0.9)
    cap.release()
    tracker.close()

    probs = model.predict(faces) if faces else []
    if progress:
        progress(1.0)
    return {
        "fps": fps,
        "frames_read": i,
        "times": times,
        "probs": probs,
        "faces": faces,
        "ear": ear_series,
        "blink": count_blinks(ear_series, fps),
    }


# ------------------------------------------------------------------ image
def ela_image(bgr, quality=90):
    """Error Level Analysis: edited/pasted regions JPEG recompress par alag dikhte hain."""
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    diff = np.array(ImageChops.difference(pil, Image.open(buf).convert("RGB")), dtype=np.float32)
    diff *= 255.0 / max(diff.max(), 1.0)
    return diff.astype(np.uint8)  # RGB


def analyze_image(model, bgr):
    tracker = FaceTracker(static=True)
    pts = tracker.process(bgr)
    tracker.close()
    out = {"ela": ela_image(bgr)}
    if pts is None:
        out["face_found"] = False
        return out
    face, box = crop_from_pts(bgr, pts)
    out.update({
        "face_found": True,
        "face": face,
        "box": box,
        "fake_prob": model.predict([face])[0],
        "cam": model.explain(face),
    })
    return out


# ------------------------------------------------------------------ verdict
def decide(probs, blink=None, fake_thr=0.65, real_thr=0.35):
    if not probs:
        return {"label": "NO FACE", "score": None, "frac_fake": None, "notes": ["Koi face nahi mila"]}
    arr = np.array(probs, dtype=float)
    score = float(arr.mean())
    notes = []
    if blink and blink.get("no_blink_flag"):
        score = min(1.0, score + 0.10)
        notes.append("Lambe video mein ek bhi blink nahi mila (+0.10 fake score)")
    label = "FAKE" if score >= fake_thr else "REAL" if score <= real_thr else "UNCERTAIN"
    return {
        "label": label,
        "score": round(score, 3),
        "frac_fake": round(float((arr > 0.5).mean()), 3),
        "notes": notes,
    }


# ------------------------------------------------------------------ logging
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def log_analysis(filename, sha, label, score):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "analysis_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["time", "file", "sha256", "label", "score"])
        wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), filename, sha, label, score])


# ==================================================================
# STREAMLIT UI
# ==================================================================
import base64
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

import types

D = types.SimpleNamespace(
    MODEL_PATH=MODEL_PATH,
    DeepfakeModel=DeepfakeModel,
    analyze_image=analyze_image,
    analyze_video=analyze_video,
    decide=decide,
    log_analysis=log_analysis,
    sha256_bytes=sha256_bytes,
)


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

show_preview = st.sidebar.checkbox("Video preview dikhao", value=False)

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
    """Image ko base64 PNG bana kar seedha HTML mein dikhata hai (media server use nahi hota)."""
    if image is None:
        container.warning(f"{caption}: image available nahi hai.")
        return

    try:
        pil_image = image_to_pil(image, color_format)

        if pil_image.width > 900:
            ratio = 900 / pil_image.width
            pil_image = pil_image.resize((900, int(pil_image.height * ratio)))

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()

        container.markdown(
            f'<div style="text-align:center">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%;height:auto;border-radius:8px">'
            f'<div style="color:gray;font-size:0.85em;margin-top:4px">{caption}</div>'
            f'</div>',
            unsafe_allow_html=True,
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

        if show_preview:
            try:
                st.video(video_bytes)
            except Exception:
                st.caption("Video preview nahi dikh paya, analysis phir bhi chalega.")

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
                    st.subheader("Highest-Score Frames + Grad-CAM")
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
                        st.dataframe(frame_data)

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
