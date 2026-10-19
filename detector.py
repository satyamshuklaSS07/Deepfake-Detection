"""
app.py - Deepfake Detector Pro (SINGLE FILE: engine + Streamlit UI)
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
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

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

st.set_page_config(page_title="Deepfake Detector Pro", page_icon="🕵️", layout="wide")


# ------------------------------------------------------------------ helpers
def show_img(img, caption="", bgr=True, max_w=700):
    """Image ko base64 PNG bana kar HTML mein embed karta hai."""
    if img is None:
        st.warning(f"{caption}: image nahi mili")
        return
    img = np.ascontiguousarray(img).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif not bgr:  # RGB -> BGR (imencode BGR maangta hai)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        st.warning(f"{caption}: image encode nahi hui")
        return
    b64 = base64.b64encode(buf.tobytes()).decode()
    st.markdown(
        f'<div style="text-align:center">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:100%;height:auto;border-radius:8px">'
        f'<div style="color:gray;font-size:0.85em;margin-top:4px">{caption}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def banner(res):
    if res["label"] == "NO FACE":
        st.warning("Koi face nahi mila. Saaf, seedhe face wali image try karo.")
        return
    msg = (f"**{res['label']}** | Fake score: {res['score']:.2f} | "
           f"Fake frames: {res['frac_fake'] * 100:.0f}%")
    if res["label"] == "FAKE":
        st.error(msg)
    elif res["label"] == "REAL":
        st.success(msg)
    else:
        st.warning(msg)
    for n in res["notes"]:
        st.caption("• " + n)


@st.cache_resource(show_spinner="Model load ho raha hai...")
def get_model(path):
    return D.DeepfakeModel(path)


# ------------------------------------------------------------------ sidebar
st.title("🕵️ Deepfake Detector Pro")
st.caption("CNN fake-score + Grad-CAM + blink (liveness) analysis + ELA + audit log")

default_model = D.MODEL_PATH
if not os.path.isabs(default_model):
    cand = Path(__file__).resolve().parent / default_model
    if cand.exists():
        default_model = str(cand)

st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model file", default_model)
max_seconds = st.sidebar.slider("Video ke pehle kitne seconds analyze karein", 5, 120, 30)
cnn_per_sec = st.sidebar.slider("CNN frames per second", 1, 6, 3)
fake_thr = st.sidebar.slider("FAKE threshold", 0.50, 0.95, 0.65, 0.01)
real_thr = st.sidebar.slider("REAL threshold", 0.05, 0.50, 0.35, 0.01)
show_preview = st.sidebar.checkbox("Video preview dikhao", value=False)
st.sidebar.info("Score REAL aur FAKE threshold ke beech ho to result 'UNCERTAIN' aata hai.")

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' nahi mili. Pehle model train karke is folder mein rakho.")
    st.stop()

try:
    model = get_model(model_path)
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

tab_image, tab_video, tab_info = st.tabs(["🖼️ Image", "🎬 Video", "ℹ️ How it works"])

# ------------------------------------------------------------------ image tab
with tab_image:
    upi = st.file_uploader("Image upload karo", type=["jpg", "jpeg", "png", "webp"], key="img")
    if upi is not None:
        raw = upi.getvalue()
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Image padh nahi payi. Dusri image try karo.")
        else:
            h, w = bgr.shape[:2]
            if max(h, w) > 1600:  # bahut badi image slow hoti hai
                s = 1600.0 / max(h, w)
                bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

            st.subheader("Uploaded Image")
            show_img(bgr, "Original Image")

            try:
                r = D.analyze_image(model, bgr)
            except Exception as e:
                st.error("Analysis mein error aaya:")
                st.exception(e)
                st.stop()

            if not r["face_found"]:
                banner({"label": "NO FACE"})
                st.subheader("Image Analysis Results")
                show_img(r["ela"], "Error Level Analysis", bgr=False)
            else:
                res = D.decide([r["fake_prob"]], None, fake_thr, real_thr)
                D.log_analysis(upi.name, D.sha256_bytes(raw), res["label"], res["score"])
                banner(res)

                st.subheader("Image Analysis Results")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Detected Face**")
                    show_img(r["face"], "Detected Face", max_w=400)
                with c2:
                    st.markdown("**Grad-CAM**")
                    show_img(r["cam"], "Grad-CAM (model ka focus)", max_w=400)
                with c3:
                    st.markdown("**Error Level Analysis**")
                    show_img(r["ela"], "ELA", bgr=False, max_w=400)
                st.caption("Grad-CAM aur ELA supportive visual tools hain. "
                           "Ye akele manipulation ka final proof nahi hain.")

# ------------------------------------------------------------------ video tab
with tab_video:
    up = st.file_uploader("Video upload karo", type=["mp4", "avi", "mov", "mkv", "webm"], key="vid")
    if up is not None:
        data = up.getvalue()
        if show_preview:
            try:
                st.video(data)
            except Exception:
                st.caption("Video preview nahi dikh paya, analysis phir bhi chalega.")

        if st.button("Analyze video", type="primary"):
            suffix = os.path.splitext(up.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            bar = st.progress(0.0, text="Analyzing...")
            try:
                r = D.analyze_video(model, tmp_path, max_seconds, cnn_per_sec,
                                    progress=lambda v: bar.progress(min(max(v, 0.0), 1.0),
                                                                    text="Analyzing..."))
            except Exception as e:
                bar.empty()
                st.error("Video analysis mein error aaya:")
                st.exception(e)
                st.stop()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            bar.empty()

            res = D.decide(r["probs"], r["blink"], fake_thr, real_thr)
            sha = D.sha256_bytes(data)
            D.log_analysis(up.name, sha, res["label"], res["score"])
            banner(res)

            if r["probs"]:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Frames analyzed (CNN)", len(r["probs"]))
                c2.metric("Max fake score", f"{max(r['probs']):.2f}")
                c3.metric("Blinks", r["blink"]["blinks"])
                c4.metric("Blink rate / min", r["blink"]["rate_per_min"])
                st.caption("Normal insaan: lagbhag 8-30 blinks/min. Naye deepfakes blink bhi achhe se "
                           "kar lete hain, isliye blink sirf ek chhota signal hai.")

                df = pd.DataFrame({"time_s": np.round(r["times"], 2),
                                   "fake_prob": np.round(r["probs"], 4)})
                st.subheader("Fake score timeline")
                st.line_chart(df.set_index("time_s")["fake_prob"])

                ear = pd.DataFrame({
                    "time_s": np.arange(len(r["ear"])) / r["fps"],
                    "EAR": [np.nan if v is None else v for v in r["ear"]],
                })
                st.subheader("Eye Aspect Ratio (blink graph)")
                st.line_chart(ear.set_index("time_s")["EAR"])

                st.subheader("Sabse suspicious frames + Grad-CAM")
                st.caption("Laal/peela hissa = jahan model ko sabse zyada 'fake' ke nishaan mile.")
                top = np.argsort(r["probs"])[::-1][:4]
                cols = st.columns(len(top))
                for col, idx in zip(cols, top):
                    with col:
                        show_img(model.explain(r["faces"][idx]),
                                 f"t={r['times'][idx]:.1f}s | fake {r['probs'][idx]:.2f}",
                                 max_w=300)

                report = {
                    "file": up.name, "sha256": sha,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "verdict": res, "blink": r["blink"],
                    "settings": {"max_seconds": max_seconds, "cnn_per_sec": cnn_per_sec,
                                 "fake_thr": fake_thr, "real_thr": real_thr},
                }
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Frame-wise CSV", df.to_csv(index=False).encode(),
                                   "frames.csv", "text/csv")
                d2.download_button("⬇️ JSON report", json.dumps(report, indent=2, default=str),
                                   "report.json")
                with st.expander("Frame-wise table"):
                    st.dataframe(df)

# ------------------------------------------------------------------ info tab
with tab_info:
    st.markdown("""
**Pipeline**
1. MediaPipe FaceLandmarker face + landmarks nikalta hai.
2. Face crop (margin ke saath) EfficientNet-B0 ko jaata hai, jo *fake probability* deta hai.
3. Video mein har frame par eye-aspect-ratio (EAR) se blinks count hote hain.
4. Grad-CAM dikhata hai ki model ne face ke kis hisse ki wajah se fake kaha.
5. Final score = average fake probability (+0.10 agar lambe video mein ek bhi blink nahi).
6. Har analysis file ke SHA-256 ke saath `logs/analysis_log.csv` mein record hota hai.

**Limitations**
- Model sirf wahi deepfake types pakadta hai jin par train hua hai.
- Heavy compression, kam roshni ya side-face par accuracy kam ho sakti hai.
- Ye sahayak tool hai, 100% saboot nahi. Zaruri faisle ke liye insaani review karo.
""")
