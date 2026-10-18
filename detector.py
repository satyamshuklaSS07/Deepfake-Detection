from pathlib import Path
"""
detector.py - Core engine for Deepfake Detector Pro

Isme ye sab hai:
  * FaceTracker      : MediaPipe FaceMesh se face + 468 landmarks
  * DeepfakeModel    : EfficientNet-B0 classifier + Grad-CAM explainability
  * analyze_video    : frame-wise fake score + blink (liveness) analysis
  * analyze_image    : single image analysis + Grad-CAM + ELA
  * decide           : final verdict (REAL / FAKE / UNCERTAIN)
  * log_analysis     : audit log (SHA-256 hash ke saath)
"""
import csv
import hashlib
import io
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageChops
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("DF_MODEL", "deepfake_model.pt")
LOG_DIR = "logs"

TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# MediaPipe FaceMesh landmark indices (EAR ke liye)
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

class FaceTracker:
    def __init__(self, static=False):
        model_path = str(Path(__file__).resolve().parent / "models" / "face_landmarker.task")

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=(
                vision.RunningMode.IMAGE
                if static
                else vision.RunningMode.VIDEO
            ),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.mesh = vision.FaceLandmarker.create_from_options(options)
        self.static = static
        self._timestamp_ms = 0

    def process(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        if self.static:
            result = self.mesh.detect(image)
        else:
            self._timestamp_ms += 33
            result = self.mesh.detect_for_video(
                image, self._timestamp_ms
            )

        if not result.face_landmarks:
            return None

        h, w = bgr.shape[:2]
        points = result.face_landmarks[0]
        return np.array(
            [[point.x * w, point.y * h] for point in points],
            dtype=np.float32
        )

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
        ckpt = torch.load(path, map_location=DEVICE)
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
