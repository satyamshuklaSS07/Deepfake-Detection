"""
detector.py - Core engine for Deepfake Detector Pro

Isme ye sab hai:
  * FaceTracker      : MediaPipe FaceLandmarker (Tasks API) se face + landmarks
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
    """face_landmarker.task file na ho to apne aap download kar leta
