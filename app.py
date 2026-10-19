
"""
app.py - Deepfake Detector Pro
Run: streamlit run app.py
"""

import json
import os
import tempfile
import time

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
st.caption(
    "CNN fake-score + Grad-CAM + blink analysis + ELA + audit log"
)


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
    "CNN frames per second",
    1, 6, 3
)

fake_thr = st.sidebar.slider(
    "FAKE threshold",
    0.50, 0.95, 0.65, 0.01
)

real_thr = st.sidebar.slider(
    "REAL threshold",
    0.05, 0.50, 0.35, 0.01
)

st.sidebar.info(
    "REAL threshold aur FAKE threshold ke beech score ho "
    "to result UNCERTAIN aa sakta hai."
)


# --------------------------------------------------
# IMAGE CONVERSION HELPERS
# --------------------------------------------------
def to_uint8(image):
    """Convert an image-like value to a safe uint8 array."""
    arr = np.asarray(image)

    if arr.size == 0:
        raise ValueError("Image array empty hai.")

    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr)

        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = arr * 255.0

        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def bgr_to_pil(image):
    """Convert OpenCV BGR/grayscale image to PIL."""
    arr = to_uint8(image)

    if arr.ndim == 2:
        return Image.fromarray(arr)

    if arr.ndim == 3 and arr.shape[2] == 4:
        rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(np.ascontiguousarray(rgba))

    if arr.ndim == 3 and arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(np.ascontiguousarray(rgb))

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def rgb_to_pil(image):
    """Convert RGB/grayscale image to PIL."""
    arr = to_uint8(image)

    if arr.ndim == 2:
        return Image.fromarray(arr)

    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")

    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")

    raise ValueError(f"Unsupported RGB image shape: {arr.shape}")


def explanation_to_pil(image):
    """Convert OpenCV BGR Grad-CAM output to PIL RGB."""
    return bgr_to_pil(image)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
if not os.path.exists(model_path):
    st.error(
        f"Model file '{model_path}' nahi mili. "
        "Model file add karo aur MODEL_PATH check karo."
    )
    st.stop()


@st.cache_resource
def get_model(path):
    return D.DeepfakeModel(path)


try:
    model = get_model(model_path)
except Exception as e:
    st.error("Model load nahi ho paya.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# RESULT BANNER
# --------------------------------------------------
def banner(res):
    if res.get("label") == "NO FACE":
        st.warning("Koi face nahi mila.")
        return

    score = float(res.get("score", 0.0))
    frac_fake = float(res.get("frac_fake", 0.0))

    msg = (
        f"**{res.get('label', 'UNCERTAIN')}** | "
        f"Fake score: {score:.2f} | "
        f"Fake frames: {frac_fake * 100:.0f}%"
    )

    {
        "FAKE": st.error,
        "REAL": st.success,
    }.get(res.get("label"), st.warning)(msg)

    for note in res.get("notes", []):
        st.caption("• " + str(note))


# --------------------------------------------------
# TABS
# --------------------------------------------------
tab_video, tab_image, tab_info = st.tabs(
    ["🎬 Video", "🖼️ Image", "ℹ️ How it works"]
)


# --------------------------------------------------
# VIDEO TAB
# --------------------------------------------------
with tab_video:
    up = st.file_uploader(
        "Video upload karo",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="vid",
    )

    if up is not None:
        data = up.getvalue()
        st.video(data)

        if st.button("Analyze video", type="primary"):
            tmp_path = None
            bar = None

            try:
                suffix = os.path.splitext(up.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name

                bar = st.progress(0.0, text="Analyzing...")

                def update_progress(value):
                    value = min(1.0, max(0.0, float(value)))
                    bar.progress(value, text="Analyzing...")

                r = D.analyze_video(
                    model,
                    tmp_path,
                    max_seconds,
                    cnn_per_sec,
                    progress=update_progress,
                )

                if not r.get("probs"):
                    st.warning(
                        "Video se koi face/frame analyze nahi ho paya. "
                        "Doosra video try karo."
                    )

                else:
                    res = D.decide(
                        r["probs"],
                        r.get("blink"),
                        fake_thr,
                        real_thr,
                    )

                    sha = D.sha256_bytes(data)

                    D.log_analysis(
                        up.name,
                        sha,
                        res["label"],
                        res["score"],
                    )

                    banner(res)

                    df = pd.DataFrame(
                        {
                            "time_s": np.round(r["times"], 2),
                            "fake_prob": np.round(r["probs"], 4),
                        }
                    )

                    st.subheader("Fake score timeline")
                    st.line_chart(
                        df.set_index("time_s")["fake_prob"]
                    )

                    fps = r.get("fps", 0)
                    ear_values = r.get("ear", [])

                    if fps and fps > 0 and len(ear_values) > 0:
                        ear = pd.DataFrame(
                            {
                                "time_s": (
                                    np.arange(len(ear_values)) / fps
                                ),
                                "EAR": [
                                    np.nan if value is None else value
                                    for value in ear_values
                                ],
                            }
                        )

                        st.subheader("Eye Aspect Ratio (blink graph)")
                        st.line_chart(ear.set_index("time_s")["EAR"])

                    st.subheader("Suspicious frames + Grad-CAM")

                    st.caption(
                        "Highlighted regions model ke focus ko dikhate hain; "
                        "ye apne aap mein fake hone ka proof nahi hain."
                    )

                    probabilities = np.asarray(
                        r["probs"],
                        dtype=float,
                    )

                    top = np.argsort(probabilities)[::-1][:4]
                    faces = r.get("faces", [])
                    cols = st.columns(max(1, len(top)))

                    for col, idx in zip(cols, top):
                        if idx >= len(faces):
                            col.warning("Frame image unavailable")
                            continue

                        try:
                            explanation = model.explain(faces[idx])

                            if explanation is None:
                                col.warning("Grad-CAM image unavailable")
                                continue

                            col.image(
                                explanation_to_pil(explanation),
                                caption=(
                                    f"t={r['times'][idx]:.1f}s | "
                                    f"fake {probabilities[idx]:.2f}"
                                ),
                                use_container_width=True,
                            )

                        except Exception as image_error:
                            col.warning(
                                "Is frame ka Grad-CAM display nahi hua: "
                                f"{image_error}"
                            )

                    report = {
                        "file": up.name,
                        "sha256": sha,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "verdict": res,
                        "blink": r.get("blink"),
                        "settings": {
                            "max_seconds": max_seconds,
                            "cnn_per_sec": cnn_per_sec,
                            "fake_thr": fake_thr,
                            "real_thr": real_thr,
                        },
                    }

                    d1, d2 = st.columns(2)

                    d1.download_button(
                        "⬇️ Frame-wise CSV",
                        df.to_csv(index=False).encode("utf-8"),
                        "frames.csv",
                        "text/csv",
                    )

                    d2.download_button(
                        "⬇️ JSON report",
                        json.dumps(report, indent=2, default=str),
                        "report.json",
                        "application/json",
                    )

                    with st.expander("Frame-wise table"):
                        st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error("Video analysis failed.")
                st.exception(e)

            finally:
                if bar is not None:
                    bar.empty()

                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass


# --------------------------------------------------
# IMAGE TAB
# --------------------------------------------------
with tab_image:
    upi = st.file_uploader(
        "Image upload karo",
        type=["jpg", "jpeg", "png"],
        key="img",
    )

    if upi is not None:
        raw = upi.getvalue()

        bgr = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if bgr is None:
            st.error(
                "Image read nahi ho payi. JPG ya PNG image try karo."
            )

        else:
            st.subheader("Uploaded image")

            st.image(
                bgr_to_pil(bgr),
                caption="Original image",
                use_container_width=True,
            )

            try:
                r = D.analyze_image(model, bgr)

                if not r.get("face_found", False):
                    banner({"label": "NO FACE"})

                    st.info(
                        "Image mein face detect nahi hua. Clear, "
                        "front-facing image try karo."
                    )

                else:
                    res = D.decide(
                        [float(r["fake_prob"])],
                        None,
                        fake_thr,
                        real_thr,
                    )

                    D.log_analysis(
                        upi.name,
                        D.sha256_bytes(raw),
                        res["label"],
                        res["score"],
                    )

                    banner(res)

                    st.subheader("Image analysis")
                    c1, c2, c3 = st.columns(3)

                    # Face crop is OpenCV BGR.
                    if r.get("face") is not None:
                        c1.image(
                            bgr_to_pil(r["face"]),
                            caption="Detected face",
                            use_container_width=True,
                        )
                    else:
                        c1.warning("Detected face image unavailable")

                    # Grad-CAM output is OpenCV BGR.
                    if r.get("cam") is not None:
                        c2.image(
                            explanation_to_pil(r["cam"]),
                            caption="Grad-CAM",
                            use_container_width=True,
                        )
                    else:
                        c2.warning("Grad-CAM image unavailable")

                    # ELA output from this project is RGB.
                    if r.get("ela") is not None:
                        c3.image(
                            rgb_to_pil(r["ela"]),
                            caption="Error Level Analysis (ELA)",
                            use_container_width=True,
                        )
                    else:
                        c3.warning("ELA image unavailable")

                    st.caption(
                        "Grad-CAM aur ELA sirf supportive visual clues hain; "
                        "ye akela kisi image ke fake hone ka proof nahi hain."
                    )

            except Exception as e:
                st.error("Image analysis failed.")
                st.exception(e)


# --------------------------------------------------
# INFORMATION TAB
# --------------------------------------------------
with tab_info:
    st.markdown(
        """
## How it works

1. MediaPipe face detector face aur landmarks detect karta hai.
2. Face crop EfficientNet-B0 model ko diya jata hai.
3. Video frames mein eye movements aur blink patterns analyze hote hain.
4. Grad-CAM model ke focus regions dikhata hai.
5. ELA image compression differences ko highlight karne mein madad karta hai.
6. Analysis ka SHA-256 hash audit log mein record hota hai.

## Limitations

- Accuracy training data aur model quality par depend karti hai.
- Low light, compression aur side-face images results ko affect kar sakte hain.
- Blink detection akela deepfake identify nahi kar sakta.
- ELA akela image manipulation ka proof nahi hai.
- Results ko supporting evidence samjhein, final proof nahi.
"""
    )
