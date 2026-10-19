
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
from PIL import Image
import detector as D

# --------------------------------------------------
# Page configuration
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
# Sidebar settings
# --------------------------------------------------
st.sidebar.header("Settings")

model_path = st.sidebar.text_input("Model file", D.MODEL_PATH)
max_seconds = st.sidebar.slider(
    "Video ke pehle kitne seconds analyze karein", 5, 120, 30
)
cnn_per_sec = st.sidebar.slider("CNN frames per second", 1, 6, 3)
fake_thr = st.sidebar.slider(
    "FAKE threshold", 0.50, 0.95, 0.65, 0.01
)
real_thr = st.sidebar.slider(
    "REAL threshold", 0.05, 0.50, 0.35, 0.01
)

st.sidebar.info(
    "REAL threshold aur FAKE threshold ke beech score ho "
    "to result UNCERTAIN aa sakta hai."
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------
if not os.path.exists(model_path):
    st.error(
        f"Model file '{model_path}' nahi mili. "
        "Model file upload/add karo aur MODEL_PATH check karo."
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
# Result banner
# --------------------------------------------------
def banner(res):
    if res["label"] == "NO FACE":
        st.warning("Koi face nahi mila.")
        return

    msg = (
        f"**{res['label']}** | "
        f"Fake score: {res['score']:.2f} | "
        f"Fake frames: {res['frac_fake'] * 100:.0f}%"
    )

    {
        "FAKE": st.error,
        "REAL": st.success,
    }.get(res["label"], st.warning)(msg)

    for note in res.get("notes", []):
        st.caption("• " + note)


# --------------------------------------------------
# IMPORTANT: Create tabs before using them
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

                if not r["probs"]:
                    st.warning(
                        "Video se koi face/frame analyze nahi ho paya. "
                        "Doosra video try karo."
                    )
                else:
                    res = D.decide(
                        r["probs"],
                        r["blink"],
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

                    df = pd.DataFrame({
                        "time_s": np.round(r["times"], 2),
                        "fake_prob": np.round(r["probs"], 4),
                    })
                    st.subheader("Fake score timeline")
                    st.line_chart(df.set_index("time_s")["fake_prob"])

                    if r.get("fps", 0) > 0 and r.get("ear"):
                        ear = pd.DataFrame({
                            "time_s": (
                                np.arange(len(r["ear"])) / r["fps"]
                            ),
                            "EAR": [
                                np.nan if v is None else v
                                for v in r["ear"]
                            ],
                        })

                        st.subheader("Eye Aspect Ratio (blink graph)")
                        st.line_chart(ear.set_index("time_s")["EAR"])

                    st.subheader("Suspicious frames + Grad-CAM")
                    st.caption(
                        "Highlighted regions model ke focus ko dikhate hain; "
                        "ye apne aap mein fake hone ka proof nahi hain."
                    )

                    top = np.argsort(r["probs"])[::-1][:4]
                    cols = st.columns(len(top))

                    for col, idx in zip(cols, top):
                        explanation = model.explain(r["faces"][idx])

                        if explanation is None:
                            col.warning("Grad-CAM image unavailable")
                            continue

                        explanation = np.asarray(explanation)

                        if explanation.dtype != np.uint8:
                            if explanation.size and explanation.max() <= 1.0:
                                explanation = (
                                    explanation * 255
                                ).clip(0, 255).astype(np.uint8)
                            else:
                                explanation = explanation.clip(
                                    0, 255
                                ).astype(np.uint8)
                             
                        if explanation.ndim == 2:
                            col.image(
                                explanation,
                                caption=f"t={r['times'][idx]:.1f}s | fake {r['probs'][idx]:.2f}",
                                use_container_width=True,
                            )
                        else:
                            col.image(
                                cv2.cvtColor(explanation, cv2.COLOR_BGR2RGB),
                                caption=f"t={r['times'][idx]:.1f}s | fake {r['probs'][idx]:.2f}",
                                use_container_width=True,
                            )

                    report = {
                        "file": up.name,
                        "sha256": sha,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "verdict": res,
                        "blink": r["blink"],
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
            np.frombuffer(raw, np.uint8),
            cv2.IMREAD_COLOR,
        )

        if bgr is None:
            st.error("Image read nahi ho payi.")
        else:
            try:
                r = D.analyze_image(model, bgr)

                if not r["face_found"]:
                    banner({"label": "NO FACE"})
                    st.image(bgr, channels="BGR", caption="Original image")

                else:
                    res = D.decide(
                        [r["fake_prob"]],
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

                    c1, c2, c3 = st.columns(3)

                    face_rgb = cv2.cvtColor(
                        r["face"], cv2.COLOR_BGR2RGB
                    )
                    cam_rgb = cv2.cvtColor(
                        r["cam"], cv2.COLOR_BGR2RGB
                    )

                    face_img = Image.fromarray(
                        np.ascontiguousarray(face_rgb)
                    )
                    cam_img = Image.fromarray(
                        np.ascontiguousarray(cam_rgb)
                    )
                    ela_img = Image.fromarray(
                        np.ascontiguousarray(r["ela"])
                    )

                    st.write("Face shape:", r["face"].shape)
                    st.write("CAM shape:", r["cam"].shape)
                    st.write("ELA shape:", r["ela"].shape)

                    c1.image(face_img, caption="Detected face")
                    c2.image(cam_img, caption="Grad-CAM")
                    c3.image(ela_img, caption="Error Level Analysis (ELA)")
                except Exception as e:
                    st.error("Image analysis failed.")
                    st.exception(e)

# --------------------------------------------------
# INFORMATION TAB
# --------------------------------------------------
with tab_info:
    st.markdown("""
## How it works

1. MediaPipe FaceMesh face aur landmarks detect karta hai.
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
""")
