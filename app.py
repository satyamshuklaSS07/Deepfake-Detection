"""
app.py - Streamlit dashboard
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

import detector as D

st.set_page_config(page_title="Deepfake Detector Pro", page_icon="🕵️", layout="wide")
st.title("🕵️ Deepfake Detector Pro")
st.caption("CNN fake-score + Grad-CAM + blink (liveness) analysis + ELA + audit log")

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model file", D.MODEL_PATH)
max_seconds = st.sidebar.slider("Video ke pehle kitne seconds analyze karein", 5, 120, 30)
cnn_per_sec = st.sidebar.slider("CNN frames per second", 1, 6, 3)
fake_thr = st.sidebar.slider("FAKE threshold", 0.50, 0.95, 0.65, 0.01)
real_thr = st.sidebar.slider("REAL threshold", 0.05, 0.50, 0.35, 0.01)
st.sidebar.info("Score REAL threshold aur FAKE threshold ke beech ho to result 'UNCERTAIN' aata hai.")

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' nahi mili. Pehle `python train.py train ...` chalao.")
    st.stop()


@st.cache_resource
def get_model(path):
    return D.DeepfakeModel(path)


model = get_model(model_path)


def banner(res):
    if res["label"] == "NO FACE":
        st.warning("Koi face nahi mila.")
        return
    msg = f"**{res['label']}** | fake score: {res['score']:.2f} | fake frames: {res['frac_fake'] * 100:.0f}%"
    {"FAKE": st.error, "REAL": st.success}.get(res["label"], st.warning)(msg)
    for n in res["notes"]:
        st.caption("• " + n)


tab_video, tab_image, tab_info = st.tabs(["🎬 Video", "🖼️ Image", "ℹ️ How it works"])

# ------------------------------------------------------------------ video tab
with tab_video:
    up = st.file_uploader("Video upload karo", type=["mp4", "avi", "mov", "mkv", "webm"], key="vid")
    if up is not None:
        data = up.getvalue()
        st.video(data)
        if st.button("Analyze video", type="primary"):
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1], delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            bar = st.progress(0.0, text="Analyzing...")
            try:
                r = D.analyze_video(model, tmp_path, max_seconds, cnn_per_sec,
                                    progress=lambda v: bar.progress(v, text="Analyzing..."))
            finally:
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
                st.caption("Normal insaan: lagbhag 8-30 blinks/min. Naye deepfakes blink bhi achhe se kar lete hain, "
                           "isliye blink sirf ek chhota signal hai.")

                df = pd.DataFrame({"time_s": np.round(r["times"], 2), "fake_prob": np.round(r["probs"], 4)})
                st.subheader("Fake score timeline")
                st.line_chart(df.set_index("time_s")["fake_prob"])

                ear = pd.DataFrame({"time_s": np.arange(len(r["ear"])) / r["fps"],
                                    "EAR": [np.nan if v is None else v for v in r["ear"]]})
                st.subheader("Eye Aspect Ratio (blink graph)")
                st.line_chart(ear.set_index("time_s")["EAR"])

                st.subheader("Sabse suspicious frames + Grad-CAM")
                st.caption("Laal/peela hissa = jahan model ko sabse zyada 'fake' ke nishaan mile.")
                top = np.argsort(r["probs"])[::-1][:4]
                cols = st.columns(len(top))
                for col, idx in zip(cols, top):
                    col.image(model.explain(r["faces"][idx]), channels="BGR",
                              caption=f"t={r['times'][idx]:.1f}s | fake {r['probs'][idx]:.2f}")

                report = {
                    "file": up.name, "sha256": sha, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "verdict": res, "blink": r["blink"],
                    "settings": {"max_seconds": max_seconds, "cnn_per_sec": cnn_per_sec,
                                 "fake_thr": fake_thr, "real_thr": real_thr},
                }
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Frame-wise CSV", df.to_csv(index=False).encode(), "frames.csv", "text/csv")
                d2.download_button("⬇️ JSON report", json.dumps(report, indent=2, default=str), "report.json")
                with st.expander("Frame-wise table"):
                    st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------------ image tab
with tab_image:
    upi = st.file_uploader("Image upload karo", type=["jpg", "jpeg", "png"], key="img")
    if upi is not None:
        raw = upi.getvalue()
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Image padh nahi payi.")
        else:
            r = D.analyze_image(model, bgr)
            if not r["face_found"]:
                banner({"label": "NO FACE"})
                st.image(bgr, channels="BGR", caption="Original")
            else:
                res = D.decide([r["fake_prob"]], None, fake_thr, real_thr)
                D.log_analysis(upi.name, D.sha256_bytes(raw), res["label"], res["score"])
                banner(res)
                c1, c2, c3 = st.columns(3)
                c1.image(r["face"], channels="BGR", caption="Detected face")
                c2.image(r["cam"], channels="BGR", caption="Grad-CAM (model ka focus)")
                c3.image(r["ela"], caption="ELA (edited hisse zyada chamakte hain)")
                st.caption("ELA ek purani forensic technique hai; heavily compressed ya social-media images par "
                           "false signals de sakti hai. Isse sirf supporting evidence samjho.")

# ------------------------------------------------------------------ info tab
with tab_info:
    st.markdown("""
**Pipeline**
1. MediaPipe FaceMesh face + 468 landmarks nikalta hai.
2. Face crop (margin ke saath) EfficientNet-B0 ko jaata hai, jo *fake probability* deta hai.
3. Video mein har frame par eye-aspect-ratio (EAR) se blinks count hote hain.
4. Grad-CAM dikhata hai ki model ne face ke kis hisse ki wajah se fake kaha.
5. Final score = average fake probability (+0.10 agar lambe video mein ek bhi blink nahi).
6. Har analysis file ke SHA-256 ke saath `logs/analysis_log.csv` mein record hota hai.

**Limitations**
- Model sirf wahi deepfake types pakadta hai jin par train hua hai.
- Heavy compression, low light ya side-face par accuracy kam ho sakti hai.
- Ye aid hai, 100% proof nahi. Important faisle ke liye insaani review zaruri hai.
""")
