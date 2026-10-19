"""
app.py - Deepfake Detector Pro (Streamlit)

Saari images base64 se seedha page mein embed hoti hain (st.image use nahi hota),
isliye Streamlit ke media server ki dikkat se "toota hua image icon" nahi aata.

Run: streamlit run app.py
"""
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

import detector as D

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
