# Deepfake Detector Pro

Real vs Fake face detection: CNN + Grad-CAM + blink/liveness + ELA + audit log.

## Files

| File | Kaam |
|---|---|
| `detector.py` | Core engine (face tracking, model, Grad-CAM, blink, ELA, verdict, logging) |
| `train.py` | Video se frames nikalna aur model train karna |
| `app.py` | Streamlit dashboard (video + image upload) |
| `live_liveness.py` | Live webcam challenge-response liveness check |
| `requirements.txt` | Python packages |

## Setup

Python 3.10 ya 3.11 use karo (MediaPipe 0.10.14 ke liye).

```
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Dataset

Public datasets use karo: FaceForensics++, Celeb-DF, DFDC (Kaggle).

```
data/videos/real/*.mp4
data/videos/fake/*.mp4
```

Faces nikalo:

```
python train.py extract --videos data/videos/real --out data/frames/real
python train.py extract --videos data/videos/fake --out data/frames/fake
```

Tip: real aur fake frames ki sankhya lagbhag barabar rakho, aur kam se kam 100+ alag videos per class lo.

## Step 2: Train

```
python train.py train --data data/frames --epochs 8
```

- Split video ke hisaab se hota hai (data leakage nahi hota).
- Best model `deepfake_model.pt` mein save hota hai, metrics `metrics.json` mein.
- GPU na ho to Google Colab use karo.

## Step 3: Dashboard

```
streamlit run app.py
```

Video ya image upload karo. Result mein verdict, fake-score timeline, blink graph, Grad-CAM heatmap, ELA, aur CSV/JSON download milta hai.

## Step 4: Live liveness check

```
python live_liveness.py
```

Random order mein: do baar blink, sir left, sir right. Result `logs/liveness_log.csv` mein jaata hai, fail par snapshot bhi save hota hai.

## Limitations (zaruri padho)

- Model sirf un deepfake types ko pakadta hai jin par train hua. Naye generators par accuracy gir sakti hai.
- Heavy compression, kam roshni, side-face par results kharab ho sakte hain.
- Blink detection naye deepfakes ke khilaf kamzor signal hai.
- `live_liveness.py` mein CNN deepfake data par train hai. Phone/screen replay attacks pakadne ke liye anti-spoofing dataset (CelebA-Spoof, OULU-NPU, Replay-Attack) par alag se train karo.
- Ye sahayak tool hai, 100% saboot nahi.

## Privacy

Face data sensitive biometric data hai. Sirf consent wale ya public datasets use karo, aur `logs/` folder ko surakshit rakho.
