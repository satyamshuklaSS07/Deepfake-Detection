"""
live_liveness.py - Live webcam liveness check (challenge-response)

Run: python live_liveness.py
Keys: r = restart, q = quit

Kaise kaam karta hai:
  1. 1 second calibration (aapki normal aankh ka EAR)
  2. Random order mein 3 challenges: BLINK x2, turn head LEFT, turn head RIGHT
     (random order isliye ki pehle se record kiya hua video replay na ho sake)
  3. Agar deepfake model file maujood hai to challenge ke dauran face frames par
     CNN fake-score bhi nikalta hai
  4. Result logs/liveness_log.csv mein, fail hone par snapshot logs/ mein save
"""
import csv
import os
import random
import time

import cv2
import numpy as np

import detector as D

CHALLENGES = ["BLINK", "LEFT", "RIGHT"]
TIMEOUT = 8.0          # har challenge ke liye seconds
CNN_FAKE_LIMIT = 0.60  # isse upar average fake score = fail


def head_ratio(pts):
    """~0.5 = seedha. Mirrored frame par: chhota = user ne LEFT ghumaya, bada = RIGHT."""
    xl, xr = pts[234][0], pts[454][0]
    lo, hi = min(xl, xr), max(xl, xr)
    return float((pts[1][0] - lo) / (hi - lo + 1e-6))


def new_session():
    order = CHALLENGES[:]
    random.shuffle(order)
    return {"phase": "calibrate", "calib": [], "thr": None, "queue": order, "step": 0,
            "t0": time.time(), "blinks": 0, "closed": 0, "hold": 0,
            "faces": [], "frame_i": 0, "result": None, "reason": ""}


def instruction(ch):
    return {"BLINK": "Do baar BLINK karo", "LEFT": "Sir apne LEFT ghumao",
            "RIGHT": "Sir apne RIGHT ghumao"}[ch]


def log_result(s, cnn_score, frame):
    os.makedirs(D.LOG_DIR, exist_ok=True)
    path = os.path.join(D.LOG_DIR, "liveness_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["time", "result", "reason", "challenges", "cnn_fake_score"])
        wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), s["result"], s["reason"],
                     "|".join(s["queue"]), "" if cnn_score is None else round(cnn_score, 3)])
    if s["result"] == "FAIL":
        cv2.imwrite(os.path.join(D.LOG_DIR, f"fail_{int(time.time())}.jpg"), frame)


def finish(s, ok, reason, model, frame):
    cnn_score = None
    if ok and model is not None and s["faces"]:
        cnn_score = float(np.mean(model.predict(s["faces"])))
        if cnn_score > CNN_FAKE_LIMIT:
            ok, reason = False, f"CNN ne fake pakda (score {cnn_score:.2f})"
    s["phase"] = "done"
    s["result"] = "PASS" if ok else "FAIL"
    s["reason"] = reason
    log_result(s, cnn_score, frame)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam nahi khuli.")
        return
    tracker = D.FaceTracker()
    model = D.DeepfakeModel() if os.path.exists(D.MODEL_PATH) else None
    if model is None:
        print("Note: deepfake model file nahi mili, sirf challenge-response chalega.")
    s = new_session()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        pts = tracker.process(frame)
        text, color = "", (255, 255, 255)

        if s["phase"] == "calibrate":
            text = "Camera ki taraf seedha dekho..."
            if pts is not None:
                s["calib"].append(D.mean_ear(pts))
            if time.time() - s["t0"] > 1.5 and len(s["calib"]) >= 15:
                s["thr"] = 0.75 * float(np.median(s["calib"]))
                s["phase"] = "challenge"
                s["t0"] = time.time()

        elif s["phase"] == "challenge":
            ch = s["queue"][s["step"]]
            text = f"[{s['step'] + 1}/{len(s['queue'])}] {instruction(ch)}"
            left = TIMEOUT - (time.time() - s["t0"])

            if pts is None:
                text += "  (face nahi dikh raha)"
            else:
                s["frame_i"] += 1
                if model is not None and s["frame_i"] % 10 == 0 and len(s["faces"]) < 12:
                    face, _ = D.crop_from_pts(frame, pts)
                    if face.size:
                        s["faces"].append(face)

                passed = False
                if ch == "BLINK":
                    if D.mean_ear(pts) < s["thr"]:
                        s["closed"] += 1
                    else:
                        if s["closed"] >= 2:
                            s["blinks"] += 1
                        s["closed"] = 0
                    text += f"  ({s['blinks']}/2)"
                    passed = s["blinks"] >= 2
                else:
                    r = head_ratio(pts)
                    hit = r < 0.36 if ch == "LEFT" else r > 0.64
                    s["hold"] = s["hold"] + 1 if hit else 0
                    passed = s["hold"] >= 3

                if passed:
                    s["step"] += 1
                    s["t0"], s["blinks"], s["closed"], s["hold"] = time.time(), 0, 0, 0
                    if s["step"] >= len(s["queue"]):
                        finish(s, True, "Saare challenges pass", model, frame)

            if s["phase"] == "challenge" and left <= 0:
                finish(s, False, f"Timeout: {ch} challenge pura nahi hua", model, frame)
            elif s["phase"] == "challenge":
                text += f"   {left:.0f}s"

        if s["phase"] == "done":
            good = s["result"] == "PASS"
            color = (0, 200, 0) if good else (0, 0, 255)
            text = f"{s['result']}: {s['reason']}  |  r=restart q=quit"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Liveness Check", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            s = new_session()

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
