
# ------------------------------------------------------------------ video tab
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
                    suffix=suffix, delete=False
                ) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name

                bar = st.progress(0.0, text="Analyzing...")

                r = D.analyze_video(
                    model,
                    tmp_path,
                    max_seconds,
                    cnn_per_sec,
                    progress=lambda v: bar.progress(
                        min(1.0, max(0.0, float(v))),
                        text="Analyzing...",
                    ),
                )

                res = D.decide(
                    r["probs"], r["blink"], fake_thr, real_thr
                )
                sha = D.sha256_bytes(data)
                D.log_analysis(up.name, sha, res["label"], res["score"])
                banner(res)

                if not r["probs"]:
                    st.warning(
                        "Video se koi face/frame analyze nahi ho paya. "
                        "Doosra video try karo."
                    )
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Frames analyzed (CNN)", len(r["probs"]))
                    c2.metric("Max fake score", f"{max(r['probs']):.2f}")
                    c3.metric("Blinks", r["blink"]["blinks"])
                    c4.metric("Blink rate / min", r["blink"]["rate_per_min"])

                    df = pd.DataFrame({
                        "time_s": np.round(r["times"], 2),
                        "fake_prob": np.round(r["probs"], 4),
                    })

                    st.subheader("Fake score timeline")
                    st.line_chart(df.set_index("time_s")["fake_prob"])

                    if r["fps"] > 0 and r["ear"]:
                        ear = pd.DataFrame({
                            "time_s": np.arange(len(r["ear"])) / r["fps"],
                            "EAR": [
                                np.nan if v is None else v
                                for v in r["ear"]
                            ],
                        })
                        st.subheader("Eye Aspect Ratio (blink graph)")
                        st.line_chart(ear.set_index("time_s")["EAR"])

                    st.subheader("Suspicious frames + Grad-CAM")
                    top = np.argsort(r["probs"])[::-1][:4]
                    cols = st.columns(len(top))

                    for col, idx in zip(cols, top):
                        col.image(
                            model.explain(r["faces"][idx]),
                            channels="BGR",
                            caption=(
                                f"t={r['times'][idx]:.1f}s | "
                                f"fake {r['probs'][idx]:.2f}"
                            ),
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
