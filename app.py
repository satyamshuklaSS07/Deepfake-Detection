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
            st.error("Image read nahi ho payi. JPG ya PNG image try karo.")
        else:
            st.subheader("Uploaded image")
            st.image(bgr_to_pil(bgr), caption="Original image", use_container_width=True)

            try:
                r = D.analyze_image(model, bgr)

                if not r.get("face_found", False):
                    banner({"label": "NO FACE"})
                    st.info(
                        "Image mein face detect nahi hua. Clear, front-facing "
                        "image try karo."
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

                    # OpenCV face and Grad-CAM arrays are BGR.
                    if r.get("face") is not None:
                        c1.image(
                            bgr_to_pil(r["face"]),
                            caption="Detected face",
                            use_container_width=True,
                        )
                    else:
                        c1.warning("Detected face image unavailable")

                    if r.get("cam") is not None:
                        c2.image(
                            explanation_to_pil(r["cam"]),
                            caption="Grad-CAM",
                            use_container_width=True,
                        )
                    else:
                        c2.warning("Grad-CAM image unavailable")

                    # detector.ela_image() returns RGB according to this project.
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

