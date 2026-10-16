"""Streamlit front end: upload an MRI image, send it to the API, show the result."""
import os
import time

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_SECONDS = 60
UNCERTAIN_BELOW = 0.70  # confidence under 70% is flagged as uncertain

st.set_page_config(page_title="BrainScan AI", page_icon="🧠", layout="centered")

st.title("🧠 BrainScan AI")
st.caption("Educational prototype. Not a medical device. Not for real diagnosis.")


# ---------- sidebar: API status ----------
with st.sidebar:
    st.header("API status")
    st.code(API_URL)
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health.get("status") == "ok":
            st.success("Connected")
        else:
            st.warning("API reachable but not healthy")
    except requests.RequestException:
        st.error("API not reachable")


# ---------- main: upload + predict ----------
uploaded = st.file_uploader(
    "Upload a brain MRI image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded is not None:
    # small image on the left, text on the right
    col_img, col_txt = st.columns([1, 2])
    col_img.image(Image.open(uploaded), width=220)

    with col_txt:
        st.write(f"**{uploaded.name}**")
        clicked = st.button("Predict", type="primary")
        result_box = st.empty()  # the class name and confidence appear here

    if clicked:
        with st.spinner("Analyzing..."):
            start = time.perf_counter()
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                st.stop()
            elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"API error ({response.status_code}): {detail}")
            st.stop()

        result = response.json()

        with result_box.container():
            st.markdown(f"Class: **{result['prediction']}**")
            st.caption(f"Confidence: {result['confidence']:.1%} · {elapsed_ms:.0f} ms")

        if result["confidence"] < UNCERTAIN_BELOW:
            st.warning("Low confidence: the result is uncertain. "
                       "A specialist should review this image.")

        probs = pd.DataFrame(
            {"probability": result["probabilities"]}
        ).sort_values("probability", ascending=False)
        st.bar_chart(probs)

st.info("This tool is a student prototype for educational purposes only.")