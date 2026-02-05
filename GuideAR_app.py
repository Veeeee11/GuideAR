# =====================================================
# 🧠 GuideAR — Streamlit Smart Appliance Assistant
# =====================================================

import streamlit as st
import cv2
import numpy as np
import tempfile, os, time
import easyocr
import onnxruntime as ort
from transformers import OwlViTProcessor
from gtts import gTTS
from openai import OpenAI

# -----------------------------
# 🔑 API Setup
# -----------------------------
API_KEY = "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"
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# -----------------------------
# ⚙️ Model Initialization
# -----------------------------
reader = easyocr.Reader(['en'])
onnx_path = "/Users/gorugantusreevindhya/Desktop/guideAR_washingmachine/owlvit_base_patch32.onnx"

try:
    sess = ort.InferenceSession(onnx_path)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    ONNX_READY = True
except:
    sess, processor = None, None
    ONNX_READY = False

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------
st.set_page_config(page_title="GuideAR", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2f, #2a2a3c);
        color: white;
    }
    h1 {
        color: #00FFFF !important;
        text-align: center;
        font-size: 2.8em;
    }
    .stButton>button {
        background-color: #00FFFF;
        color: black;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 1.1em;
    }
    .stSuccess {
        background-color: #222232;
        border-left: 5px solid #00FFFF;
        padding: 1em;
        border-radius: 8px;
    }
    .tag {
        display: inline-block;
        background-color: #00FFFF;
        color: black;
        padding: 5px 10px;
        border-radius: 20px;
        margin: 3px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 GuideAR — Smart Appliance Assistant")
st.markdown(
    "#### Scan your appliance control panel to get an **AI-powered operating guide**. "
    "Detects buttons, reads labels, and explains how to operate step-by-step — with audio instructions!"
)

st.markdown("---")

# -----------------------------
# 📸 Image Upload / Capture
# -----------------------------
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📷 Upload or Capture Image")
    source = st.radio("Select Image Source:", ["Upload Image", "Use Webcam"])

    if source == "Upload Image":
        uploaded = st.file_uploader("Upload an appliance panel image:", type=["jpg", "png", "jpeg"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
    else:
        capture_btn = st.button("📸 Capture from Webcam")
        frame = None
        if capture_btn:
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            st.info("Initializing camera... press 'q' in the window to capture.")
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                cv2.imshow("Capture (press Q to snap)", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    frame = img
                    break
            cap.release()
            cv2.destroyAllWindows()

if "frame" not in locals():
    st.stop()

# -----------------------------
# 🔍 OCR + Detection
# -----------------------------
if frame is not None:
    st.markdown("### 🔍 Detecting Labels...")
    with st.spinner("Analyzing image... please wait"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        ocr_results = reader.readtext(thresh)
        detected_words = [t.lower() for (_, t, c) in ocr_results if c > 0.45]

        # Draw detection boxes
        for (bbox, text, conf) in ocr_results:
            if conf > 0.45:
                (tl, tr, br, bl) = bbox
                cv2.rectangle(frame, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(tl[0]), int(tl[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detected Labels", use_container_width=True)

    if not detected_words:
        st.warning("No readable text detected. Try better lighting or a closer shot.")
        st.stop()

    # Display words as colorful tags
    st.markdown("### 🏷️ Detected Labels:")
    tag_html = "".join([f"<span class='tag'>{word}</span>" for word in sorted(set(detected_words))])
    st.markdown(tag_html, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------
    # 🧠 Generate Operating Procedure
    # -----------------------------
    words = ", ".join(sorted(set(detected_words)))
    st.markdown("### 🧠 Generating Operating Procedure...")
    with st.spinner("AI is writing your step-by-step guide..."):
        prompt = f"""
        You are a helpful home appliance assistant. Based on these detected control panel labels: {words},
        create a clear, detailed step-by-step guide to operate the appliance.
        Include power-on, cycle selection, temperature, rinse/spin instructions, and finishing steps.
        Write like an expert manual, in under 8 steps.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
            )
            guide = response.choices[0].message.content.strip()

            st.success("### 🧾 Generated Operating Procedure:")
            st.markdown(f"**{guide}**")

            # Save text
            output_dir = "/Users/gorugantusreevindhya/Desktop/guideAR_washingmachine/generated_guides"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            text_path = os.path.join(output_dir, f"procedure_{timestamp}.txt")
            with open(text_path, "w") as f:
                f.write(guide)
            st.info(f"💾 Saved to: `{text_path}`")

            # -----------------------------
            # 🔊 Audio Playback
            # -----------------------------
            tts = gTTS(guide)
            audio_path = os.path.join(output_dir, f"procedure_{timestamp}.mp3")
            tts.save(audio_path)
            st.audio(audio_path, format="audio/mp3")
            st.success("🔊 Audio Guide Ready — Click Play Above!")

        except Exception as e:
            st.error(f"⚠️ LLM request failed: {e}")
