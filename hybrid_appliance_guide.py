# ==============================================
# 🧠 Hybrid Smart Appliance AR Guide
# (Local OCR + ONNX + Cloud LLM + Speech)
# ==============================================

import cv2
import numpy as np
import onnxruntime as ort
import easyocr
from transformers import OwlViTProcessor
from gtts import gTTS
from openai import OpenAI
import tempfile, os, time

# ==============================================
# 1️⃣  API Setup
# ==============================================
API_KEY = "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"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ==============================================
# 2️⃣  Load Vision Models
# ==============================================
onnx_path = "/Users/gorugantusreevindhya/Desktop/guideAR_washingmachine/owlvit_base_patch32.onnx"

try:
    sess = ort.InferenceSession(onnx_path)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    print("✅ ONNX model loaded successfully.")
except Exception as e:
    sess, processor = None, None
    print("⚠️ ONNX model not found or failed to load. Running OCR-only mode.")

reader = easyocr.Reader(['en'])
print("✅ EasyOCR initialized.\n")

texts = [[
    "power", "start", "pause", "temperature", "rinse",
    "wash", "spin", "eco", "timer", "heat", "mode", "cool",
    "lock", "dry", "soak", "door", "delay", "speed"
]]

# 3️⃣  Initialize iPhone Camera (force iPhone only)
# ==============================================
import time

print("📱 Searching for iPhone camera (Iriun / EpocCam)...")

iphone_camera_found = False
iphone_index = None

# Try first few indices to find the iPhone camera
for i in range(5):
    cap_test = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap_test.isOpened():
        # Test one frame
        ret, _ = cap_test.read()
        if ret:
            print(f"✅ Camera detected at index {i}")
            # Try to identify iPhone camera based on frame size (usually 1920x1080)
            width = cap_test.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if width >= 1280 or height >= 720:
                iphone_camera_found = True
                iphone_index = i
                cap_test.release()
                break
        cap_test.release()

if not iphone_camera_found:
    raise Exception("❌ iPhone camera not detected! Please open Iriun or EpocCam on your phone and connect via Wi-Fi or USB.")

# Open iPhone camera permanently
cap = cv2.VideoCapture(iphone_index, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise Exception("❌ Failed to open iPhone camera. Check that the Iriun app is running and the phone is connected.")

print(f"🎥 Using iPhone camera at index {iphone_index}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("🎥 Live Appliance AR Guide Started — Press 'Q' to quit\n")


# ==============================================
# 4️⃣  Main Loop
# ==============================================
last_query = ""
last_time = 0
cooldown = 12  # seconds between LLM calls

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        break

    frame = cv2.resize(frame, (1280, 720))
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🧩 ONNX Detection
    if sess is not None:
        try:
            inputs = processor(images=rgb, text=texts, return_tensors="np")
            outputs = sess.run(None, {
                "pixel_values": inputs["pixel_values"],
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            })
            logits, boxes = outputs
            logits, boxes = np.squeeze(logits), np.squeeze(boxes)
            scores, labels = logits.max(-1), logits.argmax(-1)
            for score, label, box in zip(scores, labels, boxes):
                if score > 0.25:
                    x0, y0, x1, y1 = (box * [w, h, w, h]).astype(int)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    label_text = texts[0][label]
                    cv2.putText(frame, f"{label_text} ({score:.2f})",
                                (x0, max(25, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
        except Exception as e:
            print("⚠️ ONNX inference skipped:", str(e))

    # 🔍 OCR Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    ocr_results = reader.readtext(thresh)

    detected_words = [t.lower() for (_, t, c) in ocr_results if c > 0.45]
    for (bbox, text, conf) in ocr_results:
        if conf > 0.45:
            (tl, tr, br, bl) = bbox
            cv2.rectangle(frame, (int(tl[0]), int(tl[1])),
                          (int(br[0]), int(br[1])), (255, 255, 0), 2)
            cv2.putText(frame, text, (int(tl[0]), int(tl[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if detected_words:
        print("🧾 OCR detected:", detected_words)

    # 💬 LLM Instruction Generator
    # 💬 LLM Enhanced Instruction Generator
current_time = time.time()
if detected_words and current_time - last_time > cooldown:
    words = ", ".join(sorted(set(detected_words)))
    if words != last_query:
        print(f"\n🧠 Sending detected words to LLM: {words}\n")

        prompt = f"""
You are an intelligent appliance assistant. You detect these control panel labels: {words}.
Write a clear, multi-step operating procedure in a helpful tone (like a user manual).
Include all relevant button functions (e.g., power, start, rinse, spin, etc.)
and describe how to properly operate the appliance step by step.

Output format example:
1. Step 1 — Turn on the power.
2. Step 2 — Select your desired wash cycle.
3. Step 3 — Adjust temperature, rinse, and spin settings.
4. Step 4 — Press start to begin washing.

Keep it under 8 steps, and make it easy to follow.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            full_procedure = response.choices[0].message.content.strip()

            # --- Print in console ---
            print("🧾 Generated Operating Procedure:\n")
            print(full_procedure)
            print("\n----------------------------------------\n")

            # --- Save to text file ---
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_dir = "/Users/gorugantusreevindhya/Desktop/guideAR_washingmachine/generated_guides"
            os.makedirs(output_dir, exist_ok=True)
            text_path = os.path.join(output_dir, f"procedure_{timestamp}.txt")

            with open(text_path, "w") as f:
                f.write(full_procedure)
            print(f"💾 Saved detailed procedure to: {text_path}")

            # --- Convert to speech ---
            tts = gTTS(full_procedure)
            audio_path = os.path.join(output_dir, f"procedure_{timestamp}.mp3")
            tts.save(audio_path)
            os.system(f"open '{audio_path}'")  # macOS playback
            print(f"🔊 Audio guide generated and playing: {audio_path}")

            # --- Update last query ---
            last_query = words
            last_time = current_time

        except Exception as e:
            print("⚠️ LLM request failed:", e)


# ==============================================
# 5️⃣  Cleanup
# ==============================================
cap.release()
cv2.destroyAllWindows()
print("🛑 Session ended successfully.")
