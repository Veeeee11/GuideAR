import cv2
import onnxruntime as ort
import numpy as np
import easyocr
from transformers import OwlViTProcessor
from gtts import gTTS
import tempfile, os

# ================================
# 🧠 Washing Machine Smart Guide (Final Version)
# ================================

# --- Load Model ---
onnx_path = r"/Users/gorugantusreevindhya/Desktop/guideAR_washingmachine/owlvit_base_patch32.onnx"  # ✅ change this to your .onnx model file
sess = ort.InferenceSession(onnx_path)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

reader = easyocr.Reader(['en'])
print("✅ Model and OCR ready")

# --- Prompts for buttons (expanded & simplified) ---
texts = [[
    "start", "power", "pause", "temperature", "rinse", "wash",
    "dry", "door lock", "eco", "spin", "cycle"
]]

# --- Initialize webcam ---
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
  # ✅ use 0 unless your webcam is at another index
if not cap.isOpened():
    raise Exception("❌ Webcam not found. Try reconnecting or using a different index.")

print("🎥 Live AR Guide Started — Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for better OCR accuracy
    frame_resized = cv2.resize(frame, (1280, 720))
    h, w, _ = frame_resized.shape
    image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # --- Object Detection (OwlViT ONNX) ---
    try:
        inputs = processor(images=image_rgb, text=texts, return_tensors="np")
        outputs = sess.run(None, {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })
        logits, boxes = outputs
        logits = np.squeeze(logits)
        boxes = np.squeeze(boxes)
        scores = logits.max(axis=-1)
        labels = logits.argmax(axis=-1)

        for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
            if score > 0.15:
                x0, y0, x1, y1 = box
                x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
                cv2.rectangle(frame_resized, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"{texts[0][label]} ({score:.2f})",
                            (x0, max(25, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
    except Exception as e:
        print("⚠️ ONNX detection skipped (possibly lighting issue).")

    # --- OCR Text Detection ---
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ocr_results = reader.readtext(thresh)
    detected_texts = [text.lower() for (_, text, conf) in ocr_results if conf > 0.4]

    if detected_texts:
        print("🧾 OCR detected:", detected_texts)

    # --- Draw OCR boxes ---
    for (bbox, text, conf) in ocr_results:
        if conf > 0.4:
            (tl, tr, br, bl) = bbox
            color = (255, 255, 0)
            cv2.rectangle(frame_resized, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), color, 2)
            cv2.putText(frame_resized, text, (int(tl[0]), int(tl[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- Voice Instruction Trigger ---
    for t in detected_texts:
        if "start" in t:
            instruction = "Step 1: Press the start button to begin the washing cycle."
        elif "power" in t:
            instruction = "Step 2: Press the power button to turn on the washing machine."
        elif "rinse" in t:
            instruction = "Step 3: Select the rinse option for cleaner clothes."
        elif "temp" in t or "temperature" in t:
            instruction = "Step 4: Use the temperature button to set desired water temperature."
        else:
            continue

        print(f"🗣️ Instruction: {instruction}")
        tts = gTTS(instruction)
        temp_path = tempfile.mktemp(suffix=".mp3")
        tts.save(temp_path)

        # ✅ macOS-friendly audio play
        os.system(f"afplay '{temp_path}'")
        break  # only one instruction per frame

    cv2.imshow("🧠 Washing Machine Smart Guide", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Session ended")

