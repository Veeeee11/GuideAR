import cv2
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load your washing machine image
image_path = r"D:\WashingMachine_ARGuide\washing_machine.jpg"  # change to your image name
img = cv2.imread(image_path)

# Perform OCR
results = reader.readtext(img)

# Print text + confidence
for (bbox, text, conf) in results:
    print(f"{text} ({conf:.2f})")

# Visualize results
for (bbox, text, conf) in results:
    (tl, tr, br, bl) = bbox
    cv2.rectangle(img, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0, 255, 0), 2)
    cv2.putText(img, text, (int(tl[0]), int(tl[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("🧠 OCR Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
