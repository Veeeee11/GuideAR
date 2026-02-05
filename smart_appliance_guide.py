import cv2
import easyocr
import time
from gtts import gTTS
import pygame
import os
from openai import OpenAI
import threading
import queue
import json

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize OpenAI client with OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Set extra headers for OpenRouter
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com",  # Site URL for rankings
    "X-Title": "Smart Appliance Guide",    # Site title for rankings
}

# Initialize pygame for audio
pygame.mixer.init()

class SmartApplianceGuide:
    def __init__(self):
        self.unique_words = set()
        self.instruction_queue = queue.Queue()
        self.is_running = True
        self.last_instruction_time = 0
        self.instruction_cooldown = 10  # seconds between instructions
        
    def generate_instructions(self, detected_text):
        try:
            # Combine detected text into a meaningful prompt
            prompt = f"""Based on these detected elements from an appliance interface: {', '.join(detected_text)},
            provide a clear, short instruction on how to operate this feature. Keep it concise and practical."""
            
            response = client.chat.completions.create(
                model="minimax/minimax-m2:free",
                messages=[
                    {"role": "system", "content": "You are a helpful appliance guide assistant. Provide clear, concise instructions."},
                    {"role": "user", "content": prompt}
                ],
                extra_headers=OPENROUTER_HEADERS,
                extra_body={}
            )
            
            instruction = response.choices[0].message.content
            return instruction
        except Exception as e:
            print(f"Error generating instructions: {e}")
            return None

    def text_to_speech(self, text, filename="instruction.mp3"):
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            
            # Play the audio
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            pygame.mixer.music.unload()
            os.remove(filename)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def process_instructions(self):
        while self.is_running:
            try:
                instruction = self.instruction_queue.get(timeout=1)
                current_time = time.time()
                
                if current_time - self.last_instruction_time >= self.instruction_cooldown:
                    print("\n🔊 New Instruction:")
                    print(instruction)
                    self.text_to_speech(instruction)
                    self.last_instruction_time = current_time
                
                self.instruction_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing instructions: {e}")

    def run(self):
        # Initialize webcam
        cap = cv2.VideoCapture(0)  # Try 0 or 1 depending on your setup
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return

        # Start instruction processing thread
        instruction_thread = threading.Thread(target=self.process_instructions)
        instruction_thread.start()

        print("🎥 Starting Smart Appliance Guide... Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to grab frame.")
                    break

                # Resize for faster processing
                small_frame = cv2.resize(frame, (640, 480))
                
                # OCR processing
                results = reader.readtext(small_frame)
                
                # Process detected text
                current_words = set()
                for (bbox, text, conf) in results:
                    if float(conf) > 0.5:  # Confidence threshold
                        text_clean = text.strip().lower()
                        if text_clean:
                            current_words.add(text_clean)
                            
                            # Draw bounding box and text
                            (tl, tr, br, bl) = bbox
                            cv2.rectangle(small_frame, 
                                        (int(tl[0]), int(tl[1])),
                                        (int(br[0]), int(br[1])), 
                                        (0, 255, 0), 2)
                            cv2.putText(small_frame, 
                                      text, 
                                      (int(tl[0]), int(tl[1]) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, 
                                      (0, 255, 0), 
                                      2)

                # Check for new words and generate instructions
                new_words = current_words - self.unique_words
                if new_words:
                    self.unique_words.update(new_words)
                    instruction = self.generate_instructions(current_words)
                    if instruction:
                        self.instruction_queue.put(instruction)

                # Display the frame
                cv2.imshow("Smart Appliance Guide", small_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n🛑 Stopping the guide...")
        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            instruction_thread.join()

            # Save recognized words
            with open("recognized_features.json", "w", encoding="utf-8") as f:
                json.dump({
                    "detected_features": list(self.unique_words)
                }, f, indent=4)
            
            print("✅ Guide stopped. Detected features saved to recognized_features.json")

if __name__ == "__main__":
    guide = SmartApplianceGuide()
    guide.run()