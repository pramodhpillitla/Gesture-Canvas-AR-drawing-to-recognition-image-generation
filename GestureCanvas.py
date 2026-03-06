import requests
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai
import textwrap
from io import BytesIO
import os
import threading
import time

ai_busy = False
output_text = ''
generated_image = None

# AI Configuration
genai.configure(api_key="Your_genai_api")
model = genai.GenerativeModel('gemini-2.5-flash')

# Hugging Face API Configuration
HF_API_KEY = "Your_HF_api"
HF_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

def generate_image(prompt):
    """Generate an image from a text prompt using Hugging Face API."""
    print(f"Generating image with prompt: {prompt}")
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            try:
                image = Image.open(BytesIO(response.content)).convert("RGB")
                print("Image generated successfully")
                return image
            except Exception as e:
                print(f"Error processing image: {e}")
                return None
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def display_image(image):
    """Convert PIL image to OpenCV format for display."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Video capture setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height
cap.set(cv2.CAP_PROP_FPS, 30)

# Hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0,
                        detectionCon=0.8, minTrackCon=0.5)

def getHandInfo(img):
    """Detect hand, draw fingertip circles, and return finger states and landmarks."""
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        fingers = detector.fingersUp(hand1)

        # Draw small green circles at fingertip landmarks
        fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        for idx in fingertip_indices:
            x, y = lmList[idx][0:2]
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

        return fingers, lmList
    return None, None

def draw(info, prev_pos, canvas, move_threshold=5):
    """Draw on canvas based on finger gestures."""
    fingers, lmList = info
    current_pos = None

    # Index finger up → draw
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        prev_arr = np.array(prev_pos, dtype=np.int32)
        curr_arr = np.array(current_pos, dtype=np.int32)

        if np.linalg.norm(curr_arr - prev_arr) > move_threshold:
            cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
            prev_pos = current_pos

    # All fingers up → clear canvas
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(canvas)
        prev_pos = None

    # Not drawing (index not up) → reset prev_pos
    else:
        prev_pos = None

    return prev_pos, canvas


def sendToAI(model, canvas, fingers):
    """Send canvas to Gemini model for interpretation (threaded)."""
    global ai_busy, output_text, generated_image

    if ai_busy:
        return ''  # skip if already running

    if fingers == [1, 0, 0, 0, 0]:  # Thumb up triggers AI
        ai_busy = True
        print("AI thread started...")

        def ai_task():
            global ai_busy, output_text, generated_image
            try:
                pil_image = Image.fromarray(canvas)
                response = model.generate_content(["Recognise this drawing and give a prompt for generating realistic image?", pil_image])
                new_output = response.text

                if new_output and new_output != output_text:
                    output_text = new_output
                    generated_image = generate_image(output_text)
                    print("AI processing done.")
            except Exception as e:
                print(f"AI thread error: {e}")
            finally:
                ai_busy = False

        threading.Thread(target=ai_task, daemon=True).start()
    return ''

def display_text(output_text, width=400, height=300):
    """Create an image with wrapped text for display."""
    text_img = np.zeros((height, width, 3), dtype=np.uint8)
    wrapped_text = textwrap.wrap(output_text, width=30)
    y_offset = 40
    font_scale = 0.8 if len(wrapped_text) <= 3 else 0.6
    cv2.putText(text_img, "Drawing Identified:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    for line in wrapped_text:
        cv2.putText(text_img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1)
        y_offset += 30
    return text_img

def test_image_generation():
    """Test image generation with a user-provided prompt."""
    prompt = input("Enter a prompt for image generation (or press Enter to skip): ")
    if prompt:
        print("Generating test image...")
        image = generate_image(prompt)
        if image:
            print("Test image generated successfully!")
            img_cv = display_image(image)
            cv2.imshow("Test Generated Image", img_cv)
            cv2.waitKey(0)
            cv2.destroyWindow("Test Generated Image")
        else:
            print("Failed to generate test image.")

# Globals
ai_busy = False
last_trigger_time = 0
AI_COOLDOWN = 3.0
generated_image = None
output_text = ''
save_generated = True

# Threaded hand detection globals
latest_info = (None, None)
latest_frame = None
stop_thread = False

def hand_detection_thread():
    """Background thread for continuous hand detection."""
    global latest_info, latest_frame, stop_thread
    while not stop_thread:
        if latest_frame is not None:
            frame_copy = latest_frame.copy()
            info = getHandInfo(frame_copy)
            latest_info = info
        time.sleep(0.02)

def ai_worker(model, canvas_copy):
    """Background AI worker for image understanding."""
    global ai_busy, output_text, generated_image, last_trigger_time
    ai_busy = True
    try:
        new_output = sendToAI(model, canvas_copy, [1, 0, 0, 0, 0])
        print(f"[ai_worker] Gemini output: {new_output}")
        if new_output and new_output != output_text:
            output_text = new_output
            img = generate_image(output_text)
            if img:
                generated_image = img
                if save_generated:
                    filename = f"generated_{int(time.time())}.png"
                    img.save(filename)
                    print(f"Saved generated image: {filename}")
    except Exception as e:
        print(f"[ai_worker] Error: {e}")
    finally:
        last_trigger_time = time.time()
        ai_busy = False

def create_status_overlay(img, text):
    """Draw a semi-transparent overlay with status text."""
    overlay = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    alpha = 0.5
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img

def main():
    """Main function to run the gesture-based image generation project."""
    global output_text, generated_image
    prev_pos = None
    canvas = None

    # Optional: Test image generation before starting
    test_image_generation()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture video frame.")
            break

        img = cv2.flip(img, 1)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        if canvas is None:
            canvas = np.zeros_like(img)

        info = getHandInfo(img)
        if info[0] is not None:
            prev_pos, canvas = draw(info, prev_pos, canvas)
            sendToAI(model, canvas.copy(), info[0])  # thread-safe call

        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        output_display = display_text(output_text)

        cv2.imshow("Gesture Canvas", image_combined)
        cv2.imshow("AI Output", output_display)

        if generated_image is not None:
            gen_img_cv = display_image(generated_image)
            cv2.imshow("Generated Image", gen_img_cv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
