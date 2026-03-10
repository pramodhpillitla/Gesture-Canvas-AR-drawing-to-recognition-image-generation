import requests
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai
import textwrap
from io import BytesIO
import threading
import time

ai_busy = False
output_text = ''
generated_image = None

# AI Configuration
genai.configure(api_key="")  # <-- add your Gemini key
model = genai.GenerativeModel('gemini-2.5-flash')

# Hugging Face API Configuration
HF_API_KEY = "" # <-- add your HF key          
HF_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"


def generate_image(prompt):
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
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# Webcam Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_FPS, 30)


# Hand Detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0,
                        detectionCon=0.8, minTrackCon=0.5)


def getHandInfo(img):
    """Detect hand, draw fingertip circles, return finger states + landmarks."""
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)

        # Draw fingertip points
        tip_ids = [4, 8, 12, 16, 20]
        for idx in tip_ids:
            x, y = lmList[idx][0:2]
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

        return fingers, lmList
    return None, None


def draw(info, prev_pos, canvas, move_threshold=5):
    fingers, lmList = info
    current_pos = None

    # Index finger alone = draw
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]

        if prev_pos is None:
            prev_pos = current_pos

        if np.linalg.norm(np.array(prev_pos) - np.array(current_pos)) > move_threshold:
            cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
            prev_pos = current_pos

    # All fingers up = clear
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(canvas)
        prev_pos = None

    else:
        prev_pos = None

    return prev_pos, canvas


def sendToAI(model, canvas, fingers):
    """Trigger Gemini + FLUX when thumb is up."""
    global ai_busy, output_text, generated_image

    if ai_busy:
        return

    # Thumb up gesture
    if fingers == [1, 0, 0, 0, 0]:
        ai_busy = True
        print("AI thread started...")

        def ai_task():
            global ai_busy, output_text, generated_image
            try:
                pil_image = Image.fromarray(canvas)
                response = model.generate_content(
                    ["Recognise this drawing and give a prompt for generating realistic image?", pil_image]
                )

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


def display_text(output_text, width=400, height=300):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    wrapped = textwrap.wrap(output_text, width=30)

    cv2.putText(img, "Drawing Identified:", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    y = 40
    for line in wrapped:
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 30
    return img


def main():
    global output_text, generated_image
    prev_pos = None
    canvas = None

    while True:
        success, img = cap.read()
        if not success:
            print("Camera read error.")
            break

        img = cv2.flip(img, 1)

        if canvas is None:
            canvas = np.zeros_like(img)

        info = getHandInfo(img)
        if info[0] is not None:
            prev_pos, canvas = draw(info, prev_pos, canvas)
            sendToAI(model, canvas.copy(), info[0])

        blended = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        cv2.imshow("Gesture Canvas", blended)

        text_window = display_text(output_text)
        cv2.imshow("AI Output", text_window)

        if generated_image is not None:
            cv_img = display_image(generated_image)
            cv2.imshow("Generated Image", cv_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
