import requests
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai
import textwrap
from io import BytesIO
import os

# AI Configuration
genai.configure(api_key="your_genai_api_key")
model = genai.GenerativeModel('gemini-1.5-flash')

# Hugging Face API Configuration
HF_API_KEY = "your_hugging_faces_api_key"  # Replace with your valid API key
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
                image = Image.open(BytesIO(response.content))
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
cap.set(3, 1920)  # Width
cap.set(4, 1080)  # Height
cap.set(cv2.CAP_PROP_FPS, 30)  # Frame rate

# Hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0, detectionCon=0.8, minTrackCon=0.5)

def getHandInfo(img):
    """Detect hand and return finger states and landmarks."""
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        fingers = detector.fingersUp(hand1)
        for idx in [4, 8, 12, 16, 20]:  # Thumb, Index, Middle, Ring, Pinky tips
            cv2.circle(img, tuple(lmList[idx][0:2]), 10, (0, 255, 0), -1)
        return fingers, lmList
    return None, None

def draw(info, prev_pos, canvas):
    """Draw on canvas based on finger gestures."""
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
    elif fingers == [1, 1, 1, 1, 1]:  # All fingers up (reset canvas)
        canvas = np.zeros_like(canvas)
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    """Send canvas to Gemini model for interpretation."""
    if fingers == [1, 0, 0, 0, 0]:  # Thumb up triggers AI
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["What is this drawing?", pil_image])
        return response.text
    return ''

def display_text(output_text, width=800, height=300):
    """Create an image with wrapped text for display."""
    text_img = np.zeros((height, width, 3), dtype=np.uint8)
    wrapped_text = textwrap.wrap(output_text, width=40)
    y_offset = 50
    font_scale = 1 if len(wrapped_text) <= 3 else 0.7
    cv2.putText(text_img, "Drawing Identified:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    for line in wrapped_text:
        cv2.putText(text_img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        y_offset += 40
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

def main():
    """Main function to run the gesture-based image generation project."""
    prev_pos = None
    canvas = None
    output_text = ''
    generated_image = None

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
            print(f"Gesture detected: {info[0]}")  # Debug gesture
            prev_pos, canvas = draw(info, prev_pos, canvas)
            if info[0] == [1, 0, 0, 0, 0]:
                new_output_text = sendToAI(model, canvas, info[0])
                print(f"AI output: {new_output_text}")  # Debug AI output
                if new_output_text and new_output_text != output_text:
                    output_text = new_output_text
                    generated_image = generate_image(output_text)
        
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
