# GestureCanvas: AR Drawing to Recognition and Image Generation

GestureCanvas is an interactive AI-based AR tool that allows users to draw in the air using hand gestures. It interprets the drawing using Google's Gemini AI and generates a related image using a Hugging Face model.

## Features

- ✍️ Draw in the air using hand gestures via webcam
- 🧠 Recognize your drawing with Gemini AI (Google Generative AI)
- 🖼️ Generate matching image with Hugging Face FLUX.1-dev model
- 🎯 Simple UI with OpenCV + real-time feedback

## Getting Started

1. Clone the Repository
git clone https://github.com/pramodhpillitla/Gesture-Canvas-AR-drawing-to-recognition-image-generation.git
cd GestureCanvas

2. Install Requirements
pip install -r requirements.txt

3. Set API Keys
Replace the API keys in gesture_canvas.py or use .env and python-dotenv.

4. Run the App
python gesture_canvas.py

🧪 Controls
☝️ Index finger up: Draw

👍 Thumb up: Send drawing to AI

✋ All fingers up: Clear canvas

Press q: Quit the app

📦 Dependencies
opencv-python

numpy

cvzone

Pillow

requests

google-generativeai

📌 Notes
Requires a working webcam.

Avoid pushing your API keys to GitHub – use .env and .gitignore.

