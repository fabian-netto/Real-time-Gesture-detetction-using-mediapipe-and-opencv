✋ Real-Time Hand Gesture Detection using MediaPipe & OpenCV
📌 Project Overview

This project implements a real-time hand gesture (finger count) detection system using MediaPipe Hands and OpenCV.
The system detects a single hand from a webcam feed, counts the number of raised fingers, and displays the result using a visual HUD with stability control to avoid flickering and false detections.

The project focuses on accuracy, smoothness, and real-time performance rather than deep learning models.

🎯 Key Features

Real-time hand tracking using MediaPipe

Orientation-aware thumb detection (left vs right hand)

Finger count smoothing using a rolling buffer

Stability-based gesture confirmation

Visual HUD with progress bar

Works fully offline after setup

⚠️ Major Challenge (Important)

This project is highly version-dependent.

✅ Python Version (Mandatory)

Python 3.10.12


✅ MediaPipe Version (Mandatory)

mediapipe==0.10.9


❌ Newer Python versions (3.11, 3.12) and newer MediaPipe versions may cause:

AttributeError: module 'mediapipe' has no attribute 'solutions'

Installation failures

Runtime crashes

This constraint was the main challenge of the project.

🛠️ Technologies Used

Python

OpenCV

MediaPipe

Collections (deque)

Real-time computer vision

📦 Installation & Setup
1️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

2️⃣ Install Required Packages
pip install opencv-python
pip install mediapipe==0.10.9

3️⃣ Verify Python Version
python --version


Output should be:

Python 3.10.12

▶️ How It Works

Captures live video from webcam

Detects hand landmarks using MediaPipe

Counts raised fingers using landmark positions

Smooths detection using a rolling buffer

Confirms gesture only after it stays stable for 0.7 seconds

Displays output via a visual HUD

🧠 Core Logic – Finger Counting

Thumb detection is hand-orientation aware

Other fingers are detected by comparing fingertip and joint Y-coordinates

A buffer ensures stable results instead of instant noisy changes


🚀 Possible Enhancements

Gesture-based control (volume, slides, media)

Voice feedback integration

Multi-hand support

FPS optimization

Deployment on Raspberry Pi

📄 License

This project is open-source and available for educational and personal use.
