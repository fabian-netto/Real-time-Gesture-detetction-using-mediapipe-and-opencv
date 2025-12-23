import sys
import mediapipe as mp

print("Python EXE:", sys.executable)
print("Python VER:", sys.version)
print("MediaPipe file:", mp.__file__)
print("Has solutions:", hasattr(mp, "solutions"))
