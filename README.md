# 👁️ OcuLaunch
### Contactless HCI Using Eye Blink Detection and Hand Mesh Navigation
# Overview

**OcuLaunch** is a contactless Human-Computer Interaction (HCI) system that allows users to control their computer using only their **eyes and hand gestures** — no mouse, no keyboard, no touch.

> 👁️👁️ **Blink twice** → Launch the app  
> 🖐️ **Hand gestures** → Navigate through apps  

Built for accessibility, OcuLaunch empowers people with motor disabilities, paralysis, or anyone who wants a futuristic hands-free computing experience — using just a standard webcam.

---

## 🎯 Problem Statement

Millions of people with motor disabilities, paralysis, or tremors cannot use traditional input devices like a mouse or keyboard. Existing solutions are expensive, bulky, and require specialized hardware.

OcuLaunch solves this using just a **standard webcam** and **open-source computer vision** — making accessible computing available to everyone, for free.

---

## ✨ Features

- 👁️ Real-time eye blink detection using EAR (Eye Aspect Ratio) algorithm
- 🖐️ Hand gesture navigation using MediaPipe Hand Mesh
- 🚀 Double blink to launch any application
- 📊 Live EAR value and confidence score display during testing
- ⚡ Works on standard 720p webcam — no special hardware needed
- ♿ Accessibility-first design for motor-impaired users

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Core language |
| OpenCV | Webcam feed and image processing |
| MediaPipe Face Mesh | Eye landmark detection |
| MediaPipe Hands | Hand gesture navigation |
| Scikit-learn | Model training and evaluation |
| XGBoost | Blink classification |
| Jupyter Notebook | Development environment |
| Joblib | Model saving and loading |
| NumPy / Pandas | Data processing |
| Matplotlib / Seaborn | Visualizations |

---

## 🧠 How It Works
