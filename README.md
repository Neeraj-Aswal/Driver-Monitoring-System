# 🚗 Real-Time Driver Monitoring System

An AI-powered driver monitoring system designed to improve road safety by detecting mobile phone usage and driver drowsiness in real time using computer vision and deep learning.

The system combines **YOLOv8 object detection**, **MediaPipe facial landmark detection**, and **Eye Aspect Ratio (EAR)** to monitor driver behavior through a webcam and generate audio alerts when potentially unsafe activities are detected.

## ✨ Features

- 📱 **Mobile Phone Detection:** Detects mobile phones near the driver using a custom-trained YOLOv8 model.
- 😴 **Drowsiness Detection:** Monitors eye movements using facial landmarks and Eye Aspect Ratio (EAR).
- 🚨 **Real-Time Audio Alerts:** Generates alerts when phone usage or drowsiness is detected.
- 🎥 **Live Camera Monitoring:** Processes webcam input in real time.
- 📊 **Interactive Dashboard:** Streamlit-based interface for monitoring the system.
- ⚡ **ONNX Model Inference:** Uses ONNX Runtime for efficient model execution.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 Nano | Mobile phone detection |
| OpenCV | Image and video processing |
| MediaPipe | Facial landmark detection |
| ONNX Runtime | Model inference |
| Streamlit | Interactive dashboard |
| NumPy | Numerical operations |
| Roboflow | Dataset preparation |
| Google Colab | Model training |

## 🧠 System Architecture

```text
             Webcam Input
                  |
                  v
          Frame Acquisition
                  |
          +-------+-------+
          |               |
          v               v
     Phone Detection   Face Detection
       (YOLOv8)         (MediaPipe)
          |               |
          v               v
     Phone Status      Eye Landmarks
                          |
                          v
                    EAR Calculation
                          |
                          v
                  Drowsiness Detection
          |               |
          +-------+-------+
                  |
                  v
             Alert Engine
                  |
                  v
             Audio Warning
                  |
                  v
          Streamlit Dashboard
```

## 🤖 Machine Learning Model

### Mobile Phone Detection

The mobile phone detection model was trained using a custom Roboflow dataset and YOLOv8 Nano.

**Training Details:**

- Model: YOLOv8 Nano
- Dataset: Custom driver phone detection dataset
- Dataset size: 1,073 images
- Training epochs: 100
- Framework: Ultralytics
- Deployment format: ONNX

**Model Performance (Training Results):**

| Metric | Score |
|---|---:|
| Precision | 0.976 |
| Recall | 0.978 |
| mAP@50 | 0.992 |
| mAP@50-95 | 0.550 |

*Note: These are the reported evaluation results from the trained model, not a guarantee of real-world performance.*

### Drowsiness Detection

The system uses MediaPipe Face Landmarker to identify facial landmarks around the eyes.

The Eye Aspect Ratio (EAR) is calculated to estimate whether the driver's eyes are closed.

The EAR formula:

\[
EAR = \frac{||p_2-p_6||+||p_3-p_5||}{2||p_1-p_4||}
\]

When the EAR remains below a defined threshold for a specified duration, the system identifies possible drowsiness and triggers an alert.

## 📁 Project Structure

```text
driver-monitoring-system/
│
├── models/
│   ├── best.onnx
│   └── face_landmarker.task
│
├── alert/
│   └── alert_engine.py
│
├── main_monitor.py
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/driver-monitoring-system.git
```

### 2. Navigate to the Project Directory

```bash
cd driver-monitoring-system
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add Required Models

Place the following model files inside the `models/` directory:

- `best.onnx` — trained YOLOv8 phone detection model.
- `face_landmarker.task` — MediaPipe face landmark model.

### 6. Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Make sure your webcam is connected and accessible.

## 🎯 Configuration

The monitoring system uses configurable parameters for detection and alert behavior.

| Parameter | Default Value |
|---|---:|
| PHONE_CONFIDENCE | 0.20 |
| NMS_THRESHOLD | 0.45 |
| EAR_THRESHOLD | 0.20 |
| DROWSY_TIME | 1.5 seconds |
| ALERT_TIME | 2.5 seconds |

These values can be adjusted depending on camera quality, lighting conditions, and detection requirements.

## 🔮 Future Improvements

- Head pose estimation for detecting driver distraction.
- Yawning detection for improved fatigue monitoring.
- Improved phone detection across different lighting conditions.
- Driver identity recognition.
- Cloud-based monitoring and analytics.
- GPS integration for location-aware safety monitoring.
- Optimization for deployment on embedded devices.

## ⚠️ Limitations

- Detection accuracy may vary with lighting, camera positioning, and occlusion.
- Eye-based drowsiness estimation may be affected by glasses and facial visibility.
- Phone detection depends on the quality and variety of training data.
- This is an academic prototype and should not be considered a certified automotive safety system.

## 👨‍💻 Project Information

**Project Name:** Real-Time Driver Monitoring System

**Domain:** Artificial Intelligence | Machine Learning | Computer Vision

**Developed by:** Neeraj Aswal and Team

**Project Type:** Academic / PBL Project

---

⭐ If you find this project interesting, consider giving the repository a star!

**Disclaimer:** This project is developed for educational and research purposes to demonstrate real-time computer vision applications for driver safety.
