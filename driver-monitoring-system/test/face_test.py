import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/face_landmarker.task"


# ==========================================
# CHECK MODEL
# ==========================================

import os

if not os.path.exists(MODEL_PATH):
    print("ERROR: Face Landmarker model not found!")
    print(f"Expected location: {MODEL_PATH}")
    exit()

print("Face Landmarker model found.")


# ==========================================
# CREATE FACE LANDMARKER
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.FaceLandmarker.create_from_options(options)

print("MediaPipe Face Landmarker loaded successfully!")


# ==========================================
# OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    detector.close()
    exit()

print("Webcam started.")
print("Press Q to quit.")


# ==========================================
# FACE LANDMARK CONNECTIONS
# ==========================================

# MediaPipe Face Landmarker returns 478 landmarks
# when the model supports them.

# We'll draw the landmarks as small circles.
# This avoids depending on the old mp.solutions
# drawing API.


# ==========================================
# MAIN LOOP
# ==========================================

frame_timestamp_ms = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam frame.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # OpenCV BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp must increase
    frame_timestamp_ms += 33

    # Run Face Landmarker
    results = detector.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # ======================================
    # DRAW FACE LANDMARKS
    # ======================================

    if results.face_landmarks:

        face_landmarks = results.face_landmarks[0]

        # Draw landmarks
        for landmark in face_landmarks:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            # Only draw valid points
            if (
                0 <= x < frame.shape[1]
                and 0 <= y < frame.shape[0]
            ):

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )


        # ----------------------------------
        # FACE DETECTED
        # ----------------------------------

        cv2.putText(
            frame,
            "FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Landmarks: {len(face_landmarks)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    else:

        # ----------------------------------
        # NO FACE
        # ----------------------------------

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "Driver Monitoring - Face Landmarker",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
detector.close()
cv2.destroyAllWindows()

print("Face detection stopped.")