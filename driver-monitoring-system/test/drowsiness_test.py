import cv2
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/face_landmarker.task"

# Eye Aspect Ratio threshold
# Lower value = eye more closed
EAR_THRESHOLD = 0.20


# ==========================================
# MEDIAPIPE SETUP
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

detector = vision.FaceLandmarker.create_from_options(
    options
)

print("MediaPipe Face Landmarker loaded successfully!")


# ==========================================
# EYE LANDMARKS
# ==========================================
#
# MediaPipe landmark indices
#
# Left eye:
#   362, 385, 387, 263, 373, 380
#
# Right eye:
#   33, 160, 158, 133, 153, 144
#
# These six points allow us to calculate
# the Eye Aspect Ratio (EAR).
# ==========================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ==========================================
# EAR FUNCTION
# ==========================================

def calculate_ear(landmarks, eye_indices, frame_width, frame_height):

    points = []

    for index in eye_indices:

        landmark = landmarks[index]

        x = landmark.x * frame_width
        y = landmark.y * frame_height

        points.append((x, y))


    # Six points:
    #
    # p1 ----- p4
    #  |       |
    # p2       p6
    #  |       |
    # p3 ----- p5

    p1, p2, p3, p4, p5, p6 = points


    # Vertical distances
    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)

    # Horizontal distance
    horizontal = math.dist(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    detector.close()

    exit()


print("Webcam started.")
print("Press Q to quit.")


# ==========================================
# TIMESTAMP
# ==========================================

timestamp_ms = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Failed to read webcam frame.")

        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)


    frame_height, frame_width = frame.shape[:2]


    # ======================================
    # CONVERT TO MEDIAPIPE IMAGE
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Timestamp must continuously increase
    timestamp_ms += 33


    # ======================================
    # FACE LANDMARK DETECTION
    # ======================================

    results = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ======================================
    # FACE FOUND
    # ======================================

    if results.face_landmarks:

        landmarks = results.face_landmarks[0]


        # ==================================
        # CALCULATE EAR
        # ==================================

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE,
            frame_width,
            frame_height
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE,
            frame_width,
            frame_height
        )


        # Average both eyes
        ear = (left_ear + right_ear) / 2.0


        # ==================================
        # OPEN / CLOSED
        # ==================================

        if ear < EAR_THRESHOLD:

            eye_status = "CLOSED"

            status_color = (0, 0, 255)

        else:

            eye_status = "OPEN"

            status_color = (0, 255, 0)


        # ==================================
        # DRAW EYE LANDMARKS
        # ==================================

        for eye_indices in [LEFT_EYE, RIGHT_EYE]:

            for index in eye_indices:

                landmark = landmarks[index]

                x = int(
                    landmark.x * frame_width
                )

                y = int(
                    landmark.y * frame_height
                )

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (255, 255, 0),
                    -1
                )


        # ==================================
        # DISPLAY STATUS
        # ==================================

        cv2.putText(
            frame,
            f"EYES: {eye_status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2
        )


        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # ======================================
    # NO FACE
    # ======================================

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "Driver Monitoring - Eye Detection",
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

print("Drowsiness test stopped.")