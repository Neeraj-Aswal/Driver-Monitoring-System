import cv2
import math
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/face_landmarker.task"

# Eye Aspect Ratio threshold
EAR_THRESHOLD = 0.20

# How long eyes must remain closed
# before we consider it drowsiness.
DROWSY_TIME = 1.5

# Longer duration = stronger alert
ALERT_TIME = 2.5


# ==========================================
# EYE LANDMARKS
# ==========================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ==========================================
# EAR FUNCTION
# ==========================================

def calculate_ear(landmarks, eye_indices):

    points = []

    for index in eye_indices:

        landmark = landmarks[index]

        points.append(
            (landmark.x, landmark.y)
        )

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)

    horizontal = math.dist(p1, p4)

    if horizontal == 0:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


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

print("MediaPipe loaded successfully!")


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
# VARIABLES
# ==========================================

timestamp_ms = 0

eyes_closed_start = None

drowsiness_state = "NORMAL"


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
    # MEDIAPIPE IMAGE
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

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
            LEFT_EYE
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        ear = (
            left_ear + right_ear
        ) / 2.0


        # ==================================
        # CHECK EYES
        # ==================================

        if ear < EAR_THRESHOLD:

            eye_status = "CLOSED"

            # Start timer
            if eyes_closed_start is None:

                eyes_closed_start = time.monotonic()


            # Calculate duration
            closed_duration = (
                time.monotonic()
                - eyes_closed_start
            )


            # ------------------------------
            # DROWSINESS STATE
            # ------------------------------

            if closed_duration >= ALERT_TIME:

                drowsiness_state = "DROWSINESS ALERT"

            elif closed_duration >= DROWSY_TIME:

                drowsiness_state = "DROWSINESS WARNING"

            else:

                drowsiness_state = "NORMAL"


        else:

            eye_status = "OPEN"

            # Reset timer
            eyes_closed_start = None

            closed_duration = 0.0

            drowsiness_state = "NORMAL"


        # ==================================
        # DISPLAY EYE STATUS
        # ==================================

        if eye_status == "CLOSED":

            eye_color = (0, 0, 255)

        else:

            eye_color = (0, 255, 0)


        cv2.putText(
            frame,
            f"EYES: {eye_status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            eye_color,
            2
        )


        # ==================================
        # EAR
        # ==================================

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ==================================
        # CLOSED DURATION
        # ==================================

        cv2.putText(
            frame,
            f"CLOSED: {closed_duration:.1f}s",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ==================================
        # DROWSINESS STATUS
        # ==================================

        if drowsiness_state == "NORMAL":

            state_color = (0, 255, 0)

        elif drowsiness_state == "DROWSINESS WARNING":

            state_color = (0, 165, 255)

        else:

            state_color = (0, 0, 255)


        cv2.putText(
            frame,
            drowsiness_state,
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            state_color,
            2
        )


        # ==================================
        # DRAW EYE LANDMARKS
        # ==================================

        for eye_indices in [
            LEFT_EYE,
            RIGHT_EYE
        ]:

            for index in eye_indices:

                landmark = landmarks[index]

                x = int(
                    landmark.x * frame_width
                )

                y = int(
                    landmark.y * frame_height
                )

                if (
                    0 <= x < frame_width
                    and 0 <= y < frame_height
                ):

                    cv2.circle(
                        frame,
                        (x, y),
                        3,
                        (255, 255, 0),
                        -1
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
            0.8,
            (0, 0, 255),
            2
        )

        # Reset
        eyes_closed_start = None

        drowsiness_state = "NORMAL"


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "Driver Monitoring - Drowsiness Detection",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

detector.close()

cv2.destroyAllWindows()

print("Drowsiness detector stopped.")