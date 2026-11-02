import cv2
import math
import time
import numpy as np
import mediapipe as mp
import onnxruntime as ort

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from alert.alert_engine import AlertEngine


# =========================================================
# SETTINGS
# =========================================================

YOLO_MODEL_PATH = "models/best.onnx"
FACE_MODEL_PATH = "models/face_landmarker.task"

YOLO_INPUT_SIZE = 640

# Keep this low because our difficult test image
# was successfully detected at a lower confidence.
PHONE_CONFIDENCE = 0.38
PHONE_CONFIRM_TIME = 3.0

NMS_THRESHOLD = 0.45

EAR_THRESHOLD = 0.20

DROWSY_TIME = 1.5
ALERT_TIME = 2.5


# =========================================================
# EYE LANDMARKS
# =========================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# =========================================================
# EAR CALCULATION
# =========================================================

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


# =========================================================
# LOAD YOLO
# =========================================================

print("Loading YOLO model...")

yolo_session = ort.InferenceSession(
    YOLO_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

yolo_input_name = yolo_session.get_inputs()[0].name

print("YOLO model loaded successfully!")
print("YOLO input:", yolo_input_name)


# =========================================================
# LOAD MEDIAPIPE
# =========================================================

print("Loading MediaPipe...")

base_options = python.BaseOptions(
    model_asset_path=FACE_MODEL_PATH
)

face_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)

print("MediaPipe loaded successfully!")


# =========================================================
# LOAD ALERT ENGINE
# =========================================================

alert_engine = AlertEngine()

print("Alert engine loaded successfully!")


# =========================================================
# OPEN WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    face_detector.close()

    exit()


print()
print("======================================")
print(" DRIVER MONITORING SYSTEM STARTED")
print("======================================")
print("Press Q to quit.")
print()


# =========================================================
# VARIABLES
# =========================================================

timestamp_ms = 0

eyes_closed_start = None

drowsiness_state = "NORMAL"

phone_detected = False

phone_confidence = 0.0

closed_duration = 0.0

phone_detected_start = None
phone_duration = 0.0


# =========================================================
# MAIN LOOP
# =========================================================
try:
  while True:

    ret, frame = cap.read()

    if not ret:

        print("Failed to read webcam frame.")

        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)

    frame_height, frame_width = frame.shape[:2]


    # =====================================================
    # PHONE DETECTION — YOLO
    # =====================================================

    yolo_image = cv2.resize(
        frame,
        (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
    )

    yolo_image = cv2.cvtColor(
        yolo_image,
        cv2.COLOR_BGR2RGB
    )

    yolo_image = yolo_image.astype(
        np.float32
    ) / 255.0

    # HWC -> CHW
    yolo_image = np.transpose(
        yolo_image,
        (2, 0, 1)
    )

    # Add batch dimension
    yolo_image = np.expand_dims(
        yolo_image,
        axis=0
    )


    # Run YOLO
    outputs = yolo_session.run(
        None,
        {
            yolo_input_name: yolo_image
        }
    )

    predictions = outputs[0][0]

    # YOLOv8 output:
    # 5 x 8400
    predictions = predictions.T


    boxes = []
    scores = []


    # =====================================================
    # EXTRACT PHONE DETECTIONS
    # =====================================================

    for detection in predictions:

        cx, cy, w, h, confidence = detection

        if confidence < PHONE_CONFIDENCE:
            continue


        x1 = int(
            (cx - w / 2)
            * frame_width
            / YOLO_INPUT_SIZE
        )

        y1 = int(
            (cy - h / 2)
            * frame_height
            / YOLO_INPUT_SIZE
        )

        box_w = int(
            w
            * frame_width
            / YOLO_INPUT_SIZE
        )

        box_h = int(
            h
            * frame_height
            / YOLO_INPUT_SIZE
        )


        boxes.append(
            [x1, y1, box_w, box_h]
        )

        scores.append(
            float(confidence)
        )


    # =====================================================
    # NMS
    # =====================================================

    phone_detected = False
    phone_confidence = 0.0


    if boxes:

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            PHONE_CONFIDENCE,
            NMS_THRESHOLD
        )

        if len(indices) > 0:

            for i in indices:

                i = int(i)

                x, y, w, h = boxes[i]

                confidence = scores[i]

                phone_detected = True

                phone_confidence = max(
                    phone_confidence,
                    confidence
                )


                # Draw phone box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"PHONE {confidence:.2f}",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )


    # =====================================================
    # PHONE ALERT
    # =====================================================

    
    if phone_detected:
    
        if phone_detected_start is None:
            phone_detected_start = time.monotonic()
    
        phone_duration = (
            time.monotonic() - phone_detected_start
        )
    
        if phone_duration > PHONE_CONFIRM_TIME:
            alert_engine.trigger_phone_alert()
    
    else:
    
        phone_detected_start = None
        phone_duration = 0.0
    
        alert_engine.reset_phone_alert()



    # =====================================================
    # FACE / EYE DETECTION — MEDIAPIPE
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms += 33

    face_results = face_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # =====================================================
    # FACE FOUND
    # =====================================================

    if face_results.face_landmarks:

        landmarks = face_results.face_landmarks[0]


        # Calculate EAR
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


        # =================================================
        # EYE STATUS
        # =================================================

        if ear < EAR_THRESHOLD:

            eye_status = "CLOSED"


            if eyes_closed_start is None:

                eyes_closed_start = time.monotonic()


            closed_duration = (
                time.monotonic()
                - eyes_closed_start
            )


            if closed_duration >= ALERT_TIME:

                drowsiness_state = "DROWSINESS ALERT"

            elif closed_duration >= DROWSY_TIME:

                drowsiness_state = "DROWSINESS WARNING"

            else:

                drowsiness_state = "NORMAL"


        else:

            eye_status = "OPEN"

            eyes_closed_start = None

            closed_duration = 0.0

            drowsiness_state = "NORMAL"


        # =================================================
        # DROWSINESS ALERT
        # =================================================

        if drowsiness_state == "DROWSINESS WARNING":

            alert_engine.trigger_drowsiness_alert()

        elif drowsiness_state == "DROWSINESS ALERT":

            alert_engine.trigger_drowsiness_alert()

        else:

            alert_engine.reset_drowsiness_alert()


        # =================================================
        # DRAW EYE POINTS
        # =================================================

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
                        (0, 255, 255),
                        -1
                    )


        # =================================================
        # EYE TEXT
        # =================================================

        if eye_status == "OPEN":

            eye_color = (0, 255, 0)

        else:

            eye_color = (0, 0, 255)


        cv2.putText(
            frame,
            f"EYES: {eye_status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            eye_color,
            2
        )


        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"CLOSED: {closed_duration:.1f}s",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # =====================================================
    # NO FACE
    # =====================================================

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

        eyes_closed_start = None

        closed_duration = 0.0

        drowsiness_state = "NORMAL"

        alert_engine.reset_drowsiness_alert()


    # =====================================================
    # DROWSINESS STATUS
    # =====================================================

    if drowsiness_state == "NORMAL":

        drowsy_color = (0, 255, 0)

    elif drowsiness_state == "DROWSINESS WARNING":

        drowsy_color = (0, 165, 255)

    else:

        drowsy_color = (0, 0, 255)


    cv2.putText(
        frame,
        drowsiness_state,
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        drowsy_color,
        2
    )


    # =====================================================
    # PHONE STATUS
    # =====================================================

    if phone_detected:

        phone_text = (
            f"PHONE DETECTED "
            f"{phone_confidence:.2f}"
        )

        phone_color = (255, 0, 0)

    else:

        phone_text = "PHONE: NOT DETECTED"

        phone_color = (0, 255, 0)


    cv2.putText(
        frame,
        phone_text,
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        phone_color,
        2
    )


    # =====================================================
    # ALERT STATUS
    # =====================================================

    if (
        alert_engine.phone_alert
        or alert_engine.drowsiness_alert
    ):

        cv2.putText(
            frame,
            "!!! DRIVER ALERT !!!",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # =====================================================
    # DISPLAY
    # =====================================================
    
    print(
        f"[TELEMETRY] EAR: {ear:.3f} | State: {drowsiness_state} | Phone: {phone_detected} (Conf: {phone_confidence:.2f})", 
        flush=True
    )
    
    cv2.imshow(
        "Driver Monitoring System",
        frame
    )


    # =====================================================
    # QUIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

except KeyboardInterrupt:
    print("\nMonitoring stopped by user.")

finally:
    cap.release()
    face_detector.close()
    cv2.destroyAllWindows()
    print("Driver Monitoring System stopped.")
