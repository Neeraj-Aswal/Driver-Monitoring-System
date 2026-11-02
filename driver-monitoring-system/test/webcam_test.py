import cv2
import numpy as np
import onnxruntime as ort

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/best.onnx"

CONFIDENCE_THRESHOLD = 0.10
NMS_THRESHOLD = 0.45

INPUT_SIZE = 640


# ==========================================
# LOAD ONNX MODEL
# ==========================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Model loaded successfully!")
print("Input:", input_name)


# ==========================================
# OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam frame.")
        break

    original = frame.copy()

    # --------------------------------------
    # Resize and normalize
    # --------------------------------------

    image = cv2.resize(
        frame,
        (INPUT_SIZE, INPUT_SIZE)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    # HWC → CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # --------------------------------------
    # Run model
    # --------------------------------------

    outputs = session.run(
        None,
        {input_name: image}
    )

    predictions = outputs[0][0]

    # YOLOv8 output:
    # [4 box coordinates + confidence]
    #
    # Shape:
    # (5, 8400)

    predictions = predictions.T

    # --------------------------------------
    # Extract detections
    # --------------------------------------

    boxes = []
    scores = []

    frame_h, frame_w = original.shape[:2]

    for detection in predictions:

        cx, cy, w, h, confidence = detection

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Convert from 640x640 coordinates
        x1 = int(
            (cx - w / 2)
            * frame_w / INPUT_SIZE
        )

        y1 = int(
            (cy - h / 2)
            * frame_h / INPUT_SIZE
        )

        box_w = int(
            w * frame_w / INPUT_SIZE
        )

        box_h = int(
            h * frame_h / INPUT_SIZE
        )

        boxes.append([
            x1,
            y1,
            box_w,
            box_h
        ])

        scores.append(
            float(confidence)
        )

    # --------------------------------------
    # Non-Maximum Suppression
    # --------------------------------------

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD
    )

    # --------------------------------------
    # Draw detections
    # --------------------------------------

    if len(indices) > 0:

        for i in indices:

            i = int(i)

            x, y, w, h = boxes[i]

            confidence = scores[i]

            cv2.rectangle(
                original,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            label = (
                f"Phone {confidence:.2f}"
            )

            cv2.putText(
                original,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # --------------------------------------
    # Display
    # --------------------------------------

    cv2.imshow(
        "Driver Monitoring - Phone Detection",
        original
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Webcam stopped.")