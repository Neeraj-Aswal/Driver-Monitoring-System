from ultralytics import YOLO
import cv2

MODEL_PATH = "models/best.onnx"
IMAGE_PATH = "test_images/image1.jpg"

# Load ONNX model through Ultralytics
model = YOLO(MODEL_PATH)

print("ONNX model loaded successfully!")

# Run detection
results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.10,
    iou=0.45,
    verbose=True
)

# Get first result
result = results[0]

print("\nDetected objects:")

if result.boxes is None or len(result.boxes) == 0:
    print("No phone detected.")
else:
    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        print(
            f"Class: {class_id} | "
            f"Confidence: {confidence:.3f}"
        )

# Draw boxes
annotated = result.plot()

# Save result
output_path = "test_result.jpg"
cv2.imwrite(output_path, annotated)

print(f"\nResult saved to: {output_path}")