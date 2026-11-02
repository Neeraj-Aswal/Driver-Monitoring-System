import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL_PATH = "models/best.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

image = cv2.imread("test_images/image1.jpg")

image = cv2.resize(image, (640, 640))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, axis=0)

start = time.perf_counter()

outputs = session.run(
    None,
    {input_name: image}
)

end = time.perf_counter()

print(f"Inference time: {(end - start) * 1000:.2f} ms")
print("Inference completed successfully!")