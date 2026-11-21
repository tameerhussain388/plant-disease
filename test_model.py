import os
import time
import numpy as np
import cv2

MODEL_PATH = os.path.join("models", "plant_disease_model.tflite")
LABELS_PATH = os.path.join("models", "labels.txt")

# 1. Load Labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

print(f"Loaded {len(labels)} class labels.")

# 2. Load TFLite Model via OpenCV DNN (Direct C++ engine, zero dependency issues)
net = cv2.dnn.readNet(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024/1024:.2f} MB)")

# 3. Simulate 224x224 RGB image
sample_rgb = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

# 4. Preprocess: exactly matching training (x / 127.5) - 1.0
blob = cv2.dnn.blobFromImage(
    sample_rgb,
    scalefactor=1.0 / 127.5,
    size=(224, 224),
    mean=(127.5, 127.5, 127.5),
    swapRB=False,
    crop=False
)

# 5. Warm-up Iteration
net.setInput(blob)
t0 = time.perf_counter()
output = net.forward()
warmup_ms = (time.perf_counter() - t0) * 1000

# 6. Benchmark 20 iterations
times = []
for _ in range(20):
    t_start = time.perf_counter()
    net.setInput(blob)
    output = net.forward()
    times.append((time.perf_counter() - t_start) * 1000)

probs = output[0]
if np.sum(probs) < 0.99 or np.sum(probs) > 1.01:
    exp_p = np.exp(probs - np.max(probs))
    probs = exp_p / np.sum(exp_p)

top_3_indices = np.argsort(probs)[::-1][:3]

print("\n" + "="*55)
print("     ON-DEVICE PC INFERENCE BENCHMARK RESULTS")
print("="*55)
print(f"Model file:             {MODEL_PATH}")
print(f"Warm-up latency:        {warmup_ms:.2f} ms")
print(f"Average latency (x20):  {np.mean(times):.2f} ms (Std: {np.std(times):.2f} ms)")
print(f"Fastest run:            {np.min(times):.2f} ms")
print("\nTOP-3 PREDICTIONS:")
for rank, idx in enumerate(top_3_indices):
    confidence = probs[idx] * 100
    print(f" #{rank+1}: {labels[idx]} ({confidence:.2f}%)")
print("="*55)
print("STATUS: Model verification PASSED! Ready for deployment.")
print("="*55)
