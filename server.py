import os
import io
import json
import time
import base64
import numpy as np
import cv2
from http.server import HTTPServer, SimpleHTTPRequestHandler

MODEL_PATH = os.path.join("models", "plant_disease_model.tflite")
LABELS_PATH = os.path.join("models", "labels.txt")

# Load Labels
with open(LABELS_PATH, "r") as f:
    LABELS = [line.strip() for line in f.readlines()]
print(f"Loaded {len(LABELS)} classes.")

# Load Model via OpenCV DNN (Runs TFLite natively on CPU with SIMD acceleration)
net = cv2.dnn.readNet(MODEL_PATH)
print("Loaded plant_disease_model.tflite via OpenCV DNN engine!")

class ModelServerHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                body = json.loads(post_data.decode('utf-8'))
                image_b64 = body.get('image', '')
                
                if ',' in image_b64:
                    image_b64 = image_b64.split(',')[1]
                
                image_bytes = base64.b64decode(image_b64)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # BGR
                
                if img is None:
                    raise ValueError("Failed to decode image")

                # Preprocessing:
                # 1. Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 2. Resize to 224x224
                img_resized = cv2.resize(img_rgb, (224, 224))
                # 3. Preprocess matching MobileNetV2: (x / 127.5) - 1.0
                # OpenCV blobFromImage handles (x - mean) * scalefactor
                # scale = 1/127.5 = 0.007843137, mean = 127.5
                blob = cv2.dnn.blobFromImage(
                    img_rgb,
                    scalefactor=1.0 / 127.5,
                    size=(224, 224),
                    mean=(127.5, 127.5, 127.5),
                    swapRB=False,
                    crop=False
                )

                # Inference & Benchmark
                t0 = time.perf_counter()
                net.setInput(blob)
                output = net.forward()
                latency_ms = (time.perf_counter() - t0) * 1000

                # Output probabilities (apply softmax if raw logits)
                probs = output[0]
                if np.sum(probs) < 0.99 or np.sum(probs) > 1.01:
                    exp_probs = np.exp(probs - np.max(probs))
                    probs = exp_probs / np.sum(exp_probs)

                top_3_indices = np.argsort(probs)[::-1][:3]
                predictions = [
                    {
                        "rank": int(rank + 1),
                        "label": LABELS[idx],
                        "confidence": float(probs[idx]),
                        "percentage": f"{probs[idx] * 100:.1f}%"
                    }
                    for rank, idx in enumerate(top_3_indices)
                ]

                response_data = {
                    "success": True,
                    "latency_ms": round(latency_ms, 1),
                    "predictions": predictions
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
        else:
            super().do_POST()

if __name__ == '__main__':
    port = 8000
    server_address = ('', port)
    httpd = HTTPServer(server_address, ModelServerHandler)
    print(f"Server running with REAL TFLite model at http://localhost:{port}")
    httpd.serve_forever()
