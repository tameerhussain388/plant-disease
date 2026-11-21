# Plant Disease Classification, Grad-CAM Explainability & Mobile Deployment

A computer vision project focused on training and evaluating convolutional neural networks on the PlantVillage dataset, using Grad-CAM to inspect what the model actually learns, and deploying a quantized model to an on-device React Native app.

I built this project to deepen my understanding of deep learning and computer vision fundamentals (convolutions, transfer learning, model explainability) before starting graduate studies in computer vision.

---

## What this project covers
- **Data & Limitations:** Analyzing class imbalance on 38 plant disease classes and investigating how laboratory backgrounds lead to shortcut learning.
- **Custom CNN vs Transfer Learning:** Building a 4-block CNN from scratch, then fine-tuning MobileNetV2 with frozen batch normalization and a low learning rate.
- **Grad-CAM from scratch:** Writing gradient-weighted class activation mapping directly using TensorFlow's `GradientTape` without high-level wrapper libraries.
- **TFLite Quantization:** Converting the fine-tuned model to float16 (reducing size from 8.9 MB to 4.39 MB with zero accuracy drop).
- **On-device Mobile App:** React Native app using `react-native-fast-tflite` for local inference.

---

## Dataset & Splitting
- **Dataset:** PlantVillage (54,303 leaf images across 38 classes).
- **Split:** Deterministic 70% train (38,012 images), 15% validation (8,146 images), and 15% test (8,145 images) with fixed seed `42`.
- **Class Imbalance:** Highly skewed (Citrus Greening has 5,507 images while Potato Healthy has only 152 images). Because of this, I used **Macro F1** rather than standard accuracy as the primary evaluation metric.

---

## Experiments & Results

| Model | Setup | Parameters | Size | Test Accuracy | Macro F1 | Epoch Time (Colab T4) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small CNN (Scratch)** | 4 Conv blocks + GlobalAvgPool | ~420 K | 1.7 MB | 88.4% | 86.1% | ~24s |
| **MobileNetV2** | Phase A (Frozen Backbone) | ~2.3 M | 9.2 MB | 94.8% | 93.2% | ~18s |
| **MobileNetV2** | Phase B (Fine-tuned > layer 100) | ~2.3 M | 9.2 MB | **98.6%** | **98.1%** | ~21s |

---

## Grad-CAM Findings: Shortcut Learning
The primary motivation for this project was to understand model interpretability using Grad-CAM.

### Observations:
1. **On lab test images:** The model correctly localizes spots, chlorosis, and leaf edges with high confidence.
2. **On natural background images:** When leaves were tested against natural backgrounds (hands holding leaves, outdoor soil, sun glare), prediction confidence dropped from ~98% to ~65-72%.
3. **Why?** PlantVillage leaves were photographed on plain gray sheets, tables, or black cardboard. The feature maps revealed that the network often relies on background textures and leaf contours rather than solely inspecting the lesion pathology. This is a clear demonstration of **shortcut learning** in vision models.

---

## Model Quantization & Parity

I exported the fine-tuned model to TensorFlow Lite and verified output parity against Keras across 50 test samples:

- **Float32:** 8.9 MB
- **Float16 Quantized:** 4.39 MB (50% reduction, average probability difference < 0.002, 100% class match)
- **INT8 Dynamic Range:** 2.6 MB (71% reduction)

The Float16 model (`plant_disease_model.tflite`) is stored in `models/`.

---

## Repository Structure
```
plant-disease-cv/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_gradcam.ipynb
│   └── 05_tflite_export.ipynb
├── mobile-app/              # React Native app (TypeScript)
├── models/
│   ├── plant_disease_model.tflite
│   └── labels.txt
├── results/                 # Confusion matrix, distribution plots, metrics
├── server.py                # Local test server using OpenCV DNN
├── index.html               # Quick web tester
├── test_model.py            # CLI verification script
└── requirements.txt
```

---

## Running Locally

### 1. Verification Script
```bash
python test_model.py
```

### 2. Local Web Tester
```bash
python server.py
# Open http://localhost:8000
```

### 3. Mobile App
```bash
cd mobile-app
npm install
npx react-native run-android
```
