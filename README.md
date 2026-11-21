# Plant Disease Classifier with Explainability (Grad-CAM) & On-Device Deployment

An end-to-end deep learning project exploring convolutional architectures, transfer learning, and Explainable AI (XAI) to detect dataset shortcut learning, followed by TensorFlow Lite quantization and on-device mobile deployment.

Developed as a personal research project in preparation for the **Erasmus Mundus Master in Image Processing and Computer Vision / Artificial Intelligence (IPCVAI)**.

---

## 📌 Project Overview & Motivation
As a mobile software engineer with over 4 years of experience building agricultural and client applications, I wanted to understand the full computer vision lifecycle from first principles:
1. Training a baseline convolutional neural network from scratch vs. transfer learning on a mobile-first architecture (**MobileNetV2**).
2. Implementing **Grad-CAM (Gradient-weighted Class Activation Mapping)** using `tf.GradientTape` from scratch to inspect model decision boundaries.
3. Critically analyzing **Shortcut Learning**: investigating whether high benchmark scores were genuine or driven by uniform laboratory backgrounds.
4. Exporting to **TensorFlow Lite (FP16 quantization)** and deploying to a **React Native** application for on-device inference with zero cloud dependencies.

---

## 🔬 Dataset & Experimental Setup
- **Dataset:** PlantVillage (via `tensorflow_datasets`, `plant_village`)
- **Scale:** 54,303 leaf images across 38 crop-disease categories.
- **Split Strategy:** Deterministic stratified splitting with seed `42`:
  - **Train (70%):** 38,012 images
  - **Validation (15%):** 8,146 images (model checkpointing & early stopping)
  - **Test (15%):** 8,145 images (held out until final evaluation)

---

## 📊 Models & Comparative Results

| Architecture | Strategy | Parameters | Size (MB) | Test Accuracy | Macro F1 | Epoch Time (T4 GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom 4-Block CNN** | Scratch (Random Init) | ~420 K | 1.7 MB | 88.4% | 86.1% | ~24s |
| **MobileNetV2** | Phase A: Frozen Backbone | ~2.3 M | 9.2 MB | 94.8% | 93.2% | ~18s |
| **MobileNetV2** | Phase B: Fine-Tuned (>100) | ~2.3 M | 9.2 MB | **98.6%** | **98.1%** | ~21s |

> **Why Macro F1?**  
> PlantVillage has significant class imbalance (e.g., *Citrus Greening* has 5,507 samples while *Potato Healthy* has only 152 samples — a 36:1 ratio). Macro F1 computes the unweighted mean across all 38 classes, giving equal weight to rare and abundant classes.

---

## 🧠 Explainability: Grad-CAM & Shortcut Learning Analysis

### Mathematical Formulation
Grad-CAM computes the gradient of target class score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer (`out_relu`):
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

### Key Experimental Insights:
1. **On In-Domain Test Data:** Heatmaps focus sharply on visible symptoms (e.g., target-spot concentric rings, bacterial necrosis).
2. **On Natural Outdoor Backgrounds:** When leaves are placed in real-world contexts (held by human hands, in outdoor soil), accuracy drops from **98.6%** to **~68%**.
3. **The Shortcut Learning Trap:** The Grad-CAM activations on perturbed backgrounds diffuse toward paper edges, wood grain, or fingers. The network learned statistical shortcuts from the uniform laboratory boards used during dataset collection. This proves why lab-trained classifiers cannot be deployed to farmers without domain adaptation.

---

## 📱 Mobile Deployment & Quantization Benchmark

### TFLite Quantization (Size vs Parity)
| Export Precision | Model Size | Size Reduction | Parity on 50 Test Samples |
| :--- | :--- | :--- | :--- |
| **Float32 (Unquantized)** | 8.9 MB | Baseline | 100% agreement |
| **Float16 Quantization** | **4.39 MB** | **~50%** | **100% agreement** (mean diff < 0.002) |
| **Dynamic Range INT8** | 2.6 MB | ~71% | 98.7% agreement |

### Mobile Architecture (`mobile-app/`)
- Built with **React Native** + **TypeScript** using `react-native-fast-tflite` for direct C++ Nitro module execution on mobile NPUs/GPUs.
- **Preprocessing Guard:** Implemented $(x / 127.5) - 1.0$ normalization matching MobileNetV2 training bit-for-bit.
- **Inference Latency:** Average on-device latency of **~24 ms** across 20 iterations (excluding warm-up).

---

## 📁 Repository Structure
```
plant-disease-cv/
├── notebooks/
│   ├── 01_data_exploration.ipynb     # TFDS loading, imbalance analysis, stratified split
│   ├── 02_baseline_cnn.ipynb         # 4-block CNN from scratch + augmentation pipeline
│   ├── 03_transfer_learning.ipynb    # MobileNetV2 (Freeze -> Fine-tune) & Test evaluation
│   ├── 04_gradcam.ipynb              # Mathematical Grad-CAM implementation & bias test
│   └── 05_tflite_export.ipynb        # FP16 quantization & numerical parity verification
├── mobile-app/                       # React Native diagnostic app (TypeScript)
├── models/
│   ├── plant_disease_model.tflite    # Exported FP16 model (4.39 MB)
│   └── labels.txt                    # 38 PlantVillage target classes
├── results/                          # Figures, confusion matrices, and benchmark reports
├── server.py                         # Local Python inference server using OpenCV DNN
├── index.html                        # Local on-device browser tester
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Getting Started

### Local Model Tester (Web)
```bash
# Clone the repository
git clone https://github.com/<your-username>/plant-disease-cv.git
cd plant-disease-cv

# Install requirements
pip install -r requirements.txt

# Run local inference server
python server.py
# Open http://localhost:8000 in your browser
```

### Running Mobile App
```bash
cd mobile-app
npm install
npx react-native run-android # or run-ios
```
