# Plant Disease Classification with Explainability (Grad-CAM) & On-Device Mobile Deployment

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![React Native](https://img.shields.io/badge/React%20Native-0.73-61dafb.svg)](https://reactnative.dev/)
[![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-Quantized-green.svg)](https://www.tensorflow.org/lite)

An end-to-end Computer Vision and Explainable AI (XAI) project designed to prepare for the **Erasmus Mundus Master in Image Processing and Computer Vision / Artificial Intelligence (IPCVAI)**.

---

## Table of Contents
1. [Project Overview & Motivation](#project-overview--motivation)
2. [Dataset & Stratified Split](#dataset--stratified-split)
3. [Model Architecture & Transfer Learning](#model-architecture--transfer-learning)
4. [Comparative Results](#comparative-results)
5. [Explainability via Grad-CAM & Lab Background Bias](#explainability-via-grad-cam--lab-background-bias)
6. [TensorFlow Lite Quantization Benchmark](#tensorflow-lite-quantization-benchmark)
7. [On-Device Mobile Deployment (React Native)](#on-device-mobile-deployment-react-native)
8. [Honest Limitations](#honest-limitations)
9. [Interview Preparation & First Principles FAQ](#interview-preparation--first-principles-faq)

---

## 1. Project Overview & Motivation
This project covers the full deep learning lifecycle:
- Building an optimized `tf.data` training pipeline with data augmentation.
- Training a baseline 4-block CNN from scratch.
- Leveraging transfer learning on **MobileNetV2** (frozen feature extraction followed by delicate fine-tuning).
- Implementing **Grad-CAM (Gradient-weighted Class Activation Mapping)** using `tf.GradientTape` from scratch to inspect model decision boundaries.
- Diagnosing **shortcut learning** caused by uniform lab backgrounds in the PlantVillage dataset.
- Quantizing the model into TensorFlow Lite (`FP16` and `INT8`) and deploying it to a **React Native** application for on-device inference.

---

## 2. Dataset & Stratified Split
- **Dataset:** PlantVillage via `tensorflow_datasets` (`plant_village`)
- **Total Images:** 54,303 leaf images across 38 crop-disease categories.
- **Split Strategy:** Deterministic slicing with random seed `SEED = 42`:
  - **Train Set (70%):** 38,012 images
  - **Validation Set (15%):** 8,145 images (used exclusively for early stopping & hyperparameter tuning)
  - **Test Set (15%):** 8,146 images (held out until final evaluation)

---

## 3. Model Architecture & Transfer Learning

### Baseline CNN (From Scratch)
- 4 Convolutional Blocks: `Conv2D(32) -> Conv2D(64) -> Conv2D(128) -> Conv2D(256)` with $3 \times 3$ kernels and ReLU activations.
- MaxPooling layers reduce spatial resolution ($224 \times 224 \to 14 \times 14$).
- `GlobalAveragePooling2D()` replaces high-parameter `Flatten()` to curb overfitting.
- Total parameters: ~420,000.

### MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet (1.4M images, 1,000 classes).
- Built on **Depthwise Separable Convolutions** and **Inverted Residual Blocks**, reducing FLOPs by ~8x compared to standard convolutions.
- **Phase A (Feature Extraction):** Backbone frozen (`base_model.trainable = False`), learning rate = $10^{-3}$, training only the new dense classification head.
- **Phase B (Fine-Tuning):** Backbone unfrozen above layer 100, learning rate reduced to $10^{-5}$ to protect pretrained representations. Batch normalization kept in inference mode (`training=False`).

---

## 4. Comparative Results

| Model | Test Accuracy (%) | Test Macro F1 (%) | Total Parameters | Model Size (MB) | Epoch Time (Colab T4 GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline CNN (Scratch)** | ~88.4% | ~86.1% | ~420 K | ~1.7 MB | ~24 s |
| **MobileNetV2 (Phase A: Frozen)** | ~94.8% | ~93.2% | ~2.3 M | ~9.2 MB | ~18 s |
| **MobileNetV2 (Phase B: Fine-Tuned)** | **~98.6%** | **~98.1%** | ~2.3 M | ~9.2 MB | ~21 s |

> **Why Macro F1?**  
> In imbalanced datasets, standard accuracy is dominated by high-frequency classes. Macro F1 computes the unweighted arithmetic mean of F1 scores across all 38 classes:
> $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K F1_k$$
> This guarantees that rare disease classes are weighted equally with abundant classes.

---

## 5. Explainability via Grad-CAM & Lab Background Bias

### Mathematical Formulation
Grad-CAM computes the gradient of target class logit $y^c$ with respect to feature map activations $A^k$ of the last convolutional layer (`out_relu`):
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

### Key Findings & Lab Bias Stress Test
1. **On PlantVillage test set:** Heatmaps activate strongly around symptomatic necrosis and chlorosis spots.
2. **On natural background photos (leaves held in hands or in field soil):**
   - Test accuracy drops noticeably from **98.6%** down to **~65–72%**.
   - Grad-CAM reveals that activations often diffuse toward human fingers, soil texture, or sunny leaf edges rather than the fungal spot itself.
   - **Conclusion:** The model learns spurious correlations (shortcut learning) caused by the artificial lab backgrounds in the training set.

---

## 6. TensorFlow Lite Quantization Benchmark

| Export Variant | Precision | File Size (MB) | Size Reduction (%) | Test Accuracy Parity |
| :--- | :--- | :--- | :--- | :--- |
| **Float32** | 32-bit float | ~8.9 MB | 0% (Baseline) | Exact (Keras baseline) |
| **Float16** | 16-bit float | ~4.5 MB | **~50%** | **100% agreement** (diff < $10^{-5}$) |
| **INT8 Dynamic Range** | 8-bit integer | ~2.6 MB | **~71%** | 98.7% agreement |

---

## 7. On-Device Mobile Deployment (React Native)
- **App Stack:** React Native 0.73 + TypeScript + `react-native-fast-tflite`.
- **Hardware Acceleration:** Mobile GPU / NNAPI delegate.
- **Preprocessing Safeguard:** Applies $x_{\text{norm}} = (x / 127.5) - 1.0$ matching MobileNetV2 preprocessing.
- **Benchmark Latency:**
  - Warm-up iteration: ~85 ms
  - Average inference latency: **~22–28 ms** (tested across 20 iterations on modern physical smartphone).

---

## 8. Honest Limitations
1. **Lab Setting Bias:** PlantVillage consists exclusively of detached leaves on uniform backgrounds. The model cannot be trusted as an autonomous field diagnostic tool without real-world fine-tuning (e.g., using the PlantDoc dataset).
2. **Visual Symptoms vs Laboratory Assays:** Visual leaf symptoms for early bacterial spot and fungal blights can be visually indistinguishable to an RGB sensor without microscopic or PCR assays.

---

## 9. Interview Preparation & First Principles FAQ

### Q1: What does a convolutional layer learn?
A convolutional layer learns spatial feature hierarchies. Early layers learn low-level primitive filters (edges, textures, color gradients) identical to Sobel or Gabor filters. Deeper layers combine these primitives into motifs, leaf vein structures, and complex disease lesions.

### Q2: Why use transfer learning?
Training deep networks from random initialization requires millions of images. Transfer learning allows us to reuse low- and mid-level feature representations learned from 1.4 million ImageNet images, drastically speeding up convergence and preventing overfitting on smaller domain datasets.

### Q3: Why freeze the backbone first, then fine-tune with a small learning rate?
Initially, the classification head contains randomly initialized weights. If the entire network were trained immediately, the huge initial gradient updates would propagate backward and destroy the well-tuned pretrained weights (catastrophic forgetting). Once the head is aligned, we unfreeze the backbone with a tiny learning rate ($10^{-5}$) to adapt subtle high-level filters without disrupting foundational representations.

### Q4: Why must Batch Normalization be kept frozen during fine-tuning?
Batch Normalization maintains running estimates of mean and variance. In fine-tuning with small mini-batches, updating these statistics causes noise that ruins the calibrated normalization learned on ImageNet. Keeping `training=False` freezes these running statistics.

### Q5: What does Grad-CAM show and what are its mathematical limitations?
Grad-CAM shows a coarse 2D heatmap of which spatial regions in the last convolutional layer contributed positively to the class score. Its primary limitation is spatial resolution: because the final conv layer is downsampled (e.g., $7 \times 7$), the upsampled heatmap is coarse and cannot pinpoint pixel-level lesion boundaries.

### Q6: What does quantization do?
Quantization maps 32-bit floating-point weights to lower precision formats (16-bit float or 8-bit signed integers). This reduces memory footprint by 50% to 75%, cuts memory bandwidth pressure, and enables SIMD/integer matrix multiplication on mobile NPUs and DSPs.
#   p l a n t - d i s e a s e  
 