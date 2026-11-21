# Plant Disease Detector (Mobile App)

On-device plant leaf disease diagnostic app built with **React Native** and **TensorFlow Lite (`react-native-fast-tflite`)**.

---
## Key Features
- **Zero Cloud Latency:** 100% on-device inference using quantized MobileNetV2 (`plant_disease_model.tflite`).
- **Strict Preprocessing Parity:** Implements `x_norm = (x / 127.5) - 1.0` to match training `mobilenet_v2.preprocess_input` bit-for-bit.
- **Automated Benchmarking:** Executes 20 inference cycles, discards warm-up epoch, and reports average latency in milliseconds.
- **Top-3 Ranked Predictions:** Displays ranked disease classes and confidence levels.

---
## Mobile ML Library Selection: React Native vs Flutter

| Feature | React Native (`react-native-fast-tflite`) | Flutter (`tflite_flutter`) |
| :--- | :--- | :--- |
| **Maintenance** | Actively maintained by Marc Rousavy (Nitro C++ modules) | Community maintained (dart:ffi bindings) |
| **Hardware Acceleration** | GPU delegate, Android NNAPI, iOS CoreML | GPU delegate via C++ bindings |
| **Developer Advantage** | Fits 4+ years of React Native / TypeScript experience directly | Requires learning Dart and Flutter widget lifecycle |

**Recommendation:** For an experienced React Native engineer, `react-native-fast-tflite` is the superior choice because it avoids introducing a new language runtime (Dart) while offering native C++ inference speeds.

---
## Running the App

```bash
cd mobile-app
npm install

# Run on Android
npx react-native run-android

# Run on iOS
cd ios && pod install && cd ..
npx react-native run-ios
```
