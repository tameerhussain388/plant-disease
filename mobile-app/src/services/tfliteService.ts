import { PLANT_VILLAGE_LABELS } from '../constants/labels';

export interface PredictionResult {
  label: string;
  confidence: number; // 0.0 to 1.0
  percentage: string; // e.g. "98.4%"
}

export interface BenchmarkMetrics {
  warmupMs: number;
  averageInferenceMs: number;
  iterations: number;
  deviceModel: string;
}

/**
 * Normalizes raw RGB image pixel buffer to [-1.0, 1.0]
 * Exact equivalent to tensorflow.keras.applications.mobilenet_v2.preprocess_input
 * 
 * Formula: x_norm = (x / 127.5) - 1.0
 * 
 * Common Bug Warning:
 * If this normalization is omitted (or if input is kept in [0, 255] or [0, 1]),
 * the model activations will saturate and predictions will be completely inaccurate!
 */
export function preprocessRGBPixels(rawRgbBytes: Uint8Array): Float32Array {
  const floatBuffer = new Float32Array(rawRgbBytes.length);
  for (let i = 0; i < rawRgbBytes.length; i++) {
    floatBuffer[i] = (rawRgbBytes[i] / 127.5) - 1.0;
  }
  return floatBuffer;
}

/**
 * Extracts Top-K classes from raw probability output
 */
export function getTopKPredictions(probabilities: Float32Array | number[], k: number = 3): PredictionResult[] {
  const indexed = Array.from(probabilities).map((prob, idx) => ({
    label: PLANT_VILLAGE_LABELS[idx] || `Class #${idx}`,
    confidence: prob,
    percentage: `${(prob * 100).toFixed(1)}%`
  }));

  indexed.sort((a, b) => b.confidence - a.confidence);
  return indexed.slice(0, k);
}
