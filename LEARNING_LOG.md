# Technical Notes & Interview Preparation

Personal study notes and mathematical derivations for interview preparation.

---

## 1. Formulas & Parameters

### Conv2D Parameter Formula
$$\text{Params} = (K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}$$
- Each kernel covers the spatial size $K_h \times K_w$ across all input channels $C_{\text{in}}$.
- $+1$ is the bias for that filter.
- Multiplied by $C_{\text{out}}$ (number of output filters).

*Example:* Layer 1 in my baseline model has 32 filters of size $3 \times 3$ on 3-channel input:
$$(3 \times 3 \times 3 + 1) \times 32 = 28 \times 32 = 896 \text{ parameters}$$

---

### MobileNetV2 Depthwise Separable Convolutions
- **Standard conv cost:** $H \times W \times C_{\text{in}} \times C_{\text{out}} \times K_h \times K_w$
- **Depthwise separable cost:** $H \times W \times C_{\text{in}} \times K_h \times K_w + H \times W \times C_{\text{in}} \times C_{\text{out}}$
- **Reduction ratio:** $\approx \frac{1}{K_h \times K_w} \approx \frac{1}{9}$ (about 88% fewer operations).

---

### Grad-CAM Formulation
1. Importance weight for channel $k$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$
   Global average pooling over gradients to find how sensitive class score $y^c$ is to feature map $A^k$.

2. Heatmap:
   $$L_{\text{Grad-CAM}}^c = \max\left(0, \sum_k \alpha_k^c A^k\right)$$
   ReLU is applied because we only care about positive influences on class $c$.

---

## 2. Common Interview Questions & My Explanations

### Why freeze then fine-tune?
The new classification head starts with random weights, producing large initial gradients. If the entire network was updated immediately, these large gradients would destroy the pretrained ImageNet weights (catastrophic forgetting). Freezing the backbone trains the head first. Once the head is aligned, unfreezing top layers with a small learning rate ($10^{-5}$) adapts high-level features safely.

### Why keep BatchNorm frozen (`training=False`) during fine-tuning?
Batch normalization tracks running mean and variance. If allowed to update during fine-tuning with small batches, the noisy batch statistics ruin the calibrated statistics learned over millions of ImageNet images.

### What is shortcut learning?
When a neural network relies on spurious background correlations rather than true causal features. In PlantVillage, leaves were placed on lab tables. The network learned to associate certain background paper/wood patterns with specific classes, which causes accuracy drops when tested on real field photos.
