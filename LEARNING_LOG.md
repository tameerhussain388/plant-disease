# Engineering Learning Log & Interview Defense Notes

This log contains my personal technical notes, derivation summaries, and answers to fundamental deep learning questions in preparation for the **IPCVAI admissions interview**.

---

## 1. Mathematical Formulas & Architectural Derivations

### Convolutional Layer Parameter Count
For any 2D convolutional layer:
$$\text{Parameters} = (K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}$$
- **Why this formula?** Each filter kernel of spatial size $K_h \times K_w$ operates across all $C_{\text{in}}$ channels simultaneously. Adding 1 bias per filter and multiplying by $C_{\text{out}}$ filters yields the total weight tensor size.
- **Example from my baseline model (`conv1`):**  
  32 filters of size $3 \times 3$, 3 RGB input channels:  
  $(3 \times 3 \times 3 + 1) \times 32 = 28 \times 32 = 896 \text{ parameters}$.

### Why MobileNetV2 for Mobile? (Depthwise Separable Convolutions)
Standard convolution cost:
$$\text{Cost}_{\text{standard}} = H \times W \times C_{\text{in}} \times C_{\text{out}} \times K_h \times K_w$$
Depthwise Separable convolution splits this into two steps:
1. **Depthwise:** Filters each channel independently ($H \times W \times C_{\text{in}} \times K_h \times K_w$)
2. **Pointwise:** Combines channels via $1 \times 1$ conv ($H \times W \times C_{\text{in}} \times C_{\text{out}}$)

**Computation Reduction Ratio:**
$$\frac{\text{Cost}_{\text{separable}}}{\text{Cost}_{\text{standard}}} = \frac{1}{C_{\text{out}}} + \frac{1}{K_h \times K_w} \approx \frac{1}{9} \quad (\sim 88\% \text{ reduction in FLOPs})$$

---

## 2. Grad-CAM Mechanics & Mathematical Limitations
1. **Neuron Importance Weight ($\alpha_k^c$):**
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$
   Global average pooling over gradients measures how sensitive the predicted score $y^c$ is to the spatial activations of channel $k$.
2. **ReLU Activation:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
   ReLU is necessary because we only seek features that contribute positively to class $c$. Features that decrease the score belong to other categories and must be suppressed.
3. **Fundamental Limitation:**
   Grad-CAM derives its heatmap from the final convolutional layer (`out_relu`, spatial size $7 \times 7$). Upsampling a $7 \times 7$ grid to $224 \times 224$ produces coarse spatial heatmaps and cannot provide pixel-level segmentation boundaries.

---

## 3. Defense Q&A Preparation

### Q: Why freeze the backbone first, then fine-tune with a low learning rate?
Initially, the custom classification head consists of randomly initialized weights with high loss gradients. If the entire network is trained immediately, these large gradient updates propagate backward and destroy the well-tuned pretrained weights (catastrophic forgetting). Freezing ensures the head learns to align with the extracted features first. During fine-tuning, an ultra-small learning rate ($10^{-5}$) makes tiny, delicate adjustments to high-level filters without disrupting low-level primitive filters.

### Q: Why must Batch Normalization be kept in inference mode (`training=False`) during fine-tuning?
Batch Normalization maintains running mean and variance statistics across batches. During fine-tuning on a specialized domain with small mini-batches, updating these running statistics with noisy batch statistics damages the calibration learned on 1.4 million ImageNet images.

### Q: What is shortcut learning, and how did your experiments prove it?
Shortcut learning occurs when a model achieves high accuracy by exploiting spurious correlations rather than true causal features. In PlantVillage, leaves were photographed on uniform laboratory boards. In my background perturbation experiments, replacing the background with textured outdoor noise caused the model confidence to collapse, and Grad-CAM confirmed activations diverted from the lesion to the background borders.
