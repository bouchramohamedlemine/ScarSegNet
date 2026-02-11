# Introduction




# Architecture 

<img width="3878" height="1391" alt="Blank diagram (88)" src="https://github.com/user-attachments/assets/9f6abe17-a96b-4041-963e-ef9eacb22a4c" />

## 2.1 ConvNeXt Encoder

To mitigate overfitting and enhance both local and global representation learning, we adopt **ConvNeXt** [11], a hybrid architecture combining transformer-inspired design with CNN inductive biases, as the image encoder. The pretrained **ConvNeXt-Base** is adapted for single-channel MRI by replacing the initial $3 \times 3$ convolution with $1 \times 1$.

---

## 2.2 Adaptive Squeeze-and-Excitation (ASE) Module

The standard SE block [12] recalibrates channels using global average pooling (GAP), which suppresses sparse activations from subtle scars. We propose replacing GAP with an adaptive pooling mechanism inspired by **T-Max-Avg** [13], allowing each channel to learn an optimal trade-off between max and average pooling. This enhances sensitivity to localised scar activations without amplifying background noise.

For a feature map $x_c \in \mathbb{R}^{H \times W}$, the top-k activations are selected as $\mathbf{v}_c = \mathrm{TopK}(x_c, k)$.

The gating mechanism produces a scalar pooled descriptor $z_c$ that summarises the overall activation strength of channel $c$ after adaptive weighting between local (max) and global (average) responses: $z_c = g_c m_c + (1 - g_c) a_c$, where $m_c = \max(\mathbf{v}_c) \text{ and } a_c = (1/k) \sum_j v_{c,j}$.


The gating variable is defined as $g_c = \sigma\left(\frac{\min(\mathbf{v}_c) - T}{\tau}\right)$ and is controlled by a learnable threshold $T = \sigma(\theta)$.

The adaptive channel descriptor vector $\mathbf{z}_{\text{adaptive}} = [z_1, \dots, z_C]$ replaces the GAP output in the SE formulation $\mathbf{s} = \sigma\left(W_2 \, \delta(W_1 \mathbf{z}_{\text{adaptive}})\right)$.

The resulting ASE block enhances channel recalibration by emphasizing scar-specific activations while maintaining global context.


## 2.3 Contextual Attention Gate (CAG)

The original Attention Gate (AG) [14] enhances focus on relevant regions but its limited local receptive field restricts the ability to capture broader contextual cues necessary for detecting large scars. To address this, we extend AG with a contextual branch that employs a $3 \times 3$ convolution with dilation 3, expanding the receptive field to $7 \times 7$ while preserving local detail.

Given decoder gating signal $\mathbf{g}$ and encoder feature $\mathbf{x}$, both are projected and fused: $\mathbf{f} = \mathrm{ReLU}(\mathrm{BN}(W_g * \mathbf{g}) + \mathrm{BN}(W_x * \mathbf{x}))$.

The final attention mask combines local and contextual responses: $\psi = \sigma(\mathrm{BN}(W_{\text{local}} * \mathbf{f} + W_{\text{global}} * \mathbf{f}))$, and the attended skip features are $\tilde{\mathbf{x}} = \psi \odot \mathbf{x}$.

This dual-branch design improves segmentation of large, spatially diffuse scars without the overhead of global self-attention.

---

## 2.4 Myocardium-Constrained Loss (MCL)

Since scars are confined to the myocardium, we aim to make the model explicitly aware of where predictions are anatomically valid. Standard loss formulations provide only implicit spatial supervision, which may lead to false positives in regions outside the myocardium. Therefore, we introduce a **Myocardium-Constrained Loss (MCL)**, which penalises scar predictions outside the myocardium mask to ensure anatomically plausible segmentation.

Let $\mathbf{z}$ denote scar logits and $\mathbf{M}$ the myocardium mask. The constraint term is $\mathcal{L}_{\text{out}} = \frac{1}{HW} \sum_{i,j} \sigma(z_{ij})(1 - M_{ij})$, where $\sigma(\cdot)$ is the sigmoid function.

The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{bce}} + \mathcal{L}_{\text{dice}} + \mathcal{L}_{\text{out}}$.

We experimented with different weighting combinations among these terms and found that equal weighting gives the most stable performance. This formulation explicitly constrains learning to the myocardium region from the beginning of training, enforcing anatomically consistent predictions without requiring a separate myocardium model.


# Results 



# References
