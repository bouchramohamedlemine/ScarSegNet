# Introduction

Cardiac magnetic resonance with late gadolinium enhancement (LGE-CMR) is the reference standard for detecting myocardial scarring, yet manual segmentation remains labour-intensive and prone to inter-observer variability. We propose ScarSegNet, a novel deep learning framework for automated myocardial scar segmentation, specifically designed to handle the irregular and heterogeneous appearance of scar tissue in challenging, limited-size clinical datasets. ScarSegNet integrates a ConvNeXt encoder with attention mechanisms in both encoder and skip pathways, a learnable T-Max-Avg pooling module for adaptive channel recalibration, and a myocardium-constrained loss to enforce anatomical plausibility without requiring myocardium segmentation. Evaluated on a clinically acquired LGE-CMR dataset of 400 patients, ScarSegNet achieved 2.1% higher scar Dice score than prior SOTAs. It also demonstrated strong agreement with manual scar burden quantification (Pearson’s r = 0.901) and correctly identified 96.2% of high-risk patients above the clinically significant 5% scar threshold.

---

# Architecture 

<img width="3878" height="1391" alt="Blank diagram (88)" src="https://github.com/user-attachments/assets/9f6abe17-a96b-4041-963e-ef9eacb22a4c" />

---

## 2.1 ConvNeXt Encoder

To mitigate overfitting and enhance both local and global representation learning, we adopt **ConvNeXt** [11], a hybrid architecture combining transformer-inspired design with CNN inductive biases, as the image encoder. The pretrained **ConvNeXt-Base** is adapted for single-channel MRI by replacing the initial $3 \times 3$ convolution with $1 \times 1$.



## 2.2 Adaptive Squeeze-and-Excitation (ASE) Module

The standard SE block [2] recalibrates channels using global average pooling (GAP), which suppresses sparse activations from subtle scars. We propose replacing GAP with an adaptive pooling mechanism inspired by **T-Max-Avg** [3], allowing each channel to learn an optimal trade-off between max and average pooling. This enhances sensitivity to localised scar activations without amplifying background noise.

For a feature map $x_c \in \mathbb{R}^{H \times W}$, the top-k activations are selected as $\mathbf{v}_c = \mathrm{TopK}(x_c, k)$.

The gating mechanism produces a scalar pooled descriptor $z_c$ that summarises the overall activation strength of channel $c$ after adaptive weighting between local (max) and global (average) responses: $z_c = g_c m_c + (1 - g_c) a_c$, where $a_c = \frac{1}{k} \sum_j v_{c,j}$ and $m_c = \max(\mathbf{v}_c)$. The gating variable is controlled by a learnable threshold $T = \sigma(\theta)$ and defined as $g_c = \sigma\left(\frac{\min(\mathbf{v}_c) - T}{\tau}\right)$. 

The adaptive channel descriptor vector $z_adaptive = [z_1, \dots, z_C]$ replaces the GAP output in the SE formulation $\mathbf{s} = \sigma\left(W_2 \delta(W_1 \mathbf{z}_{\mathrm{adaptive}}\right))$. The resulting ASE block enhances channel recalibration by emphasising scar-specific activations while maintaining global context.


## 2.3 Contextual Attention Gate (CAG)

The original Attention Gate (AG) [4] enhances focus on relevant regions, but its limited local receptive field restricts the ability to capture broader contextual cues necessary for detecting large scars. To address this, we extend AG with a contextual branch that employs a $3 \times 3$ convolution with dilation 3, expanding the receptive field to $7 \times 7$ while preserving local detail.

Given decoder gating signal $\mathbf{g}$ and encoder feature $\mathbf{x}$, both are projected and fused: $\mathbf{f} = \mathrm{ReLU}(\mathrm{BN}(W_g * \mathbf{g}) + \mathrm{BN}(W_x * \mathbf{x}))$.

The final attention mask combines local and contextual responses: $\psi = \sigma(\mathrm{BN}(W_{\text{local}} * \mathbf{f} + W_{\text{global}} * \mathbf{f}))$, and the attended skip features are $\tilde{\mathbf{x}} = \psi \odot \mathbf{x}$.

This dual-branch design improves segmentation of large, spatially diffuse scars without the overhead of global self-attention.

 

## 2.4 Myocardium-Constrained Loss (MCL)

Since scars are confined to the myocardium, we aim to make the model explicitly aware of where predictions are anatomically valid. Standard loss formulations provide only implicit spatial supervision, which may lead to false positives in regions outside the myocardium. Therefore, we introduce a **Myocardium-Constrained Loss (MCL)**, which penalises scar predictions outside the myocardium mask to ensure anatomically plausible segmentation.

Let $\mathbf{z}$ denote scar logits and $\mathbf{M}$ the myocardium mask. The constraint term is $\mathcal{L}_{\text{out}} = \frac{1}{HW} \sum\_{i,j} \sigma(z\_{i,j})(1 - M\_{i,j})$, where $\sigma(\cdot)$ is the sigmoid function. 

The total loss is $L_{\text{total}} = L_{\text{bce}} + L_{\text{dice}} + L_{\text{out}}$. We experimented with different weighting combinations among these terms and found that equal weighting gives the most stable performance. This formulation explicitly constrains learning to the myocardium region from the beginning of training, enforcing anatomically consistent predictions without requiring a separate myocardium model.


---

**Table 1.** Comparison of segmentation performance on the test set, reported as Dice score and HD95 (mean $\pm$ SD).

| Model              | Scar Dice score | HD95 (px) |
|--------------------|------------------|------------|
| ScarSegNet (ours)  | $\mathbf{0.687 \pm 0.107}$ | $\mathbf{4.13 \pm 3.55}$ |
| nnU-Net            | $0.666 \pm 0.136$ | $15.96 \pm 13.26$ |
| ScarNet            | $0.381 \pm 0.311$ | $12.47 \pm 6.54$ |
| SAM-Med2D          | $0.486 \pm 0.489$ | $28.95 \pm 14.28$ |



# Results 

<img width="236" height="159" alt="Screenshot 2026-02-11 at 21 49 16" src="https://github.com/user-attachments/assets/91feded8-6a1d-406e-b0b2-056137aecdc9" />

<br>

<img width="235" height="152" alt="Screenshot 2026-02-11 at 21 49 36" src="https://github.com/user-attachments/assets/73f32b15-b8b9-412b-b993-507fdc89ef2f" />



---

## References

[1] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 11976–11986.

[2] J. Hu, L. Shen, and G. Sun, “Squeeze-and-Excitation Networks,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7132–7141.

[3] L. Zhao and Z. Zhang, “An Improved Pooling Method for Convolutional Neural Networks,” *Scientific Reports*, vol. 14, no. 1, p. 1589, 2024.

[4] J. Schlemper, O. Oktay, M. Schaap, M. Heinrich, B. Kainz, B. Glocker, and D. Rueckert, “Attention Gated Networks: Learning to Leverage Salient Regions in Medical Images,” *arXiv preprint arXiv:1808.08114*, 2018.


