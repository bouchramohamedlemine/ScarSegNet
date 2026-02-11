## Introduction




## Architecture 

<img width="3878" height="1391" alt="Blank diagram (88)" src="https://github.com/user-attachments/assets/9f6abe17-a96b-4041-963e-ef9eacb22a4c" />

## 2.1 ConvNeXt Encoder

To mitigate overfitting and enhance both local and global representation learning, we adopt **ConvNeXt** [11], a hybrid architecture combining transformer-inspired design with CNN inductive biases, as the image encoder. The pretrained **ConvNeXt-Base** is adapted for single-channel MRI by replacing the initial $3 \times 3$ convolution with $1 \times 1$.

---

## 2.2 Adaptive Squeeze-and-Excitation (ASE) Module

The standard SE block [12] recalibrates channels using global average pooling (GAP), which suppresses sparse activations from subtle scars. We propose replacing GAP with an adaptive pooling mechanism inspired by **T-Max-Avg** [13], allowing each channel to learn an optimal trade-off between max and average pooling. This enhances sensitivity to localized scar activations without amplifying background noise.

For a feature map $x_c \in \mathbb{R}^{H \times W}$, the top-$k$ activations are selected as:

$$
\mathbf{v}_c = \mathrm{TopK}(x_c, k)
$$

The gating mechanism produces a scalar pooled descriptor $z_c$ that summarises the overall activation strength of channel $c$ after adaptive weighting between local (max) and global (average) responses:

$$
z_c = g_c m_c + (1 - g_c) a_c
$$

where

$$
m_c = \max(\mathbf{v}_c), \quad
a_c = \frac{1}{k} \sum_j v_{c,j}
$$

The gating variable is defined as:

$$
g_c = \sigma\left(\frac{\min(\mathbf{v}_c) - T}{\tau}\right)
$$

and is controlled by a learnable threshold:

$$
T = \sigma(\theta)
$$

The adaptive channel descriptor vector:

$$
\mathbf{z}_{\text{adaptive}} = [z_1, \dots, z_C]
$$

replaces the GAP output in the SE formulation:

$$
\mathbf{s} = \sigma\left(W_2 \, \delta(W_1 \mathbf{z}_{\text{adaptive}})\right)
$$

The resulting ASE block enhances channel recalibration by emphasising scar-specific activations while maintaining global context.





## Results 



## References
