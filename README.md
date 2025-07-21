# Optimal Denoising

## Project Overview

This project investigates the use of invex optimisation for denoising datasets that are affected by Gaussian, Poisson-Gaussian, and other noise models, especially images. These can be low-dose scans or X-ray or Cryo-Electron maps from light, laser and neutron sources. The objective is to recover high-fidelity images that are diagnostically viable by reducing noise and preserving structural and contrast information. This framework aims to provide a family of invex functions to be used as fidelity and regulariser terms with the support for multi-scale fidelity, and patch-based optimisation strategies.

The motivation stems from the challenge of denoising while avoiding artefacts (such as blockiness or blurring), especially under strong ill-posedness due to radiation dose constraints. The key advantage here is that this is purely model-based and does not require any training data, making it suitable for scenarios where high-quality ground-truth images are not available. However, this does not preclude parameter tuning based on available data.

## Objectives

- Denoise low-dose X-Ray and Cryo-EM maps using a mathematically grounded optimisation framework.
- Explore invex regularisers to preserve fine structural features.
- Evaluate the impact of patchwise versus full-slice optimisation strategies.
- Benchmark denoising performance against ground-truth images using relevant metrics, including, but not limited to, SNR, PSNR, SSNR, SSIM, gradient correlation (GC), contrast retention (CR) and other relevant metrics.
- Avoid over-smoothing and local inconsistencies often introduced by conventional regularisers.
- Add additional regularisers by extracting features from external networks (such as AutoEncoders)

## Methodology

We formulate the denoising task as an optimisation problem:

**Objective Function:**

$$
u^* = \arg\min_u \underbrace{\mathcal{D}(u, u_{\text{noisy}})}_{\text{Invex fidelity}} + \underbrace{\mathcal{R}(u)}_{\text{Invex regulariser}}
$$

The optimisation is expected to be carried out using either full-slice or patchwise strategies, with support for warm-starting and multi-scale fidelity enforcement. There are various ways the invex terms can be formulated, both for the fidelity term and for the regulariser.

The fidelity term and regulariser terms can be any invex function, but the choice would vary depending on the modality, and individual images or even datasets. Some examples include:

### Fidelity Functions

1. Charbonnier (PG)

Smooth approximation to L1 norm:

$$
\mathcal{D}(x, y) = \sum_i \sqrt{\left( \frac{x_i - y_i}{\sigma_i} \right)^2 + \epsilon^2}
$$



2. Welsch (Leclerc)

Outlier-suppressing robust penalty:

$$
\mathcal{D}(x, y) = \sum_i \left[ 1 - \exp\left( -\frac{1}{2} \left( \frac{x_i - y_i}{c \cdot \sigma_i} \right)^2 \right) \right]
$$



3. Gradient-Weighted Welsch

Enhanced structure preservation:

$$
\mathcal{D}(x, y) = \sum_i \left[ 1 - \exp\left( -\frac{1}{2} \left( \left( \frac{x_i - y_i}{c \cdot \sigma_i} \right)^2 \cdot |\nabla y_i|^2 \right) \right) \right]
$$



4. Contrast-Aware Welsch

Contrast-normalised fidelity:

$$
\mathcal{D}(x, y) = \sum_i \left[ 1 - \exp\left( -\frac{1}{2} \left( \frac{x_i - y_i}{c \cdot \sigma_i + \gamma \cdot |\nabla y_i|} \right)^2 \right) \right]
$$



5. Fractional Loss

Robust fractional loss:

$$
\mathcal{D}(x, y) = \sum_i \frac{(x_i - y_i)^2}{(x_i - y_i)^2 + \epsilon^2}
$$



6. PowerFrac Loss

Generalised fractional:

$$
\mathcal{D}(x, y) = \sum_i \frac{|x_i - y_i|^p}{|x_i - y_i|^p + \epsilon^p}
$$



7. Pseudo-Huber Loss

Smooth L1–L2 hybrid:

$$
\mathcal{D}(x, y) = \sum_i \delta^2 \left( \sqrt{1 + \left( \frac{x_i - y_i}{\delta} \right)^2 } - 1 \right)
$$



8. Tukey (Bisquare)

Robust to large outliers:

$$
\mathcal{D}(x, y) =
\sum_i \begin{cases}
\frac{c^2}{6} \left[1 - \left(1 - \left( \frac{r_i}{c} \right)^2 \right)^3 \right], & |r_i| \leq c \\
\frac{c^2}{6}, & |r_i| > c
\end{cases}
$$

where r_i = x_i - y_i



9. Log-Cosh Loss

Smooth approximation to absolute:

$$
\mathcal{D}(x, y) = \sum_i \log\left( \cosh\left( \frac{x_i - y_i}{\sigma} \right) \right)
$$



10. Lorentzian Loss

Bounded, smooth residual:

$$
\mathcal{D}(x, y) = \sum_i \log\left( 1 + \left( \frac{x_i - y_i}{\sigma} \right)^2 \right)
$$



### Regularisation Terms

11. LogTV

Edge-preserving smoothness:

$$
\mathcal{R}(x) = \lambda \sum_i \log\left( 1 + \frac{|\nabla x_i|^2}{\beta^2} \right)
$$



12. Hessian-Log Regulariser

Curvature-preserving regulariser:

$$
\mathcal{R}(x) = \lambda \sum_i \log\left( 1 + \frac{|\nabla^2 x_i|^2}{\beta^2} \right)
$$



13. Log Perona–Malik

Diffusion-inspired smoothing:

$$
\mathcal{R}(x) = \sum_i \log\left( 1 + \left( \frac{|\nabla x_i|}{\beta} \right)^2 \right)
$$



14. Huber Total Variation

Piecewise quadratic/sparse gradient:

$$
\mathcal{R}(x) = \sum_i \begin{cases}
\frac{1}{2} |\nabla x_i|^2, & |\nabla x_i| \leq \delta \\
\delta \left( |\nabla x_i| - \frac{1}{2} \delta \right), & \text{otherwise}
\end{cases}
$$



15. Arctangent TV

Highly edge-preserving:

$$
\mathcal{R}(x) = \sum_i \arctan\left( \frac{|\nabla x_i|}{\beta} \right)
$$



16. Entropy Prior

Useful for positive-valued data (e.g., PET):

$$
\mathcal{R}(x) = -\sum_i x_i \log(x_i + \epsilon)
$$


## Optimisers

#  Optimisers for Invex-Based Denoising

This document outlines key optimisation algorithms that can be used to solve denoising problems of the form:

$$
\min_{x} \; \mathcal{D}(x, y) + \lambda \, \mathcal{R}(x)
$$

where:
- \( \mathcal{D}(x, y) \) is a fidelity (data consistency) term,
- \( \mathcal{R}(x) \) is an invex (or nonconvex) regularisation term,
- \( \lambda > 0 \) balances the two components.



## 1. L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)

Quasi-Newton method suitable for smooth (possibly nonconvex) problems.

### Update Rule (implicit):

$$
x_{k+1} = x_k - \alpha_k H_k^{-1} \nabla f(x_k)
$$

- \( H_k^{-1} \) is an approximation to the inverse Hessian.
- No explicit Hessian is formed.
- Requires \( \nabla f(x_k) = \nabla \mathcal{D}(x_k) + \lambda \nabla \mathcal{R}(x_k) \)

This is often used when both terms are differentiable (e.g., Charbonnier + LogTV)



## 2. ADMM (Alternating Direction Method of Multipliers)

Splits fidelity and regulariser via auxiliary variable.

### Problem reformulated as:

$$
\min_{x,z} \; \mathcal{D}(x, y) + \lambda \, \mathcal{R}(z) \quad \text{s.t.} \; x = z
$$

### Augmented Lagrangian:

$$
\mathcal{L}_\rho(x, z, u) = \mathcal{D}(x, y) + \lambda \mathcal{R}(z) + \frac{\rho}{2} \|x - z + u\|^2 - \frac{\rho}{2} \|u\|^2
$$

### Iterations:

$$
\begin{aligned}
x^{k+1} &:= \arg\min_x \; \mathcal{D}(x, y) + \frac{\rho}{2} \|x - z^k + u^k\|^2 \\\\
z^{k+1} &:= \arg\min_z \; \lambda \mathcal{R}(z) + \frac{\rho}{2} \|x^{k+1} - z + u^k\|^2 \\\\
u^{k+1} &:= u^k + x^{k+1} - z^{k+1}
\end{aligned}
$$

This is often used when  the regulariser is non-differentiable or has a closed-form proximal operator.



## 3. iADMM (Inverse ADMM or Proximal ADMM)

Handles nonconvex regularisers with smoother convergence.

### Modified iteration with learned or structured proximal terms:

$$
z^{k+1} := \text{prox}_{\lambda \mathcal{R}/\rho} \left( x^{k+1} + u^k \right)
$$

Where:

$$
\text{prox}_{\tau \mathcal{R}}(v) = \arg\min_z \; \frac{1}{2} \|z - v\|^2 + \tau \mathcal{R}(z)
$$

This allows better behaviour with nonconvex or invex \( \mathcal{R}(x) \).



## 4. Proximal Gradient Descent

For problems where fidelity is smooth, regulariser has prox.

### Iteration:

$$
x^{k+1} = \text{prox}_{\tau \lambda \mathcal{R}} \left( x^k - \tau \nabla \mathcal{D}(x^k) \right)
$$

This is often used when  \( \mathcal{R} \) has a known proximal operator (e.g., TV, LogTV), but \( \mathcal{D} \) is differentiable.



## 5. FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

Accelerated version of proximal gradient for convex \( \mathcal{D}, \mathcal{R} \):

$$
\begin{aligned}
y^k &= x^k + \frac{t_{k-1} - 1}{t_k}(x^k - x^{k-1}) \\\\
x^{k+1} &= \text{prox}_{\tau \lambda \mathcal{R}} \left( y^k - \tau \nabla \mathcal{D}(y^k) \right) \\\\
t_{k+1} &= \frac{1 + \sqrt{1 + 4 t_k^2}}{2}
\end{aligned}
$$

This is often used when  speed is critical, and regulariser is convex or “nearly convex”.



## 6. Primal-Dual (PDHG / Chambolle–Pock)

Solves the saddle-point problem:

$$
\min_x \max_p \; \langle Kx, p \rangle + \lambda \mathcal{R}^*(p) + \mathcal{D}(x, y)
$$

Works well for TV, LogTV, and when regulariser is expressed via dual form.



#  Summary

| Optimiser     | Smoothness Required | Handles Nondiff. | Invex-Safe | Typical Use                     |
|---------------|----------------------|------------------|------------|----------------------------------|
| L-BFGS        | Yes                  | No               | Yes        | PG+LogTV                         |
| ADMM          | No                   | Yes              | Yes        | Welsch+TV                        |
| iADMM         | No                   | Yes              | Yes        | Grad-Welsch+TV                   |
| Prox. Grad    | Yes (for D)          | Yes (for R)      | Yes        | PG + logTV                       |
| FISTA*        | Yes                  | Yes              | Conditional| Smooth PG + convex TV            |
| Primal-Dual   | Yes (for D)          | Yes              | Yes        | Imaging with dual constraints    |

Please note that there are variants of FISTA.

## Features

* Invex regularisers: LogTV, Charbonnier, Hessian-based
* Fidelity terms: L2, Welsch, Poisson-Gaussian
* Solvers: ADMM, L-BFGS, FISTA* (Note the variants)
* Modular penalties and noise models (CryoEM, X-ray)
* Multi-modality test support (CryoEM, X-ray)
* Perceptual and ROI-aware metrics

## Citation

Please acknowledge this repository as:

> Jeyan Thiyagalingam (2025), "A Framework for Optimal Denoising", GitHub Repository: https://github.com/stfc-ai4s/optimal-denoising


## Repository Structure

├── core/           # Invex denoising logic
├── interfaces/     # Model wrappers or high-level classes
├── penalties/      # Fidelity and regulariser terms
├── solvers/        # Optimisation strategies (LBFGS, ADMM, FISTA, etc.)
├── utils/          # Various helper functions
├── metrics/        # Quality metrics (SNR, SSIM, GC, etc.)


## Evaluation

Evaluation of denoising is a complex topic. While evaluation of denoising performance (as in original benchmark paper and benchmarks) can be based on specific metrics (see below) based on single image slice, actual performance has to be end-to-end or at least downstream-based (say segmentation or volumetric projection). These will be considered too, thanks to some of the latest FMs, like SAM or YoLO or SAMv2 etc. Example metrics are:

* SNR
* PSNR
* SSNR
* GC
* SSIM
* CR
* Visual inspection of anatomical fidelity and contrast $20 \log_{10}(\frac{|u_{\text{clean}}|_2}{|u - u_{\text{clean}}|_2})$


## Limitations

- PSNR/SSIM gains remain modest in many cases.
- Patchwise denoising often introduces undesirable artefacts.
- Manual tuning of hyperparameters (e.g., λ values) is computationally expensive and may not generalise.
- Over-regularisation leads to over-smoothing or contrast loss.

## License

See LICENSE file

## Getting Started

```bash
# Clone repo and install dependencies
pip install -r requirements.txt

```

## Basic Usage


```python
from interfaces.denoise import denoise
from penalties import FidelityWelschPG, RegulariserLogTV

# load the image, and call it y
y = load_image('path/to/image.png')
x_hat = denoise(
    y,
    fidelity=FidelityWelschPG(c=3.0),
    regulariser=[RegulariserLogTV(lambda_=0.2, beta=3.0)]
)
evaluate_all(x_hat, gt)
```


## Background Material

The thoughts here are based or motivated by the following: (and  the absence of any invex codes or implementations for denoising)

* J Liang, CB Schönlieb, Faster FISTA
* J Liang, T Luo, CB Schonlieb, Improving “fast iterative shrinkage-thresholding algorithm”: Faster, smarter, and greedier
* S Arridge, P Maass, O Öktem, CB Schönlieb, Solving inverse problems using data-driven models
* S Lunz, O Öktem, CB Schönlieb, Adversarial regularizers in inverse problems
* K Wei, A Aviles-Rivero, J Liang, Y Fu, CB Schönlieb, H Huang, Tuning-free plug-and-play proximal algorithm for inverse imaging problems
* Invexity and Optimization, Springer
