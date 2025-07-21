"""
Unified image-quality metrics for denoising and reconstruction benchmarking.

Supports CryoEM, X-ray, and other grayscale imaging modalities.

Dependencies:
    - torch, numpy, scipy, torchmetrics (with SSIM and optional LPIPS)

Core Features:
    - Handles both NumPy arrays and PyTorch tensors
    - Works on 2D/3D slices or volumes
    - Methods include:
        * SNR / PSNR / SSNR (log or linear radial bins)
        * SSIM / Multi-Scale SSIM (if available)
        * Gradient Correlation (GC)
        * Contrast Retention (CR)
        * CNR
        * LPIPS (if available)
        * MAE, nRMSE and their ROI-aware variants

Example Usage:
    >>> snr = Metrics.snr(denoised, ground_truth)
    >>> gc  = Metrics.gradient_corr(denoised, gt)
    >>> roi_mae = Metrics.masked_mae(denoised, gt, lesion_mask)
"""

import math
from typing import Tuple, Optional, Union, Sequence
import numpy as np
import torch
import scipy.ndimage as ndi
from torchmetrics.image import StructuralSimilarityIndexMeasure as TMSSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as TMMS_SSIM

try:
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
    _lpips_available = True
except ImportError:
    _lpips_available = False


def _to_numpy(a):
    if isinstance(a, np.ndarray):
        return a.astype(np.float32, copy=False)
    if torch.is_tensor(a):
        return a.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(a, dtype=np.float32)


def _to_tensor(a, device=None):
    if torch.is_tensor(a):
        return a.to(device) if device else a
    return torch.as_tensor(_to_numpy(a), device=device)


def _fft2d(x: np.ndarray):
    F = np.fft.fftshift(np.fft.fft2(x))
    return (F.real**2 + F.imag**2)


def _radial_profile(power: np.ndarray, n_bins: int = 50):
    h, w = power.shape
    y, x = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)
    max_r = int(r.max())
    prof = np.bincount(r.ravel(), power.ravel(), minlength=max_r + 1)
    cnts = np.bincount(r.ravel(), minlength=max_r + 1)
    idx = cnts > 0
    return np.arange(max_r + 1)[idx], prof[idx] / cnts[idx]


def _log_radial_profile(power: np.ndarray, n_bins: int = 50):
    h, w = power.shape
    y, x = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = np.log1p(r)
    r_idx = (r / r.max() * (n_bins - 1)).astype(np.int32)
    prof = np.bincount(r_idx.ravel(), power.ravel(), minlength=n_bins)
    count = np.bincount(r_idx.ravel(), minlength=n_bins)
    count[count == 0] = 1
    return np.linspace(0, 1, n_bins), prof / count


class Metrics:
    _metric_whitelist = {
        "snr", "psnr", "nrmse", "mae", "ssim", "ms_ssim", "gradient_corr",
        "contrast_retention", "ssnr", "lpips", "cnr",
        "masked_mae", "masked_nrmse"
    }

    @staticmethod
    def snr(est, ref, mask: Optional[np.ndarray] = None) -> float:
        """
        Signal-to-Noise Ratio (SNR) in decibels.
        Suitable for X-ray, and general grayscale imaging.
        """
        est_np, ref_np = _to_numpy(est), _to_numpy(ref)
        if mask is None:
            mask = np.ones_like(ref_np, dtype=bool)
        diff = est_np[mask] - ref_np[mask]
        signal_power = np.mean(ref_np[mask]) ** 2
        noise_power = np.var(diff)
        return 10.0 * math.log10(signal_power / max(noise_power, 1e-12))

    @staticmethod
    def psnr(est, ref, data_range: Optional[float] = None) -> float:
        """
        Peak Signal-to-Noise Ratio (PSNR).
        Common in image processing but not contrast-sensitive.
        Suitable for general grayscale imaging.
        """
        est_np, ref_np = _to_numpy(est), _to_numpy(ref)
        mse = np.mean((est_np - ref_np) ** 2)
        if data_range is None:
            data_range = ref_np.max() - ref_np.min()
        return 20.0 * math.log10(data_range / math.sqrt(max(mse, 1e-12)))

    @staticmethod
    def mae(est, ref) -> float:
        """
        Mean Absolute Error (MAE).
        Suitable across all grayscale imaging modalities.
        """
        est_np, ref_np = _to_numpy(est), _to_numpy(ref)
        return float(np.mean(np.abs(est_np - ref_np)))

    @staticmethod
    def nrmse(est, ref) -> float:
        """
        Normalized Root Mean Square Error (NRMSE).
        Lower is better. Modality-agnostic.
        """
        est_np, ref_np = _to_numpy(est), _to_numpy(ref)
        rmse = np.sqrt(np.mean((est_np - ref_np) ** 2))
        denom = np.mean(ref_np ** 2) ** 0.5 + 1e-8
        return rmse / denom

    @staticmethod
    def masked_mae(est, ref, mask: np.ndarray) -> float:
        """
        Mean Absolute Error (MAE) over ROI mask.
        Suitable for lesion/tissue-level evaluation.
        """
        est_np, ref_np = _to_numpy(est), _to_numpy(ref)
        mask = mask.astype(bool)
        return float(np.abs(est_np[mask] - ref_np[mask]).mean())

    @staticmethod
    def masked_nrmse(est, ref, mask: np.ndarray) -> float:
        """
        NRMSE over ROI mask.
        Suitable for region-specific evaluation.
        """
        est_np, ref_np = _to_numpy(est), _to_numpy(ref)
        mask = mask.astype(bool)
        rmse = np.sqrt(np.mean((est_np[mask] - ref_np[mask]) ** 2))
        denom = np.sqrt(np.mean(ref_np[mask] ** 2)) + 1e-8
        return rmse / denom

    @staticmethod
    def ssnr(est, ref, n_bins: int = 50, log: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spectral Signal-to-Noise Ratio (SSNR).
        Suitable for CryoEM, and microscopy.
        """
        est_np, ref_np = _to_numpy(est), _to_numpy(ref)
        noise_np = est_np - ref_np
        pwr_ref = _fft2d(ref_np)
        pwr_noise = _fft2d(noise_np) + 1e-12
        profile_fn = _log_radial_profile if log else _radial_profile
        f_ref, P_ref = profile_fn(pwr_ref, n_bins)
        _, P_noise = profile_fn(pwr_noise, n_bins)
        ssnr = 10.0 * np.log10(P_ref / P_noise)
        return f_ref, ssnr

    @staticmethod
    def gradient_corr(est, ref) -> float:
        """
        Gradient Correlation (GC).
        Measures edge alignment. Suitable for microscopy.
        """
        if torch.is_tensor(est) and torch.is_tensor(ref):
            k = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=est.dtype, device=est.device) / 4.0
            kx, ky = k[None, None], k[None, None].transpose(-1, -2)
            gx_e = torch.nn.functional.conv2d(est[None, None], kx, padding=1)
            gy_e = torch.nn.functional.conv2d(est[None, None], ky, padding=1)
            gx_r = torch.nn.functional.conv2d(ref[None, None], kx, padding=1)
            gy_r = torch.nn.functional.conv2d(ref[None, None], ky, padding=1)
            dot = (gx_e * gx_r + gy_e * gy_r).sum()
            norm = torch.linalg.norm(torch.stack((gx_e, gy_e))) * torch.linalg.norm(torch.stack((gx_r, gy_r))) + 1e-12
            return float((dot / norm).item())

        est_np, ref_np = _to_numpy(est), _to_numpy(ref)
        gx_e = ndi.sobel(est_np, axis=-1)
        gy_e = ndi.sobel(est_np, axis=-2)
        gx_r = ndi.sobel(ref_np, axis=-1)
        gy_r = ndi.sobel(ref_np, axis=-2)
        dot = (gx_e * gx_r + gy_e * gy_r).sum()
        norm = math.sqrt((gx_e**2 + gy_e**2).sum() * (gx_r**2 + gy_r**2).sum()) + 1e-12
        return float(dot / norm)

    @staticmethod
    def ssim(est, ref) -> float:
        """
        Structural Similarity Index (SSIM) in [0,1].
        Good for X-ray, and general imaging.
        """
        if (isinstance(est, np.ndarray) and est.ndim == 3) or (torch.is_tensor(est) and est.ndim == 3):
            return float(np.mean([Metrics.ssim(e, r) for e, r in zip(est, ref)]))

        device = est.device if torch.is_tensor(est) else 'cpu'
        x = _to_tensor(est, device=device).unsqueeze(0).unsqueeze(0)
        y = _to_tensor(ref, device=device).unsqueeze(0).unsqueeze(0)
        d_min = torch.min(torch.min(x), torch.min(y))
        d_max = torch.max(torch.max(x), torch.max(y))
        data_range = (d_max - d_min).clamp(min=1e-3)
        x = (x - d_min) / data_range
        y = (y - d_min) / data_range
        ssim_mod = TMSSIM(data_range=1.0).to(device)
        return float(ssim_mod(x, y).item())

    @staticmethod
    def ms_ssim(est, ref) -> float:
        """
        Multi-Scale SSIM.
        Suitable for natural images, and microscopy.
        """
        device = est.device if torch.is_tensor(est) else 'cpu'
        x = _to_tensor(est, device=device).unsqueeze(0).unsqueeze(0)
        y = _to_tensor(ref, device=device).unsqueeze(0).unsqueeze(0)
        d_min = torch.min(torch.min(x), torch.min(y))
        d_max = torch.max(torch.max(x), torch.max(y))
        data_range = (d_max - d_min).clamp(min=1e-3)
        x = (x - d_min) / data_range
        y = (y - d_min) / data_range
        model = TMMS_SSIM(data_range=1.0).to(device)
        return float(model(x, y).item())

    @staticmethod
    def contrast_retention(est, ref, lo_pct: float = 5, hi_pct: float = 95) -> float:
        """
        Contrast Retention using percentile-based dynamic range.
        Lower is better. Useful for grayscale imaging.
        """
        est_np, ref_np = _to_numpy(est), _to_numpy(ref)
        p_lo_r, p_hi_r = np.percentile(ref_np, [lo_pct, hi_pct])
        p_lo_e, p_hi_e = np.percentile(est_np, [lo_pct, hi_pct])
        range_ref = p_hi_r - p_lo_r + 1e-6
        range_est = p_hi_e - p_lo_e
        return float(abs(range_est - range_ref) / range_ref)

    @staticmethod
    def cnr(est, ref, signal_mask: np.ndarray, background_mask: np.ndarray) -> float:
        """
        Contrast-to-Noise Ratio (CNR) between signal and background regions.
        Suitable for CryoEM.
        """
        est_np = _to_numpy(est)
        mean_sig = est_np[signal_mask].mean()
        mean_bg = est_np[background_mask].mean()
        std_bg = est_np[background_mask].std() + 1e-12
        return float(abs(mean_sig - mean_bg) / std_bg)

    @staticmethod
    def lpips(est, ref, net='alex') -> float:
        """
        Learned Perceptual Image Patch Similarity (LPIPS).
        Requires pretrained model. Suitable for perceptual evaluation.
        """
        if not _lpips_available:
            raise ImportError("Install LPIPS support via `pip install torchmetrics[image]`")
        model = LPIPS(net_type=net).to('cpu')
        x = _to_tensor(est).unsqueeze(0).repeat(1, 3, 1, 1)
        y = _to_tensor(ref).unsqueeze(0).repeat(1, 3, 1, 1)
        return float(model(x, y).item())

    @staticmethod
    def evaluate_all(est, ref, ignore: Optional[Sequence[str]] = None) -> dict[str, float | None]:
        """
        Evaluate all whitelisted metrics between estimated and reference images.
        Handles both 2D and 3D volumes.
        """
        ignore = set(ignore or [])
        est_np, ref_np = _to_numpy(est), _to_numpy(ref)
        metrics: dict[str, float | None] = {}

        all_metrics = {
            name: getattr(Metrics, name)
            for name in dir(Metrics)
            if name in Metrics._metric_whitelist and name not in ignore and callable(getattr(Metrics, name))
        }

        def _safe(fn, e, r) -> float | None:
            try:
                return fn(e, r)
            except Exception:
                return None

        if est_np.ndim == 3:
            for name, fn in all_metrics.items():
                vals = [_safe(fn, e, r) for e, r in zip(est_np, ref_np)]
                filtered = [v for v in vals if v is not None]
                metrics[name] = float(np.mean(filtered)) if filtered else None
        else:
            for name, fn in all_metrics.items():
                metrics[name] = _safe(fn, est_np, ref_np)

        return metrics