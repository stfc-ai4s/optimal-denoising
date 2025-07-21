"""
Unified interface for invex-based denoising.

Exposes:
- denoise          : auto dispatcher (slice/volume)
- denoise_slice    : 2D denoising
- denoise_volume   : (future) 3D volume-aware denoising
"""


__all__ = ['denoise', 'denoise_slice', 'denoise_volume']