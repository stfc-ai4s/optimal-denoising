import csv
import io

def compare_metrics_to_csv(noisy_metrics: dict, denoised_metrics: dict) -> str:
    """
    Compare two metric dictionaries and return a CSV-formatted string.
    Useful for comparing noisy vs denoised results.

    Args:
        noisy_metrics (dict): Output of evaluate_all() for the noisy image
        denoised_metrics (dict): Output of evaluate_all() for the denoised image

    Returns:
        str: CSV-formatted string
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    
    writer.writerow(["Metric", "Noisy", "Denoised", "Delta (Denoised - Noisy)"])

    keys = sorted(set(noisy_metrics) | set(denoised_metrics))
    for key in keys:
        n_val = noisy_metrics.get(key)
        d_val = denoised_metrics.get(key)

        # Handle SSNR arrays and None values
        if isinstance(n_val, tuple) or isinstance(d_val, tuple):
            continue  # Skip large or complex metrics like SSNR
        if n_val is None and d_val is None:
            delta = ''
        elif n_val is None:
            delta = 'N/A'
        elif d_val is None:
            delta = 'N/A'
        else:
            try:
                delta = float(d_val) - float(n_val)
            except Exception:
                delta = 'N/A'

        writer.writerow([key, n_val, d_val, delta])

    return buffer.getvalue()


def print_metrics_comparison(
    noisy_metrics: dict,
    denoised_metrics: dict,
    show_delta: bool = True
):
    """
    Print side-by-side comparison of noisy vs denoised metric dictionaries.

    Args:
        noisy_metrics:     Dict[str, float] from noisy image evaluation
        denoised_metrics:  Dict[str, float] from denoised image evaluation
        show_delta:        Whether to show improvement (denoised - noisy)
    """
    if show_delta:
        print("{:<22} {:>12} {:>12} {:>12}".format("Metric", "Noisy", "Denoised", "Δ"))
        print("-" * 60)
    else:
        print("{:<22} {:>12} {:>12}".format("Metric", "Noisy", "Denoised"))
        print("-" * 46)

    for key in sorted(noisy_metrics.keys()):
        val1 = noisy_metrics[key]
        val2 = denoised_metrics.get(key)

        if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
            continue  # Skip complex or non-scalar metrics

        if show_delta:
            delta = val2 - val1
            print("{:<22} {:>12.4f} {:>12.4f} {:>12.4f}".format(key, val1, val2, delta))
        else:
            print("{:<22} {:>12.4f} {:>12.4f}".format(key, val1, val2))