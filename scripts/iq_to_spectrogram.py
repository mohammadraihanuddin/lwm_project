"""Convert IQ samples to 128x128 magnitude spectrogram for LWM-Spectro MoE (same spatial size as demo_data)."""
import numpy as np
from scipy import signal
from scipy.ndimage import zoom


def iq_to_spectrogram_magnitude(
    iq: np.ndarray,
    target_size: tuple[int, int] = (128, 128),
    nperseg: int = 64,
    noverlap: int | None = None,
) -> np.ndarray:
    """
    Convert IQ samples to magnitude spectrogram.

    Args:
        iq: Shape (L, 2) for I and Q, or (L,) complex
        target_size: (height, width) of output
        nperseg: STFT segment length
        noverlap: STFT overlap (default nperseg // 2)

    Returns:
        Spectrogram (H, W), float32 magnitude.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if iq.ndim == 2 and iq.shape[1] == 2:
        iq_complex = iq[:, 0] + 1j * iq[:, 1]
    else:
        iq_complex = np.asarray(iq, dtype=np.complex64).flatten()

    f, t, Z = signal.stft(iq_complex, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    n_f, n_t = Z.shape
    if n_f < 2 or n_t < 2:
        return np.zeros(target_size, dtype=np.float32)

    magnitude = np.abs(Z).astype(np.float32)
    h_out, w_out = target_size
    zoom_f = h_out / n_f
    zoom_t = w_out / n_t
    out = zoom(magnitude, (zoom_f, zoom_t), order=1)
    return out.astype(np.float32)
