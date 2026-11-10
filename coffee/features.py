from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import find_peaks


def image_features(path):
    with Image.open(path) as im:
        rgb = im.convert('RGB').resize((224, 224))
        arrays = [np.asarray(rgb) / 255., np.asarray(rgb.convert('HSV')) / 255.]
    values = []
    for array in arrays:
        for channel in array.transpose(2, 0, 1):
            hist = np.histogram(channel, bins=16, range=(0, 1))[0]
            values.extend(hist / hist.sum())
            values.extend([channel.mean(), channel.std()])
    return np.asarray(values, dtype=np.float64)


def audio_features(path):
    with open(path, 'rb') as f:
        if f.read(7) == b'version':
            raise ValueError(f'{path}: Git LFS pointer, run python -m coffee.download and use data/audio-lfs/datasets')
    sr, raw = wavfile.read(Path(path))
    if raw.size == 0:
        raise ValueError('Empty audio')
    x = raw.astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        if raw.dtype == np.uint8:
            x = (x - 128) / 128
        else:
            x /= max(abs(info.min), info.max)
    if x.ndim == 2:
        x = x.mean(axis=1)
    duration = len(x) / sr
    peak = np.abs(x).max()
    rms = np.sqrt(np.mean(x ** 2))
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), 1 / sr)
    total = power.sum() + 1e-20
    centroid = np.sum(freq * power) / total
    bandwidth = np.sqrt(np.sum((freq - centroid) ** 2 * power) / total)
    rolloff = freq[min(np.searchsorted(np.cumsum(power), .85 * total), len(freq)-1)]
    bands = [power[(freq >= lo) & (freq < hi)].sum() / total
             for lo, hi in [(0, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, sr/2+1)]]
    # Local transient density, NOT true crack rate of the full roasting trial.
    peaks, _ = find_peaks(np.abs(x), height=max(rms * 3, 1e-9), distance=max(1, int(sr*.01)))
    gaps = np.diff(peaks) / sr
    return np.asarray([duration, peak, rms, np.mean(x[:-1]*x[1:] < 0) if len(x)>1 else 0,
                       centroid/sr, bandwidth/sr, rolloff/sr, *bands,
                       len(peaks)/duration, gaps.mean() if gaps.size else 0,
                       gaps.std() if gaps.size else 0], dtype=np.float64)
