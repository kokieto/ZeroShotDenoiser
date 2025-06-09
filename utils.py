from config import cfg
from tqdm import tqdm
import numpy as np
from scipy.io.wavfile import write
import scipy.signal as ss
from ssspy.bss.ilrma import GaussILRMA
from matplotlib import pyplot as plt


def write_as_wav(amples, file_nm_idx):
    amples /= np.max(np.abs(amples))
    amples *= (32767 - 10)
    amples = amples.astype(np.int16)
    write(f'{file_nm_idx}.wav', cfg.sr, amples)


def sound_separation(comb_amples, n_sources, n_basis, n_iter, n_fft):
    if n_sources == 2:
        comb_amples = comb_amples[[0, 3], :].copy()
    elif n_sources == 3:
        comb_amples = comb_amples[[0, 2, 3], :].copy()
    elif n_sources >= 5:
        raise ValueError('n_sources must be either 2, 3 or 4')
    model = GaussILRMA(n_basis=n_basis, scale_restoration=True, partitioning=True)
    _, _, spectrogram_mix = ss.stft(comb_amples, nfft=n_fft, nperseg=n_fft, noverlap=n_fft*cfg.overlap_ratio)
    spectrogram_est = model(spectrogram_mix, n_iter=n_iter)
    _, separated_amples = ss.istft(spectrogram_est, nfft=n_fft, nperseg=n_fft, noverlap=n_fft*cfg.overlap_ratio)
    return separated_amples


def vizualize_spectrogram(
        amples,
        init_time = 0,
        end_time = 2,
        cut_percent = 1,
        fig_size = [6, 6],
        font_size = 28,
        max_freq = 10,
    ):
    plt.figure(figsize=fig_size)
    plt.rcParams['font.size'] = font_size
    freqs, times, spec = ss.stft(amples[cfg.sr*init_time:cfg.sr*end_time], nfft=512, nperseg=512, noverlap=256, scaling='psd', fs=cfg.sr)
    spec, freqs = 10*np.log10(np.abs(spec)), freqs/1000
    idx_freq = np.searchsorted(freqs, max_freq)
    vmin, vmax = np.percentile(spec[:idx_freq, :], [cut_percent, 100-cut_percent], axis=[0, 1])
    plt.imshow(
        spec, 
        aspect='auto', 
        origin='lower', 
        cmap='jet', 
        vmin=vmin, 
        vmax=vmax,
        extent=[times[0], times[-1], freqs[0], freqs[idx_freq]]
    )
    plt.ylim(0, max_freq)
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Time [s]')
    plt.show()


def is_2d_array(arr):
    return isinstance(arr, np.ndarray) and arr.ndim == 2