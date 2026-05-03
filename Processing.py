import numpy as np
import scipy.signal as signal  
from config import params

import numpy as np
from scipy import signal

class BandpassFilter(object):
    def __init__(self, fs, n=4, f_cut_ppg=[0.5, 8], f_cut_acc=[0.3, 10], use_channel='ppg_acc'):
        self.fs = fs
        self.n = n
        self.f_cut_ppg = f_cut_ppg
        self.f_cut_acc = f_cut_acc
        self.use_channel = str(use_channel).lower()

    def __call__(self, sample):
        s = sample['signal']              # shape: (C, L)
        s_filtered = np.copy(s)

        # Band-pass filter PPG channel (index 0)
        if self.use_channel in ['ppg', 'ppg_acc', '4ch'] and s.shape[0] >= 1:
            sos_ppg = signal.butter(self.n, self.f_cut_ppg, btype="bandpass", output="sos", fs=self.fs)
            s_filtered[0, :] = signal.sosfiltfilt(sos_ppg, s[0, :])

        # Band-pass filter ACC channels only for acc/ppg_acc; leave other channels untouched in 4ch
        if self.use_channel in ['acc', 'ppg_acc']:
            sos_acc = signal.butter(self.n, self.f_cut_acc, btype="bandpass", output="sos", fs=self.fs)
            acc_start_idx = 1 if self.use_channel == 'ppg_acc' else 0
            # If dual-channel autocorr existed at index 1, ACC would start at 2 (kept for compatibility)
            if self.use_channel == 'ppg_acc' and s.shape[0] >= 3:
                acc_start_idx = 2
            for i in range(acc_start_idx, s.shape[0]):
                s_filtered[i, :] = signal.sosfiltfilt(sos_acc, s[i, :])

        return {'signal': s_filtered, 'label': sample['label'], 'motion_score': sample['motion_score']}


class Normalize(object):
    def __init__(self, use_channel='ppg_acc', ppg_only=False):
        self.use_channel = str(use_channel).lower()
        self.ppg_only = ppg_only

    def __call__(self, sample):
        s = sample['signal']
        norm_s = np.array(s, dtype=np.float32, copy=True)
        if self.ppg_only or self.use_channel == '4ch':
            mu = np.mean(norm_s[0]); sd = np.std(norm_s[0]) + 1e-8
            norm_s[0] = (norm_s[0] - mu) / sd
        else:
            for i in range(norm_s.shape[0]):
                mu = np.mean(norm_s[i]); sd = np.std(norm_s[i]) + 1e-8
                norm_s[i] = (norm_s[i] - mu) / sd
        return {'signal': norm_s, 'label': sample['label'], 'motion_score': sample['motion_score']}


class AddNoise(object):
    def __init__(self, noise_level, ppg_only=False, use_channel='ppg_acc'):
        self.noise_level = noise_level
        self.ppg_only = ppg_only
        self.use_channel = str(use_channel).lower()

    def __call__(self, sample):
        s = sample['signal'].astype(np.float32, copy=True)
        if self.ppg_only or self.use_channel == '4ch':
            s[0] += np.random.randn(*s[0].shape).astype(np.float32) * self.noise_level
        else:
            s += np.random.randn(*s.shape).astype(np.float32) * self.noise_level
        return {'signal': s, 'label': sample['label'], 'motion_score': sample['motion_score']}


class TimeShift(object):
    def __init__(self, shift_range, ppg_only=False, use_channel='ppg_acc'):
        self.shift_range = shift_range
        self.ppg_only = ppg_only
        self.use_channel = str(use_channel).lower()

    def __call__(self, sample):
        s = sample['signal']
        shift = int(np.random.randint(-int(s.shape[-1]*self.shift_range), int(s.shape[-1]*self.shift_range)))
        s_shifted = np.array(s, copy=True)
        if self.ppg_only or self.use_channel == '4ch':
            s_shifted[0] = np.roll(s[0], shift)
        else:
            s_shifted = np.roll(s, shift, axis=-1)
        return {'signal': s_shifted, 'label': sample['label'], 'motion_score': sample['motion_score']}


class AmplitudeScaling(object):
    def __init__(self, scale_range, ppg_only=False, use_channel='ppg_acc'):
        self.scale_range = scale_range
        self.ppg_only = ppg_only
        self.use_channel = str(use_channel).lower()

    def __call__(self, sample):
        s = sample['signal'].astype(np.float32, copy=True)
        scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        if self.ppg_only or self.use_channel == '4ch':
            s[0] *= scale
        else:
            s *= scale
        return {'signal': s, 'label': sample['label'], 'motion_score': sample['motion_score']}


class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, sample):
        for t in self.transforms: sample = t(sample)
        return sample


base_transform = Compose([
    BandpassFilter(fs=params['fs'], use_channel=params.get('use_channel', 'ppg_acc')),
    Normalize()
])


augmentations = [
    AddNoise(noise_level=params['noise_level']),
    TimeShift(shift_range=params['shift_range']),
    AmplitudeScaling(scale_range=params['scale_range'])
]
