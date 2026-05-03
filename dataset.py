import os
import re
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from config import params, get_class_names
from Processing import (
    AddNoise,
    AmplitudeScaling,
    BandpassFilter,
    Compose,
    Normalize,
    TimeShift
)

def _read_csv_fallback(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        for enc in ('utf-8-sig', 'gbk', 'latin1'):
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise

# Base transform
def build_base_transform(fs, use_channel):
    return Compose([
        BandpassFilter(fs=fs, use_channel=use_channel),
        Normalize()
    ])

# Data augmentation
def build_augmentations(noise_level, shift_range, scale_range, enable=True):
    if not enable:
        return None
    return [
        AddNoise(noise_level=noise_level),
        TimeShift(shift_range=shift_range),
        AmplitudeScaling(scale_range=scale_range),
]


# Segmenter with zero-padding (kept for compatibility)
def segmenter(x, n, p=0):
    x = np.asarray(x)
    if x.ndim == 1:
        C = 1
        L = len(x)
        x = x[np.newaxis, :]
    else:
        C, L = x.shape
    n, p = int(n), int(p)
    s = n - p
    m = max(1, int(np.ceil(L / s)))
    data = []
    for i in range(m):
        start = s * i
        stop = start + n
        segment = np.zeros((C, n))
        for ch in range(C):
            signal_ch = x[ch]
            if stop > L:
                pad = np.zeros(stop - L)
                segment[ch, :] = np.concatenate([signal_ch[start:], pad])
            else:
                segment[ch, :] = signal_ch[start:stop]
        data.append(segment)
    return np.array(data)  # (num_segments, C, n)


# Segmenter that drops the tail to align strictly with motion windows
def segmenter_drop_tail(x, n):
    """
    Keep only full windows, no padding. Aligns with motion score segmentation.
    x: (C, L) or (L,)
    n: segment length (e.g., 30s*fs)
    return: (num_segments, C, n)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    C, L = x.shape
    m = L // n
    if m <= 0:
        return np.empty((0, C, n), dtype=x.dtype)
    x_cut = x[:, :m * n]
    return x_cut.reshape(C, m, n).transpose(1, 0, 2)


def get_segment_rhythm_labels(type3_array, each_slice_length, fs, class_map):
    segment_labels = []
    segs = segmenter_drop_tail(type3_array, each_slice_length * fs)
    for seg in segs:
        unique, counts = np.unique(seg, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        segment_labels.append(class_map.get(majority_label))
    return segment_labels


class PPGDataset(Dataset):
    """
    Supported channel settings:
      - 'ppg'       -> PPG only
      - 'acc'       -> ACC x/y/z
      - 'ppg_acc'   -> PPG + ACC x/y/z
      - '4ch' / 'ppg_hr_accmag_maghr' -> sidecar four channels: [PPG, HR_line_32Hz, acc_mag, magHR_32Hz]
        * In 4ch mode, base_transform is applied only to PPG; no autocorrelation is used.
    """
    def __init__(self,
                 data_dict,
                 subjects,
                 each_slice_length,
                 fs,
                 base_transform=None,
                 augmentations=None,
                 use_channel='ppg_acc',
                 hr_min=30.0, hr_max=220.0,
                 enable_base_transform_in_4ch=True,
                 enable_augment_in_4ch=False,
                 remap_maghr_segmentwise=False):

        self.base_transform = base_transform
        self.augmentations = augmentations
        self.each_slice_length = each_slice_length
        self.fs = fs
        self.use_channel = str(use_channel).lower()
        self.samples = []

        self.hr_min = float(hr_min)
        self.hr_max = float(hr_max)
        self.enable_base_transform_in_4ch = bool(enable_base_transform_in_4ch)
        self.enable_augment_in_4ch = bool(enable_augment_in_4ch)
        self.remap_maghr_segmentwise = bool(remap_maghr_segmentwise)

        class_names = get_class_names()
        class_map = {name: idx for idx, name in enumerate(class_names)}

        SEG_LEN = self.each_slice_length * self.fs

        for subject in subjects:
            entry = data_dict[subject]
            df = entry['data']
            motion_variances = entry.get('motion_variances', None)

            # Select channels
            if self.use_channel == 'ppg':
                signal_array = df[['ppg']].values.T.astype(np.float32)

            elif self.use_channel == 'acc':
                signal_array = df[['acc_x', 'acc_y', 'acc_z']].values.T.astype(np.float32)

            elif self.use_channel == 'ppg_acc':
                signal_array = df[['ppg', 'acc_x', 'acc_y', 'acc_z']].values.T.astype(np.float32)

            elif self.use_channel in ['4ch', 'ppg_hr_accmag_maghr']:
                needed = ['ppg', 'HR_line_32Hz', 'acc_mag', 'magHR_32Hz']
                miss = [c for c in needed if c not in df.columns]
                if miss:
                    raise ValueError(f"4ch requires columns {needed}, missing: {miss}")
                signal_array = df[needed].values.T.astype(np.float32)

            else:
                raise ValueError(f"Unknown use_channel: {self.use_channel}")

            # Segment (aligned with motion scores)
            segments = segmenter_drop_tail(signal_array, SEG_LEN)

            # Segment labels (majority vote)
            raw_type3 = df['Type3'].values
            type3_labels = get_segment_rhythm_labels(
                raw_type3,
                self.each_slice_length,
                self.fs,
                class_map
            )

            # Motion alignment
            if motion_variances is None:
                motion_variances = np.zeros(len(segments), dtype=float)
            else:
                if len(motion_variances) != len(segments):
                    if len(motion_variances) + 1 == len(segments):
                        segments = segments[:len(motion_variances)]
                    else:
                        print(f"Warning: Segment-mismatch for subject {subject}: "
                              f"{len(segments)} vs {len(motion_variances)} motion scores")
                    motion_variances = motion_variances[:len(segments)]

            valid_len = min(len(segments), len(type3_labels), len(motion_variances))

            # Build samples per segment
            for idx in range(valid_len):
                seg = segments[idx].astype(np.float32)  # (C, n)
                label = type3_labels[idx]
                if label is None:
                    continue
                motion_score = float(motion_variances[idx])

                # 4ch branch: transform only PPG
                if self.use_channel in ['4ch', 'ppg_hr_accmag_maghr']:
                    sample = {'signal': seg, 'label': label, 'motion_score': motion_score}
                    if self.enable_base_transform_in_4ch and self.base_transform:
                        sample = self.base_transform(sample)
                    sample['subject_id'] = subject
                    sample['segment_index'] = idx
                    self.samples.append(sample)

                    if self.enable_augment_in_4ch and self.augmentations:
                        for aug in self.augmentations:
                            aug_sample = aug(sample.copy())
                            aug_sample['subject_id'] = subject
                            aug_sample['segment_index'] = idx
                            self.samples.append(aug_sample)
                    continue

                # Non-4ch branch
                sample = {'signal': seg, 'label': label, 'motion_score': motion_score}
                if self.base_transform:
                    sample = self.base_transform(sample)
                sample['subject_id'] = subject
                sample['segment_index'] = idx

                self.samples.append(sample)

                if self.augmentations:
                    for augmentation in self.augmentations:
                        aug_sample = augmentation(sample.copy())
                        aug_sample['subject_id'] = subject
                        aug_sample['segment_index'] = idx
                        self.samples.append(aug_sample)

        # Stats
        counts = Counter([s['label'] for s in self.samples])
        print("\nSegment class distribution in PPGDataset:")
        for i, name in enumerate(class_names):
            print(f"  {name} ({i}): {counts.get(i, 0)}")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = sample['signal']; y = sample['label']; m = sample['motion_score']
        return {
            'signal': torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x,
            'label': torch.tensor(y, dtype=torch.long) if not isinstance(y, torch.Tensor) else y,
            'motion_score': torch.tensor(m, dtype=torch.float32),
            'subject_id': sample.get('subject_id', None),
            'segment_index': sample.get('segment_index', None)
        }

    def __len__(self):
        return len(self.samples)


def extract_subject_id(filename):
    """Extract the first numeric substring as subject_id"""
    m = re.findall(r'\d+', filename)
    return m[0] if m else None


def load_data(main_folder,
              folders,
              motion_variance_csv_path=None,
              use_channel='ppg_acc',
              sidecar_subdir='ppg_sidecar_4cols',
              base_subdir='updated_PPG'):
    """
    4ch: read [ppg_timestamp, ppg, Type3] from base_subdir and merge sidecar [acc_mag, HR_line_32Hz, magHR_32Hz].
    Non-4ch: read updated_PPG (ppg/acc_x/acc_y/acc_z/Type3).
    Motion score: grouped by source_file, aggregated by subject id.
    """
    data_dict = {}
    motion_variance_dict = {}

    # Load motion scores
    if motion_variance_csv_path is not None:
        mv = _read_csv_fallback(motion_variance_csv_path, low_memory=False)
        for filename, group in mv.groupby('source_file'):
            sid = extract_subject_id(filename)
            if sid:
                motion_variance_dict[sid] = group.sort_values('segment_in_file')['motion_variance'].values
    print(f"Loaded motion variances for {len(motion_variance_dict)} subjects")

    ucl = str(use_channel).lower()
    for folder in folders:
        base_dir = os.path.join(main_folder, folder, base_subdir)
        sidecar_dir = os.path.join(main_folder, folder, sidecar_subdir)

        # 4ch: merge with sidecar
        if ucl in ['4ch', 'ppg_hr_accmag_maghr']:
            if not (os.path.exists(base_dir) and os.path.exists(sidecar_dir)):
                raise FileNotFoundError(f"4ch mode requires sidecar directory: {sidecar_dir}")
            for filename in sorted(os.listdir(base_dir)):
                if not filename.lower().endswith(('.csv', '.txt')):
                    continue
                base_path = os.path.join(base_dir, filename)
                side_path = os.path.join(sidecar_dir, filename)
                if not os.path.exists(side_path):
                    continue
                try:
                    df_base = _read_csv_fallback(
                        base_path,
                        usecols=['ppg_timestamp', 'ppg', 'Type3'],
                        low_memory=False
                    )
                    df_side = _read_csv_fallback(
                        side_path,
                        low_memory=False
                    )  # ppg_timestamp, acc_mag, HR_line_32Hz, magHR_32Hz

                    # Align timestamp dtype to avoid merge issues
                    if df_base['ppg_timestamp'].dtype != df_side['ppg_timestamp'].dtype:
                        df_base['ppg_timestamp'] = df_base['ppg_timestamp'].astype(str)
                        df_side['ppg_timestamp'] = df_side['ppg_timestamp'].astype(str)

                    df = pd.merge(df_base, df_side, on='ppg_timestamp', how='inner', sort=False)
                    df = df[['ppg', 'HR_line_32Hz', 'acc_mag', 'magHR_32Hz', 'Type3']]

                    sid = extract_subject_id(filename)
                    if sid is None:
                        continue
                    data_dict[sid] = {
                        'data': df,
                        'motion_variances': motion_variance_dict.get(sid),
                        'motion_start_idx': 0
                    }
                except Exception as e:
                    print(f"Error: Failed to merge {filename}: {e}")
            continue

        # Non-4ch: read updated_PPG only
        if not os.path.exists(base_dir):
            continue

        for filename in sorted(os.listdir(base_dir)):
            if not filename.lower().endswith(('.csv', '.txt')):
                continue
            file_path = os.path.join(base_dir, filename)

            req = []
            if ucl in ['ppg', 'ppg_acc']:
                req += ['ppg']
            if ucl in ['acc', 'ppg_acc']:
                req += ['acc_x', 'acc_y', 'acc_z']
            req += ['Type3']

            try:
                df = _read_csv_fallback(file_path, usecols=req, low_memory=False)
                df = df.astype(float, errors='ignore')  # keep Type3 as string
                sid = extract_subject_id(filename)
                if sid is None:
                    continue
                data_dict[sid] = {
                    'data': df,
                    'motion_variances': motion_variance_dict.get(sid),
                    'motion_start_idx': 0
                }
            except Exception as e:
                print(f"Error: Failed to load {file_path}: {e}")
                continue

    print(f"Loaded {len(data_dict)} subjects from folders: {folders}")

    # Optional: make columns read-only
    for sid, entry in data_dict.items():
        df = entry['data']
        for col in df.columns:
            try:
                v = df[col].values
                v.setflags(write=False)
            except Exception:
                pass

    return data_dict


def prepare_data_loaders(
    grouped_data,
    params,
    use_augmentation=True,
    use_channel=None
):
    each_slice_length = params.get('each_slice_length', 30)
    fs = params.get('fs', 32)
    batch_size = params.get('batch_size', 64)
    random_seed = params.get('random_seed', 42)
    use_channel = params.get('use_channel', 'ppg_acc') if use_channel is None else use_channel

    base_transform = build_base_transform(fs=fs, use_channel=use_channel)
    augmentations = build_augmentations(
        noise_level=params.get('noise_level', 0.01),
        shift_range=params.get('shift_range', 0.1),
        scale_range=params.get('scale_range', 0.1),
        enable=use_augmentation
    )

    # Stratify by majority label per subject
    all_subjects = list(grouped_data.keys())
    random.seed(random_seed)
    subject_major_labels = []
    class_names = get_class_names()
    class_map = {name: idx for idx, name in enumerate(class_names)}
    for subj in all_subjects:
        entry = grouped_data[subj]
        type3_labels = entry['data']['Type3'].values
        mapped = [class_map.get(label) for label in type3_labels if class_map.get(label) is not None]
        if len(mapped) == 0:
            continue
        major_label = np.bincount(mapped).argmax()
        subject_major_labels.append((subj, major_label))

    label_counter = Counter(label for _, label in subject_major_labels)
    print(f"Subject major label distribution: {dict(label_counter)}")

    valid_labels = {label for label, count in label_counter.items() if count >= 2}
    subject_major_labels = [(subj, label) for subj, label in subject_major_labels if label in valid_labels]
    if len(subject_major_labels) == 0:
        raise ValueError("Error: No valid subjects after filtering small classes.")

    subjects, labels = zip(*subject_major_labels)

    try:
        train_subjects, val_subjects = train_test_split(
            subjects, test_size=0.2, stratify=labels, random_state=random_seed
        )
    except ValueError as e:
        print(f"Warning: Stratify failed ({e}), fallback to random split.")
        subjects = list(subjects)
        random.shuffle(subjects)
        split_idx = int(0.8 * len(subjects))
        train_subjects = subjects[:split_idx]
        val_subjects = subjects[split_idx:]

    return create_dataloaders(
        grouped_data, train_subjects, val_subjects,
        each_slice_length, fs, batch_size,
        base_transform, augmentations,
        use_channel=use_channel
    )


def create_dataloaders(
    grouped_data,
    train_subjects,
    val_subjects,
    each_slice_length,
    fs,
    batch_size,
    base_transform,
    augmentations,
    use_channel='ppg_acc'
):
    train_dataset = PPGDataset(
        data_dict=grouped_data,
        subjects=train_subjects,
        each_slice_length=each_slice_length,
        fs=fs,
        base_transform=base_transform,
        augmentations=augmentations,
        use_channel=use_channel
    )

    val_dataset = PPGDataset(
        data_dict=grouped_data,
        subjects=val_subjects,
        each_slice_length=each_slice_length,
        fs=fs,
        base_transform=base_transform,
        augmentations=None,
        use_channel=use_channel
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
