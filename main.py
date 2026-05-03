import os
import re
import torch
from torch.utils.data import DataLoader
from config import params
from Processing import Compose, BandpassFilter, Normalize
from dataset import PPGDataset, load_data
from model import get_model, get_optimizer, get_scheduler, get_criterion
from train import train_and_evaluate, finalize_model_training, seed_torch
from evaluation import evaluate_all_seeds_by_percentile_thresholds

import builtins
import traceback

# Protect original data directories from accidental writes
original_open = open  # keep original open reference

def read_only_guard(file, mode='r', *args, **kwargs):
    file = os.path.abspath(file)
    protected_dirs = [os.path.abspath('../04_Yang_Dataset_all')]
    if any(file.startswith(p) for p in protected_dirs) and any(m in mode for m in ('w', 'a', 'x')):
        print(f"\n [Blocked Write Attempt] File: {file}, Mode: {mode}")
        print(" Stack trace of offending call:")
        traceback.print_stack(limit=5)
        raise PermissionError(f" Attempt to write to protected file: {file} (mode: {mode})")
    return original_open(file, mode, *args, **kwargs)

builtins.open = read_only_guard

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    params['device'] = device

    root_dir = '../04_Yang_Dataset_all/1_M4M'
    folders = ['test_AF', 'test_SR', 'train_AF', 'train_SR']

    use_channel = str(params.get('use_channel', 'ppg_acc')).lower()

    # Load data (4ch merges sidecar internally)
    grouped_data = load_data(
        main_folder=root_dir,
        folders=folders,
        motion_variance_csv_path=params['motion_variance_csv_path'],
        use_channel=params['use_channel']
        # sidecar_subdir / base_subdir use default:ppg_sidecar_4cols / updated_PPG
    )

    def get_subject_ids_from_folder(folder_path):
        if not os.path.isdir(folder_path):
            return []
        return list({
            re.findall(r'\d+', fname)[0]
            for fname in os.listdir(folder_path) if fname.lower().endswith(('.csv', '.txt'))
            and re.findall(r'\d+', fname)
        })

    test_AF_subjects = get_subject_ids_from_folder(os.path.join(root_dir, 'test_AF', 'updated_PPG'))
    test_SR_subjects = get_subject_ids_from_folder(os.path.join(root_dir, 'test_SR', 'updated_PPG'))
    test_ALL_subjects = sorted(list(set(test_AF_subjects + test_SR_subjects)))

    base_transform = Compose([
        BandpassFilter(fs=params['fs'], use_channel=use_channel),
        Normalize()
    ])

    test_ALL_dataset = PPGDataset(
        data_dict=grouped_data,
        subjects=test_ALL_subjects,
        each_slice_length=params['each_slice_length'],
        fs=params['fs'],
        base_transform=base_transform,
        augmentations=None,
        use_channel=use_channel
    )
    test_ALL_loader = DataLoader(test_ALL_dataset, batch_size=params['batch_size'], shuffle=False)

    print(f"Test AF subjects: {len(test_AF_subjects)}")
    print(f"Test SR subjects: {len(test_SR_subjects)}")
    print(f"Test ALL subjects: {len(test_ALL_subjects)}")

    train_grouped_data = {sid: data for sid, data in grouped_data.items() if sid not in test_ALL_subjects}

    finalize_model_training(
        grouped_data=train_grouped_data,
        all_subjects=list(train_grouped_data.keys()),
        params=params
    )

    evaluate_all_seeds_by_percentile_thresholds(params, test_ALL_loader)

if __name__ == "__main__":
    main()
