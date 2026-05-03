import torch

params = {
# Classification Mode
    # '3class' keeps SR/AF/Other; '2class' drops one class (drop_class).
    'classification_mode': '3class',
    'drop_class': None,  # 'SR' / 'AF' / 'Other' when classification_mode == '2class'

# Model & Training Core
    'model_name': 'ResNet10_TemporalAttention_DilatedL2',
    # model_names = ['Shen19-50CNN','resnet18','resnet34','resnet50','resnet101','Liu22-DCNN','mobile_net','Bulut25-CNN17']
    'optimizer_name': 'Adam',
    'scheduler_name': 'StepLR',
    'criterion_name': 'CrossEntropyLoss',
    'learning_rate': 0.0001,
    'lrf': 0.0001,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'step_size': 20,
    'gamma': 0.1,
    'patience': 10,
    'num_epochs': 60,
    'batch_size': 64,
    'random_seed': 42,

# Signal Settings
    'fs': 32,
    'each_slice_length': 30,
    'label_column': 'Type3',
    'use_channel': 'ppg',  # 'ppg', 'acc', or 'ppg_acc'

# Data Augmentation
    'use_augmentation': False,
    'noise_level': 0.01,
    'shift_range': 0.1,
    'scale_range': 0.1,

# Motion Score Filtering
    'percentiles': [10, 20, 30, 40,50,60, 70, 80,90, 100],  # Used for evaluation
    'motion_variance_csv_path': '../15_clean_3class/motion_variances_by_segmentfirst_and_no_normal_no_a_nor.csv',

# Device & Seed
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
    'multi_seed': True,
    'seeds': [7, 1234, 9876],

# Prediction CSV Export
    'save_predictions_csv': True,
    'predictions_filename': 'predictions_full_test.csv',
    'save_predictions_ensemble_csv': True,
    'predictions_ensemble_filename': 'predictions_full_test_ensemble.csv',
 
}


def get_class_names():
    mode = params.get('classification_mode', '3class')
    base = ['SR', 'AF', 'Other']
    if mode == '2class':
        drop = params.get('drop_class', None)
        if drop not in base:
            raise ValueError(f"Invalid drop_class for 2class: {drop}")
        return [c for c in base if c != drop]
    return params.get('class_names', base)


def get_num_classes():
    return len(get_class_names())
