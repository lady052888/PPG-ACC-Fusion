import copy
import traceback

from main import main, params

# Fixed training config
NUM_EPOCHS = 60

# Experiment list
experiments = []

base_cfg = copy.deepcopy(params)
base_cfg.update({
    'num_epochs': NUM_EPOCHS,
    'criterion_name': 'CrossEntropyLoss',
    'class_weights': None,
    'dual_channel': False
})

# EXP I: compare input (same model, different channels)
for use_channel, input_tag in [
    ('ppg', 'PPG'),
    ('4ch', '4CH'),
    ('ppg_acc', 'PPG_ACC'),
]:
    model_name = 'ResNet10_TemporalAttention_DilatedL2'
    experiments.append({
        **base_cfg,
        'model_name': model_name,
        'use_channel': use_channel,
        'input_tag': input_tag,
        'apply_autocorrelation': False,
        'run_name': f'{model_name}__{use_channel}__3class__CE'
    })

# EXP II: compare model (ppg+acc)
for model_name in [
    'Han25-BiGRU',
    'Bulut25-CNN17',
    'Shen19-50CNN',
    'mobile_net',
    'ResNet10_TemporalAttention_DilatedL2',
    'Zhao25-RhythmiNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'Liu22-DCNN',
]:
    experiments.append({
        **base_cfg,
        'model_name': model_name,
        'use_channel': 'ppg_acc',
        'input_tag': 'PPG_ACC',
        'apply_autocorrelation': False,
        'run_name': f'{model_name}__ppg_acc__3class__CE'
    })

# EXP III: reproduce (mixed channels)
for model_name, use_channel, input_tag in [
    ('Bulut25-CNN17', 'ppg', 'PPG'),
    ('Han25-BiGRU', '4ch', '4CH'),
    ('Shen19-50CNN', 'ppg', 'PPG'),
    ('ResNet10_TemporalAttention_DilatedL2', 'ppg_acc', 'PPG_ACC'),
    ('Liu22-DCNN', 'ppg', 'PPG'),
    ('Zhao25-RhythmiNet', 'ppg_acc', 'PPG_ACC'),
]:
    experiments.append({
        **base_cfg,
        'model_name': model_name,
        'use_channel': use_channel,
        'input_tag': input_tag,
        'apply_autocorrelation': False,
        'run_name': f'{model_name}__{use_channel}__3class__CE'
    })

print(f'Total experiments: {len(experiments)}')
for i, cfg in enumerate(experiments, 1):
    print(f"#{i:02d} | model={cfg['model_name']:<32} "
          f"use_channel={cfg['use_channel']:<8} "
          f"criterion={cfg['criterion_name']}")
    print(f"run_name={cfg['run_name']}")

# Run experiments
orig_params = copy.deepcopy(params)

for i, cfg in enumerate(experiments, 1):
    print("\n" + "=" * 90)
    print(f"Running {i}/{len(experiments)} | "
          f"model={cfg['model_name']} | "
          f"use_channel={cfg['use_channel']}")
    print(f"run_name={cfg['run_name']}")
    print("=" * 90)
    try:
        params.update(copy.deepcopy(cfg))
        main()
    except Exception as e:
        print(f"Error: Experiment {i} failed: {e}")
        traceback.print_exc()
    finally:
        params.clear()
        params.update(copy.deepcopy(orig_params))
