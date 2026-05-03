import os
import time
import json
import gc

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from model import get_model, get_criterion
from config import get_class_names
from train import seed_torch


def _ensure_tensor_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _batch_to_list(value, batch_size, default=None):
    if value is None:
        return [default] * batch_size
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * batch_size


def collect_predictions(model, dataloader, device, class_names):
    model.eval()
    class_names = class_names or ['SR', 'AF', 'Other']

    all_subjects = []
    all_labels = []
    all_segs = []
    all_motion = []
    prob_cols = {cls: [] for cls in class_names}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collect predictions"):
            x = batch['signal'].to(device)
            outputs = _ensure_tensor_logits(model(x))
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            bs = probs.shape[0]
            all_subjects.extend(_batch_to_list(batch.get('subject_id'), bs))
            all_labels.extend(_batch_to_list(batch.get('label'), bs))
            all_segs.extend(_batch_to_list(batch.get('segment_index'), bs))
            all_motion.extend(_batch_to_list(batch.get('motion_score'), bs))

            for i, cls in enumerate(class_names):
                prob_cols[cls].extend(probs[:, i].tolist())

    out = {
        'subject_id': all_subjects,
        'label': all_labels,
        'segment_index': all_segs,
        'motion_variance': all_motion
    }
    for cls in class_names:
        out[f'prob_{cls}'] = prob_cols[cls]

    return pd.DataFrame(out)


def save_predictions_csv(model, dataloader, device, class_names, save_path):
    df = collect_predictions(model, dataloader, device, class_names)
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    df['pred_label'] = df[prob_cols].values.argmax(axis=1)
    df.to_csv(save_path, index=False)
    print(f"Saved prediction probabilities to {save_path}")


def evaluate_and_save_multiclass_roc_by_percentiles(
    model,
    dataloader,
    criterion,
    device,
    percentiles=None,
    class_names=None,
    save_dir='roc_by_percentile',
    filter_tag='all',
    higher_is_noisier=True,
    precomputed_thresholds=None
):
    """
    Cumulative evaluation: for each percentile p, collect samples that satisfy the motion score threshold
    and compute metrics.
    """
    class_names = class_names or get_class_names()
    percentiles = (percentiles or [10, 30, 50, 70, 90, 100])
    percentiles = sorted(percentiles)

    save_dir = os.path.join(save_dir, filter_tag)
    os.makedirs(save_dir, exist_ok=True)

    if precomputed_thresholds is None:
        all_motion_scores = []
        for batch in tqdm(dataloader, desc="Collect motion scores (eval)"):
            all_motion_scores.append(batch['motion_score'].cpu().numpy())
        all_motion_scores = np.concatenate(all_motion_scores)
        thresholds = np.percentile(all_motion_scores, percentiles)
    else:
        thresholds = np.asarray(precomputed_thresholds, dtype=float)

    if not higher_is_noisier:
        thresholds = thresholds[::-1]
        percentiles = percentiles[::-1]

    results = []

    for p, thr in zip(percentiles, thresholds):
        desc = f"<= {thr:.6f}" if higher_is_noisier else f">= {thr:.6f}"
        print(f"\nCUMULATIVE @ {p}th percentile (motion_score {desc})")

        all_preds, all_labels, all_probs = [], [], []
        total_loss, total_count = 0.0, 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Eval p{p}"):
                motion = batch['motion_score'].cpu().numpy()
                mask = (motion <= thr) if higher_is_noisier else (motion >= thr)
                if not mask.any():
                    continue

                x = batch['signal'][mask].to(device)
                y = batch['label'][mask].to(device)

                outputs = _ensure_tensor_logits(model(x))
                loss = criterion(outputs, y)
                probs = F.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)

                bs = y.shape[0]
                total_loss  += loss.item() * bs
                total_count += bs

                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        if total_count == 0:
            print("Warning: No valid samples for this cumulative threshold.")
            continue

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs  = np.concatenate(all_probs)
        avg_loss   = total_loss / total_count

        k = len(class_names)
        if k == 2:
            labels_arr = np.asarray(all_labels)
            binarized_labels = np.column_stack([1 - labels_arr, labels_arr])
        else:
            binarized_labels = label_binarize(all_labels, classes=list(range(k)))

        auc_scores = {}
        for i, cls_name in enumerate(class_names):
            y_true  = binarized_labels[:, i]
            y_score = all_probs[:, i]
            if y_true.max() == y_true.min():
                auc_scores[cls_name] = np.nan
            else:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_scores[cls_name] = auc(fpr, tpr)

        # macro/micro AUC
        try:
            if k == 2:
                auc_macro = roc_auc_score(all_labels, all_probs[:, 1])
                auc_micro = auc_macro
            else:
                auc_macro = roc_auc_score(binarized_labels, all_probs, average='macro', multi_class='ovr')
                auc_micro = roc_auc_score(binarized_labels, all_probs, average='micro', multi_class='ovr')
        except Exception:
            auc_macro, auc_micro = np.nan, np.nan

        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

        metrics = {
            'percentile': int(p),
            'motion_threshold': float(thr),
            'mode': 'cumulative',
            'direction': 'higher_is_noisier' if higher_is_noisier else 'higher_is_cleaner',
            'accuracy': float((all_preds == all_labels).mean()),
            'avg_loss': float(avg_loss),
            'sample_count': int(total_count),
            'confusion_matrix': str(confusion_matrix(all_labels, all_preds).tolist()),
            'auc_macro': float(auc_macro) if np.isfinite(auc_macro) else np.nan,
            'auc_micro': float(auc_micro) if np.isfinite(auc_micro) else np.nan,
            **{f'auc_{cls}': (float(v) if v==v else np.nan) for cls, v in auc_scores.items()},
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro':    recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_macro':        f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall_weighted':    recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1_weighted':        f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        }
        for cls in class_names:
            metrics[f'{cls}_precision'] = report[cls]['precision']
            metrics[f'{cls}_recall']    = report[cls]['recall']
            metrics[f'{cls}_f1']        = report[cls]['f1-score']
            metrics[f'{cls}_support']   = int(report[cls]['support'])

        df_metrics = pd.DataFrame([metrics])
        csv_path = os.path.join(save_dir, f'metrics_p{p}.csv')
        df_metrics.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} | N={int(total_count)} | "
              f"supports={{" + ", ".join([f"{c}:{int(report[c]['support'])}" for c in class_names]) + "}}")

        results.append(metrics)

    df_summary = pd.DataFrame(results)
    summary_path = os.path.join(save_dir, 'summary_by_percentile_cumulative.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    return df_summary

def evaluate_all_seeds_by_percentile_thresholds(params, test_loader):
    """
    Evaluate all seeds with cumulative percentile thresholds and aggregate results.
    """
    model_name     = params['model_name']
    criterion_name = params['criterion_name']
    device        = params['device']
    use_channel   = params.get("use_channel", "ppg_acc")
    class_names   = get_class_names()
    seeds         = params.get('seeds', [params.get('random_seed', 42)])
    criterion     = get_criterion(criterion_name)
    percentiles   = sorted(params.get("percentiles", [10, 30, 50, 70, 90, 100]))
    save_predictions = params.get("save_predictions_csv", True)
    predictions_filename = params.get("predictions_filename", "predictions_full_test.csv")
    save_ensemble = params.get("save_predictions_ensemble_csv", True)
    ensemble_filename = params.get("predictions_ensemble_filename", "predictions_full_test_ensemble.csv")

    higher_is_noisier = params.get("higher_is_noisier", True)
    run_name = params.get("run_name", None)
    run_suffix = f"_{run_name}" if run_name else ""

    filter_tag = use_channel
    summary_dir = os.path.join(
        "results_0524normalsegment",
        f"percentile_eval_{model_name}_{criterion_name}_{use_channel}_{filter_tag}{run_suffix}"
    )
    os.makedirs(summary_dir, exist_ok=True)

    all_motion = []
    for batch in test_loader:
        all_motion.append(batch['motion_score'].cpu().numpy())
    all_motion = np.concatenate(all_motion)

    pre_thr = np.percentile(all_motion, percentiles)

    all_seed_results = []
    all_seed_prob_dfs = []

    for seed in seeds:
        print(f"\nEvaluating percentile-based ROC for seed {seed}")
        seed_torch(seed)

        model_file_dir = os.path.join(
            "model_save_0524normalsegment",
            f"model_{model_name}_{criterion_name}_{use_channel}{run_suffix}",
            f"training_metrics_seed_{seed}"
        )
        model_path = os.path.join(model_file_dir, "final_best_model.pth")
        assert os.path.exists(model_path), f"Error: Model not found: {model_path}"

        model = get_model(model_name, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        seed_output_dir = os.path.join(summary_dir, f"seed_{seed}")
        os.makedirs(seed_output_dir, exist_ok=True)

        df = evaluate_and_save_multiclass_roc_by_percentiles(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            percentiles=percentiles,
            class_names=class_names,
            save_dir=seed_output_dir,
            filter_tag=filter_tag,
            higher_is_noisier=higher_is_noisier,
            precomputed_thresholds=pre_thr
        )

        all_seed_results.append(df)

        if save_predictions:
            pred_csv_path = os.path.join(seed_output_dir, predictions_filename)
            save_predictions_csv(
                model=model,
                dataloader=test_loader,
                device=device,
                class_names=class_names,
                save_path=pred_csv_path
            )
            if save_ensemble:
                pred_df = collect_predictions(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    class_names=class_names
                )
                all_seed_prob_dfs.append(pred_df)

        print(f"Done: seed {seed}")
        del model
        torch.cuda.empty_cache()
        gc.collect()

    combined_df = pd.concat(all_seed_results)
    metrics_cols = [col for col in combined_df.columns
                    if col not in ['percentile', 'motion_threshold', 'confusion_matrix', 'sample_count',
                                   'mode', 'direction']]

    summary_list = []
    for p in percentiles:
        df_p = combined_df[combined_df['percentile'] == p]
        motion_thr_mean = df_p['motion_threshold'].mean() if 'motion_threshold' in df_p else np.nan
        summary_dict = {'percentile': p, 'motion_threshold_mean': motion_thr_mean}
        for col in metrics_cols:
            summary_dict[f'{col}_mean'] = df_p[col].mean()
            summary_dict[f'{col}_std']  = df_p[col].std()
        summary_list.append(summary_dict)

    summary_df = pd.DataFrame(summary_list)
    summary_path = os.path.join(summary_dir, 'summary_mean_std_by_percentile_cumulative.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Average and std summary saved to {summary_path}")

    if save_ensemble and all_seed_prob_dfs:
        base_df = all_seed_prob_dfs[0].copy()
        prob_cols = [c for c in base_df.columns if c.startswith('prob_')]
        prob_stack = [df[prob_cols].values for df in all_seed_prob_dfs]
        prob_mean = np.mean(np.stack(prob_stack, axis=0), axis=0)
        base_df[prob_cols] = prob_mean
        base_df['pred_label'] = prob_mean.argmax(axis=1)
        ensemble_path = os.path.join(summary_dir, ensemble_filename)
        base_df.to_csv(ensemble_path, index=False)
        print(f"Saved ensemble predictions to {ensemble_path}")

    return summary_df
