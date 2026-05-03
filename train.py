import os
import random
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure matplotlib works in headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.preprocessing import label_binarize
from collections import Counter

# Project modules
from Processing import (
    Compose, BandpassFilter, Normalize,
    AddNoise, TimeShift, AmplitudeScaling
)
from dataset import PPGDataset, prepare_data_loaders
from config import params, get_class_names
from model import get_model, get_optimizer, get_scheduler, get_criterion


def _ensure_tensor_logits(output):
    """
    Some models (e.g., BiGRU_Multimodal) return (logits, aux_outputs).
    Training/eval code expects a tensor, so unwrap tuple/list outputs here.
    """
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


# Seeding
def seed_torch(seed=7):
    """Fix all random seeds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'Random seed fixed to {seed}.')


# Main training wrapper
def train_and_evaluate(model, optimizer, scheduler, criterion, grouped_data, params):
    use_channel = params.get("use_channel", "ppg_acc").lower()
    class_names = get_class_names()
    output_dir = f'training_metrics_{params["model_name"]}_{params["criterion_name"]}_{use_channel}'
    os.makedirs(output_dir, exist_ok=True)

    seed_torch(params.get('random_seed', 42))
    train_dataloader, val_dataloader = prepare_data_loaders(
        grouped_data,
        params=params,
        use_augmentation=params.get('use_augmentation', True),
        use_channel=use_channel
    )

    # Class distribution
    labels_for_counter = []
    for s in train_dataloader.dataset.samples:
        lbl = s['label']
        if hasattr(lbl, 'item'):
            lbl = lbl.item()
        labels_for_counter.append(int(lbl))
    label_counter = Counter(labels_for_counter)
    print("Training segment count per class:")
    for k in sorted(label_counter.keys()):
        label_name = class_names[k] if k < len(class_names) else str(k)
        print(f"  {label_name}: {label_counter[k]}")

    batch_losses, epoch_losses, train_accuracies, val_accuracies, val_epoch_losses = [], [], [], [], []
    best_val_accuracy = 0.0

    reset_model_weights(model)

    for epoch in range(params['num_epochs']):
        best_val_accuracy = train_and_validate(
            model, optimizer, scheduler, criterion,
            train_dataloader, val_dataloader,
            params['device'], epoch, params['num_epochs'],
            batch_losses, epoch_losses, train_accuracies, val_accuracies,
            best_val_accuracy,
            fold_idx=None,
            val_epoch_losses=val_epoch_losses,
            output_dir=output_dir
        )

        metrics_row = {
            'epoch': epoch + 1,
            'train_loss': epoch_losses[-1],
            'val_loss': val_epoch_losses[-1],
            'train_accuracy': train_accuracies[-1],
            'val_accuracy': val_accuracies[-1],
        }
        metrics_path = os.path.join(output_dir, 'metrics.csv')
        if not os.path.exists(metrics_path):
            pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)
        else:
            pd.DataFrame([metrics_row]).to_csv(metrics_path, mode='a', header=False, index=False)

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")


# Single epoch training and validation
def train_and_validate(
    model, optimizer, scheduler, criterion, train_dataloader, val_dataloader,
    device, epoch, num_epochs, batch_losses, epoch_losses, train_accuracies, val_accuracies,
    best_val_accuracy=0.0, best_model_path=None, fold_idx=None, val_epoch_losses=None, output_dir=None
):
    class_names = get_class_names()
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with tqdm(train_dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Fold {fold_idx+1 if fold_idx is not None else 'N/A'}, Epoch {epoch+1}/{num_epochs}")
        for data in tepoch:
            inputs = data['signal'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()
            outputs = _ensure_tensor_logits(model(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_predictions += (outputs.argmax(1) == labels).sum().item()
            total_predictions += labels.size(0)
            batch_losses.append(loss.item())
            tepoch.set_postfix(loss=loss.item())

    epoch_loss = running_loss / max(len(train_dataloader), 1)
    epoch_accuracy = correct_predictions / max(total_predictions, 1)
    epoch_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

# Validation
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    y_true, y_probs = [], []

    with torch.no_grad():
        for data in val_dataloader:
            inputs = data['signal'].to(device)
            labels = data['label'].to(device)
            outputs = _ensure_tensor_logits(model(inputs))
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_probs.extend(probs)
            y_true.extend(labels.cpu().numpy())
            correct_val += (outputs.argmax(1) == labels).sum().item()
            total_val += labels.size(0)

    val_loss = running_val_loss / max(len(val_dataloader), 1)
    val_accuracy = correct_val / max(total_val, 1)
    val_accuracies.append(val_accuracy)
    if val_epoch_losses is not None:
        val_epoch_losses.append(val_loss)

    print(f"Val Epoch {epoch+1} | Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")

    # save ROC curve for this fold/epoch
    if len(y_true) > 0:
        save_multiclass_roc(
            y_true=np.array(y_true),
            y_scores=np.array(y_probs),
            fold_idx=fold_idx if fold_idx is not None else 0,
            output_dir=output_dir,
            class_names=class_names
        )

    if val_accuracy > best_val_accuracy and best_model_path:
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model at epoch {epoch+1} with accuracy {val_accuracy:.4f}")

    # Scheduler
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()

    return max(val_accuracy, best_val_accuracy)


# Utils
def reset_model_weights(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def save_multiclass_roc(y_true, y_scores, fold_idx, output_dir, class_names=None):
    class_names = class_names or get_class_names()
    num_classes = len(class_names)
    """
    Save ROC curves and AUC scores for multi-class classification (One-vs-Rest).
    For 3 classes: SR / AF / Other.
    """
    if num_classes == 2:
        y_arr = np.asarray(y_true)
        y_true_bin = np.column_stack([1 - y_arr, y_arr])
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    roc_dir = os.path.join(output_dir, 'roc')
    os.makedirs(roc_dir, exist_ok=True)

    auc_scores = {}
    label_map = {i: name for i, name in enumerate(class_names)}

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[i] = roc_auc

        label_name = label_map.get(i, f'Class_{i}')
        # Per-class ROC CSV
        curve_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        curve_df.to_csv(os.path.join(roc_dir, f'fold_{fold_idx + 1}_{label_name}_roc.csv'), index=False)

        plt.plot(fpr, tpr, label=f'{label_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curve - Fold {fold_idx + 1}')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plot_path = os.path.join(roc_dir, f'fold_{fold_idx + 1}_multiclass_roc.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Multiclass ROC curve saved to {plot_path}")

    # Save AUC summary
    auc_path = os.path.join(roc_dir, 'multiclass_auc_scores.csv')
    auc_entry = {'fold': fold_idx + 1}
    auc_entry.update({f'{label_map.get(i, f"class_{i}")}_auc': v for i, v in auc_scores.items()})

    if not os.path.exists(auc_path):
        pd.DataFrame([auc_entry]).to_csv(auc_path, index=False)
    else:
        pd.DataFrame([auc_entry]).to_csv(auc_path, mode='a', header=False, index=False)

    print(f"AUC scores for Fold {fold_idx + 1} saved to {auc_path}")


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def plot_fold_metrics(fold, epoch_losses, val_epoch_losses, train_accuracies, val_accuracies, num_epochs, output_dir):
    """
    Plot training and validation metrics for a specific fold and save the plots.
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    x_range = range(1, num_epochs + 1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(x_range, epoch_losses, label='Train Loss', linewidth=2)
    plt.plot(x_range, val_epoch_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold} Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(plots_dir, f'fold_{fold}_loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Fold {fold} loss curve saved at {loss_plot_path}")

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(x_range, train_accuracies, label='Train Accuracy', linewidth=2)
    plt.plot(x_range, val_accuracies, label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Fold {fold} Accuracy Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    acc_plot_path = os.path.join(plots_dir, f'fold_{fold}_accuracy_curve.png')
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Fold {fold} accuracy curve saved at {acc_plot_path}")


def initialize_components(params):
    """Convenience: build model/optimizer/scheduler/criterion as a pack."""
    model = get_model(params['model_name'], params['device'])
    optimizer = get_optimizer(
        optimizer_name=params['optimizer_name'],
        model=model,
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
        momentum=params.get('momentum', 0.9)
    )
    scheduler = get_scheduler(
        scheduler_name=params['scheduler_name'],
        optimizer=optimizer,
        step_size=params.get('step_size', 2),
        gamma=params.get('gamma', 0.1),
        patience=params.get('patience', 10)
    )
    criterion = get_criterion(params['criterion_name'])
    return model, optimizer, scheduler, criterion


# Finalize: retrain on full data and SAVE model
def finalize_model_training(grouped_data, all_subjects, params):
    """Retrain on the full training set and save the final model for downstream evaluation."""
    model_name = params["model_name"]
    criterion_name = params["criterion_name"]
    use_channel = params.get("use_channel", "ppg_acc")
    run_name = params.get("run_name", None)
    run_suffix = f"_{run_name}" if run_name else ""

    save_root = os.path.join(
        "model_save_0524normalsegment",
        f"model_{model_name}_{criterion_name}_{use_channel}{run_suffix}"
    )

    def model_fn():
        return get_model(model_name, params['device'])

    def optimizer_fn(model):
        return get_optimizer(
            optimizer_name=params['optimizer_name'],
            model=model,
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=params.get('momentum', 0.9)
        )

    def scheduler_fn(optimizer):
        return get_scheduler(
            scheduler_name=params['scheduler_name'],
            optimizer=optimizer,
            step_size=params.get('step_size', 2),
            gamma=params.get('gamma', 0.1),
            patience=params.get('patience', 10)
        )

    def criterion_fn():
        return get_criterion(criterion_name)

    if params.get('multi_seed', False):
        seeds = params.get('seeds', [params.get('random_seed', 42)])
        for seed in seeds:
            print(f"\nRetrain with full data | seed = {seed}")
            seed_torch(seed)

            model = model_fn()
            optimizer = optimizer_fn(model)
            scheduler = scheduler_fn(optimizer)
            criterion = criterion_fn()

            output_dir = os.path.join(
                save_root,
                f'training_metrics_seed_{seed}'
            )
            os.makedirs(output_dir, exist_ok=True)

            retrain_on_full_dataset(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                grouped_data=grouped_data,
                all_subjects=all_subjects,
                params=params,
                best_model_path=os.path.join(output_dir, 'final_best_model.pth'),
                output_dir=output_dir
            )
            del model
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print("Retraining on full dataset (single-seed)...")
        model = model_fn()
        optimizer = optimizer_fn(model)
        scheduler = scheduler_fn(optimizer)
        criterion = criterion_fn()

        output_dir = os.path.join(
            save_root,
            f'training_metrics'
        )
        os.makedirs(output_dir, exist_ok=True)

        retrain_on_full_dataset(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            grouped_data=grouped_data,
            all_subjects=all_subjects,
            params=params,
            best_model_path=os.path.join(output_dir, 'final_best_model.pth'),
            output_dir=output_dir
        )


def retrain_on_full_dataset(
    model,
    optimizer,
    scheduler,
    criterion,
    grouped_data,
    all_subjects,
    params,
    best_model_path,
    output_dir
):
    """Retrain on all training subjects and save final model, training curves, confusion matrix, and report."""
    fs = params.get('fs', 32)
    batch_size = params.get('batch_size', 64)
    each_slice_length = params.get('each_slice_length', 30)
    use_channel = params.get('use_channel', 'ppg_acc')
    num_epochs = params.get('num_epochs', 20)
    device = params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    base_transform = Compose([
        BandpassFilter(fs=fs, use_channel=use_channel),
        Normalize()
    ])

    augmentations = None
    if params.get('use_augmentation', False):
        augmentations = [
            AddNoise(noise_level=params.get('noise_level', 0.01)),
            TimeShift(shift_range=params.get('shift_range', 0.1)),
            AmplitudeScaling(scale_range=params.get('scale_range', 0.1))
        ]

    full_train_dataset = PPGDataset(
        data_dict=grouped_data,
        subjects=all_subjects,
        each_slice_length=each_slice_length,
        fs=fs,
        base_transform=base_transform,
        augmentations=augmentations,
        use_channel=use_channel
    )

    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)

    batch_losses, epoch_losses, train_accuracies = [], [], []
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with tqdm(full_train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for data in tepoch:
                inputs = data['signal'].to(device)
                labels = data['label'].to(device)

                optimizer.zero_grad()
                outputs = _ensure_tensor_logits(model(inputs))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct_predictions += (outputs.argmax(1) == labels).sum().item()
                total_predictions += labels.size(0)
                batch_losses.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        epoch_loss = running_loss / max(len(full_train_loader), 1)
        epoch_acc = correct_predictions / max(total_predictions, 1)
        epoch_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, LR: {current_lr:.6f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    torch.save(model.state_dict(), best_model_path)
    print(f"Final model saved to {best_model_path}")

    # save train curves
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': epoch_losses,
        'train_accuracy': train_accuracies
    })
    metrics_path = os.path.join(output_dir, 'final_training_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Training metrics saved to {metrics_path}")

    plot_retrain_metrics(batch_losses, epoch_losses, train_accuracies, num_epochs, output_dir)

    # a final confusion matrix and classification report on the full training set
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for data in full_train_loader:
            inputs, labels = data['signal'].to(device), data['label'].to(device)
            outputs = _ensure_tensor_logits(model(inputs))
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    plot_confusion_matrix(all_labels, all_preds, output_dir)
    save_classification_report(all_labels, all_preds, output_dir)


# Final plots & reports for full retrain
def plot_confusion_matrix(y_true, y_pred, output_dir, class_names=None):
    class_names = class_names or get_class_names()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Final Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'plots', 'final_confusion_matrix.png')
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")


def save_classification_report(y_true, y_pred, output_dir, class_names=None):
    class_names = class_names or get_class_names()
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(output_dir, 'final_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved at {report_path}")


def plot_retrain_metrics(batch_losses, epoch_losses, train_accuracies, total_epochs, output_dir):
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Batch loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(batch_losses) + 1), batch_losses, label='Batch Loss', linewidth=1)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Batch Loss Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'final_batch_loss_curve.png'))
    plt.close()

    # Epoch loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, total_epochs + 1), epoch_losses, label='Epoch Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch Loss Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'final_epoch_loss_curve.png'))
    plt.close()

    # Train accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, total_epochs + 1), train_accuracies, label='Training Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'final_training_accuracy_curve.png'))
    plt.close()
