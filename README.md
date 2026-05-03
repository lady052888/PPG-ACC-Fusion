# Lightweight Multimodal PPG-Accelerometer Fusion for Clinically Realistic AF/SR/Other Classification in Elderly Inpatients

![Overview of the proposed lightweight PPG-ACC fusion pipeline. Wrist PPG and tri-axial ACC signals were collected from elderly inpatients using a Philips datalogger, while chest ECG from a Bittium Faros device served as the reference standard. ECG-derived annotations were aligned with the wearable recordings and mapped into 30-s segment-level SR, AF, and Other labels. After preprocessing, PPG and ACC signals were combined into a four-channel fusion-ready input and processed by a lightweight temporal network with residual, dilated convolutional, and temporal attention components. During evaluation, all test segments were retained, and ACC-derived motion variance was used to perform motion-percentile cumulative-coverage analysis under realistic deployment conditions.](model_overview.png)

This repository contains the experimental code for three-class rhythm classification using wrist photoplethysmography (PPG) and tri-axial accelerometer (ACC) signals. The classification task follows the clinically realistic AF/SR/Other setting described in the manuscript. The repository includes the proposed lightweight temporal residual model, reproduced literature baselines, and comparison experiments under different input configurations.



## Project Overview

The experiments are organized around three main comparisons:

- Input configuration comparison:
  comparing the proposed model under PPG-only, reproduced 4-channel input, and PPG+ACC fusion settings.

- Backbone comparison under a unified PPG+ACC input:
  comparing the proposed model with reproduced literature baselines and classical CNN backbones under matched preprocessing, training, and evaluation settings.

- Reproduced literature comparison under original or paper-matched input settings:
  evaluating reproduced baseline models using the input configurations described in their corresponding papers, while comparing them against the proposed model.

The evaluation follows a motion-aware cumulative-coverage protocol. Test segments are not excluded by signal-quality rules. Instead, performance is analyzed across ACC-based motion-percentile thresholds.

## Repository Structure

- `Networks/`  
  Model definitions for the proposed architecture, reproduced baselines, and comparison backbones.

- `results_0524normalsegment/`  
  Saved evaluation outputs, including percentile-based performance results.

- `model_save_0524normalsegment/`  
  Saved trained model checkpoints and log files.

- `run_batch_30s.py`  
  Batch runner for the main 30 s experiments.

- `run_model_compare.py`  
  Script for comparison experiments across model groups.

- `Drawfigure.ipynb`  
  Notebook for generating figures and summary plots.

## Main Experiment Groups

### EXP I: Input Configuration Comparison
Same backbone (`ResNet10_TemporalAttention_DilatedL2`) evaluated under:
- `ppg`
- `4ch`
- `ppg_acc`

These experiments correspond to the comparison of input configurations in the manuscript.

### EXP II: Backbone Comparison Under Unified PPG+ACC Input
Models evaluated under the same PPG+ACC setting:
- `Han25-BiGRU`
- `Bulut25-CNN17`
- `Shen19-50CNN`
- `mobile_net`
- `ResNet10_TemporalAttention_DilatedL2`
- `Zhao25-RhythmiNet`
- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `Liu22-DCNN`

These experiments support the backbone comparison under matched conditions.

### EXP III: Reproduced Literature Comparison With Mixed Input Settings
Models evaluated using paper-matched or reproduced input settings:
- `Bulut25-CNN17 (ppg)`
- `Han25-BiGRU (4ch)`
- `Shen19-50CNN (ppg)`
- `ResNet10_TemporalAttention_DilatedL2 (ppg_acc)`
- `Liu22-DCNN (ppg)`
- `Zhao25-RhythmiNet (ppg_acc)`

These experiments are used for comparison with reproduced literature baselines.

## Proposed Model

The proposed architecture is:

`ResNet10_TemporalAttention_DilatedL2`

It is a lightweight 1D residual temporal network designed for AF/SR/Other classification from 30 s wrist PPG and ACC segments. The model combines:
- a compact residual backbone,
- a dilated convolution block for broader temporal context,
- and a temporal attention module.

## Inputs and Task

- Input signals:
  wrist PPG and optional tri-axial ACC
- Segment length:
  30 s
- Sampling frequency:
  32 Hz
- Classification task:
  three-class rhythm classification (`AF`, `SR`, `Other`)

## Evaluation Outputs

The repository includes scripts and notebooks for generating:
- macro-AUC and micro-AUC curves,
- class-wise one-vs-rest AUC curves,
- confusion matrices at selected motion-percentile thresholds,
- parameter and FLOP summaries.

## Running Experiments

Run the main batch experiments with:

```bash
python run_batch_30s.py
```


