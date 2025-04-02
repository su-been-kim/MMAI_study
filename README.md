# 🔊🎥 Visual Sound Localization – Extended Implementation

This repository provides an extended and modular implementation of visual sound source localization. It builds upon [Localizing Visual Sounds the Hard Way](https://github.com/hche11/Localizing-Visual-Sounds-the-Hard-Way), with flexible training and testing pipelines, alternative dataset loaders, and structured experimentation scripts.

---

## Overview

The project focuses on multimodal audio-visual modeling, specifically targeting sound source localization from video using cross-modal representations. It supports training and evaluation on the VGG-SoundSource dataset and includes multiple loader variants and training pipelines.

---

## Project Slides (Figma)

For a high-level visual summary of the system architecture, data flow, and experimentation plan, refer to the following Figma board:

👉 [View on Figma – Individual Study (MMAI Lab)](https://www.figma.com/deck/2Dd0OsIHjgfEAX1WT5teJ6/Individual-study-MMAI-lab?node-id=37-1245&t=EY8wbAW7S6mKhjqy-1)

---

## Project Structure

### Training
- `train.py`: Original training code from the base implementation
- `train_baseline.py`: Baseline training pipeline with default settings
- `train_final_s_m.py`, `train_semantic_multiview.py`: Training scripts with semantic and multiview loss integration
- `train_localization.py`, `train_localization_2.py`: Training with localization loss objectives
- `train_multiview.py`: Training with multiview-only configuration

### Evaluation
- `test.py`: Evaluation script for trained models

### Dataset
- `DatasetLoader_origin.py`: Original dataset loader for VGGSoundSource
- `DatasetLoader_s_m.py`: Loader adapted for semantic & multiview supervision

### Utilities
- `topk_similarity.py`: Generates top-k similarity JSON outputs
- `utils.py`: General utility functions (logging, configuration, etc.)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/su-been-kim/MMAI_study.git
cd MMAI_study
```

---

### 2. Prepare the dataset

The project is built and tested using the **VGG-SoundSource** dataset.  
It is derived from the official [VGGSound Dataset](https://www.robots.ox.ac.uk/~vgg/data/vggsound/), provided by the Visual Geometry Group at the University of Oxford.

---

