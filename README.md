# Road Segmentation — U-Net vs Foundation Models

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)
![EPFL](https://img.shields.io/badge/EPFL-CS--433-red)
![F1 Score](https://img.shields.io/badge/F1%20Score-89.2%25-brightgreen)

Semantic segmentation of aerial road imagery using a fine-tuned U-Net architecture, outperforming SAM (Segment Anything Model) on this domain-specific task.

**Team:** Mylène Berruyer · Ghalia Bennani · Mehdi Hajoub — EPFL CS-433 Machine Learning, 2023

---

## Results

| Model | F1 Score | Accuracy | Notes |
|-------|----------|----------|-------|
| **U-Net (ours)** | **89.2%** | **94.1%** | Iterative transfer learning |
| SAM (ViT-H, zero-shot) | ~82% | ~89% | No fine-tuning |
| U-Net from scratch | ~85% | ~91% | Baseline |

> **Key finding:** Task-specific transfer learning with alternating learning rates outperforms a general-purpose foundation model (SAM) on narrow aerial imagery domains.

---

## Method

### Architecture
U-Net with pretrained encoder backbone, fine-tuned end-to-end on aerial road imagery.

### Training Strategy — Iterative Transfer Learning
- **Phase 1:** Train decoder only (encoder frozen) — lr = 1e-4
- **Phase 2:** Fine-tune full network — lr = 1e-5
- Alternating between phases until convergence

### Data Augmentation
- 45° rotation (key for road detection in all orientations)
- Horizontal and vertical flips
- Brightness/contrast jitter

---

## Demo

Run inference on a single image:

```bash
python run.py --input path/to/image.png --output predictions/
```

Or explore the full evaluation pipeline in the notebook:

```bash
jupyter notebook test_models.ipynb
```

The notebook walks through:
1. Loading pretrained weights
2. Running inference on test images
3. Comparing U-Net vs SAM predictions side by side
4. Computing F1 and accuracy metrics

---

## Project Structure

road-segmentation/
├── models/ # Model architectures (U-Net, baselines)
├── utils/ # Data loading, augmentation, metrics
├── run.py # Training & inference entry point
├── helpers.py # Utility functions
├── test_models.ipynb # Evaluation notebook & visual results
├── environment.yml # Conda environment
└── Project_2-3.pdf # Full report


---

## Installation

```bash
conda env create -f environment.yml
conda activate road-seg
python run.py
```

---

## Report

The full project report (methodology, ablation study, error analysis) is available as [`Project_2-3.pdf`](./Project_2-3.pdf).
