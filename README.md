# Part 2: Computer Vision — CNN-Based Manufacturing Defect Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem Statement

Build a Convolutional Neural Network (CNN) to classify product surface images into four categories:

| Class | Description |
|-------|-------------|
| `normal` | Defect-free product surface |
| `scratch` | Linear marks or scratches on the surface |
| `dent` | Circular indentation or dent |
| `stain` | Coloured blotch or stain on the surface |

This is a **multi-class image classification** problem solved using a custom CNN built in TensorFlow/Keras.

---

## Repository Structure

```
part-2-cnn-computer-vision/
│
├── README.md                         ← This file
├── notebook.ipynb                    ← Complete Jupyter Notebook solution
├── requirements.txt                  ← Python dependencies
│
├── sample_predictions/
│   └── prediction_outputs.png        ← Visual predictions on test images
│
└── results/
    ├── accuracy_loss_curves.png      ← Training/validation accuracy & loss
    └── confusion_matrix.png          ← Confusion matrix on test set
```

---

## Dataset

- **Source:** Synthetic Manufacturing Defect Image Dataset
- **Total images:** 480 (120 per class — perfectly balanced)
- **Image dimensions:** 96 × 96 pixels (RGB)
- **Format:** PNG, organised in per-class folders + `labels.csv`

---

## CNN Architecture

```
Input (64×64×3)
│
├── Block 1: Conv2D(32) → ReLU → BN → MaxPool → Dropout(0.25)
├── Block 2: Conv2D(64) × 2 → ReLU → BN → MaxPool → Dropout(0.25)
├── Block 3: Conv2D(128) × 2 → ReLU → BN → MaxPool → Dropout(0.30)
│
├── Flatten
├── Dense(256) → ReLU → Dropout(0.50)
└── Dense(4) → Softmax
```

---

## Results Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | ~95% |
| Optimizer | Adam (lr=0.001) |
| Loss Function | Categorical Cross-Entropy |
| Epochs | 30 (early stopping) |

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/part-2-cnn-computer-vision.git
cd part-2-cnn-computer-vision
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Download the dataset and place it so the path matches:

```
dataset/
└── part_2_cnn_computer_vision/
    ├── images/
    │   ├── normal/
    │   ├── scratch/
    │   ├── dent/
    │   └── stain/
    └── labels.csv
```

### 4. Run the notebook

```bash
jupyter notebook notebook.ipynb
```

---

## CNN Concept Explanations

### What is Convolution?

A small filter (kernel) — typically 3×3 pixels — slides across the input image. At each position it multiplies the filter values by the overlapping pixel values and sums them to produce a single output value. This creates a **feature map** that highlights specific visual patterns such as edges, textures, and shapes.

### Why is Pooling Used?

Max Pooling reduces spatial dimensions by keeping only the maximum value in each window (typically 2×2). This:
- **Reduces computation** by halving spatial size
- **Adds translation invariance** — a feature found slightly off-centre still fires
- **Prevents overfitting** by discarding less important information

### Why is ReLU Commonly Used?

`ReLU(x) = max(0, x)` is preferred because:
- It avoids the **vanishing gradient problem** that plagues sigmoid/tanh in deep networks
- It is **computationally trivial** — just a threshold operation
- **Sparse activations** (many zeros) lead to efficient, disentangled representations

### Why CNNs over Regular Feed-Forward Networks?

| Regular MLP | CNN |
|------------|-----|
| Treats each pixel independently | Exploits spatial structure — nearby pixels are related |
| Huge parameter count (e.g. 96×96×3 = 27,648 inputs) | Parameter sharing via filters — much fewer parameters |
| Not translation-invariant | Detects a pattern anywhere in the image |
| Cannot learn local texture/edge features | Hierarchical feature learning: edges → textures → shapes |

---

## Business Use Case: Manufacturing Quality Inspection

A CNN defect classifier can be deployed on a factory production line with a camera and edge GPU. Each product image is classified in under 50 ms:

- **`normal`** → Product passes inspection and moves to packaging
- **`scratch` / `dent`** → Product is rejected and diverted for rework
- **`stain`** → Product is flagged for cleaning or manual review

**Industry applications:**
- **Automotive:** Paint and panel inspection before delivery
- **Electronics:** PCB surface defect detection in semiconductor fabs
- **Textiles:** Real-time fabric flaw detection on weaving lines
- **Consumer goods:** Packaging and product surface quality assurance

---

## Requirements

See `requirements.txt` for full list. Key dependencies:

- TensorFlow ≥ 2.10
- scikit-learn ≥ 1.3
- Pillow ≥ 9.0
- matplotlib ≥ 3.7
- seaborn ≥ 0.12
- pandas ≥ 1.5
- numpy ≥ 1.23
- jupyter ≥ 1.0
