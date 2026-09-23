# EasyNodule_DL: 3D Pulmonary Nodule Malignancy Classification

An automated deep learning and radiomics framework for pulmonary (lung) nodule malignancy classification (**Benign vs. Malignant**) from 3D Chest Computed Tomography (CT) scans. Built using **Gabor-Augmented Capsule Networks (CapsNet)** with Dynamic Routing and **3D Volumetric Clinical Radiomics**.

---

## Performance Summary & Metrics

### 1. Benchmark Results

| Model Architecture | Accuracy (%) | ROC-AUC | Precision (Malignant) | Specificity (Benign) | F1-Score | Training Time (CPU) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Initial Codebase** | ~54.8% (Majority Guess) | 0.500 | 0.548 | 0.0% (Trivial) | 0.708 | >30 mins (Hung) |
| **Fast Gabor-CapsNet (Axial View)** | **61.06%** | **0.7018** | **0.661** | **62.7% (32/51)** | **0.627** | **<60 seconds** |
| **3D Volumetric Radiomics & Synergy** | **71.68%** | **0.7056** | **0.857 (85.7%)** | **88.2% (45/51)** | **0.692** | **<50 seconds** |

---

### 2. Confusion Matrices (Test Set: $N = 113$)

#### A. Fast Gabor-CapsNet (Axial Orthogonal View)
```
                  Predicted Benign (0)    Predicted Malignant (1)
True Benign (0)           32                       19
True Malignant (1)        25                       37
```
* **True Negatives (Benign)**: 32 / 51 (**62.7% Specificity**)
* **True Positives (Malignant)**: 37 / 62 (59.7% Sensitivity)
* **Overall Accuracy**: **61.06%** | **ROC-AUC**: **0.7018**

#### B. 3D Volumetric Radiomics & Deep CapsNet Synergy ("The Right Way")
```
                  Predicted Benign (0)    Predicted Malignant (1)
True Benign (0)           45                        6
True Malignant (1)        26                       36
```
* **True Negatives (Benign)**: 45 / 51 (**88.2% Specificity**)
* **True Positives (Malignant)**: 36 / 62
* **Malignancy Precision**: **85.7%** (Only 6 false positives out of 51 benign cases!)
* **Overall Accuracy**: **71.68% (81 / 113 correct)** | **ROC-AUC**: **0.7056**

---

## Architectural Breakthroughs & Engineering Fixes

### 1. Eliminating the 30-Minute Training Freeze
* **Problem**: In the initial implementation, `tfio.experimental.filter.gabor` was called inside Python eager mode on CPU across 1,024 intermediate feature maps for every batch across 35 epochs ($1,155\text{ batches} \times 2\text{s} \approx 40\text{ minutes}$).
* **Solution**: Implemented authentic Hinton CapsNet dimensions ($D_{\text{prim}}=8, D_{\text{sec}}=16$), reducing parameters from 40.8M to 2.5M. Pre-filtering spatial Gabor textures at the slice level executes in **0.05 seconds**, reducing total training time from >30 minutes to **under 60 seconds on CPU**.

### 2. Resolving Vanishing Gradients & Collapsed Predictions
* **Problem**: Primary capsule activations were not squashed before dynamic routing. Unbounded ReLU activations caused secondary capsule norm magnitudes to blow up to $\approx 3,500$. At saturation ($\|v\| \approx 1.0$), the squash derivative dropped to $10^{-11}$, freezing margin loss at $0.4050$ ($0.5 \times (1 - 0.1)^2$) and predicting only the majority class.
* **Solution**: Applied vector squashing ($u = \text{squash}(u)$) to primary capsules as specified by Hinton et al. Gradients remain active throughout training ($L < 0.17$), and decision thresholds are calibrated via Youden's $J$-index ($J = \text{Sensitivity} + \text{Specificity} - 1$).

### 3. Clinical 3D Volumetric Radiomics Pipeline
Radiologists evaluate pulmonary nodules using 3D volumetric morphology:
* **Concentric Radial Shells (0–30 mm)**: Evaluates radial density falloff from the inner core to the surrounding lung parenchyma.
* **Multi-Threshold Volumetric Segmentation**: Measures nodule volume and equivalent spherical diameter ($d_{\text{eq}}$) across 7 attenuation thresholds.
* **3D Margin Spiculation & Sphericity**: Measures boundary gradient magnitude ($\|\nabla I\|$) and surface compactness.
* **Multi-Planar Haralick GLCM Textures**: Computes energy, homogeneity, contrast, and correlation across Axial, Coronal, and Sagittal planes.
* **Game-Theoretic Decision Fusion (THJ)**: Arbitrates hard multi-view conflicts via zero-sum matrix games and Nash equilibrium (`nashpy`).

---

## Repository Structure

```
EasyNodule_DL/
├── train_and_visualize.ipynb   # Complete Jupyter Notebook (all 13 cells executed with plots)
├── train_capsnet.py           # Standalone CLI training script with model checkpointing
├── model/
│   └── CapsNetCode.py         # Original CapsNet architecture reference
├── preprocessing/
│   └── Preporcessing.py       # Clinical preprocessing & feature extraction functions
├── weights/
│   └── ModelX/
│       └── capsule-X.*        # Trained CapsNet model checkpoint weights
├── THJdata/                   # Reference indices & test splits for game-theoretic fusion
└── README.md                  # Project documentation & benchmark metrics
```

---

## Quickstart & Usage

### Environment Setup (Conda / Pip)
```bash
conda create -n easynodule python=3.11 -y
conda activate easynodule
pip install tensorflow tensorflow-io numpy scikit-learn scikit-image opencv-python pillow matplotlib nashpy scipy
```

### 1. Run the Interactive Notebook
Launch Jupyter and open `train_and_visualize.ipynb`:
```bash
jupyter notebook train_and_visualize.ipynb
```
All 13 cells are pre-executed with:
* Multi-planar CT slice visualizations (Axial, Coronal, Sagittal)
* Through-depth volume slice progressions
* Fast Gabor texture filtering demonstrations
* Training curves (active loss descent and validation AUC)
* Capsule latent reconstruction visualizations
* Feature importance bar charts for the top 12 radiological biomarkers
* Side-by-side Confusion Matrices and Comparative ROC Curves

### 2. Train via Command Line
Train an authentic Capsule Network with validation checkpointing on CPU or GPU:
```bash
# Train on Axial view (ModelX) with 8 epochs and Cosine Decay
python train_capsnet.py --axis X --epochs 8 --batch-size 32

# Train all three orthogonal axes (X, Y, Z)
python train_capsnet.py --axis all --epochs 8 --batch-size 32
```

---

## Citation & References
* Sabour, S., Frosst, N., & Hinton, G. E. (2017). *Dynamic Routing Between Capsules*. NeurIPS.
* Armato, S. G., et al. (2011). *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)*. Medical Physics.
* Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). *Textural Features for Image Classification*. IEEE TSMC.
