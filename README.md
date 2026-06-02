# Identification and Quantification of Plastic/Microplastic Mixtures from Elemental Composition

**Code for the study:**  
*Utilizing machine learning to accelerate the identification and quantification of plastics or microplastics via their basic elemental compositions*  https://doi.org/10.1016/j.jhazmat.2026.142535

<img width="865" height="567" alt="image" src="https://github.com/user-attachments/assets/b57f8657-3287-4b43-a35d-7e275b3ea96f" />
---

## 📌 Project Overview

This repository provides the implementation of a machine learning framework for the **identification and quantification of plastic (microplastic) compositions** using only **basic elemental descriptors** (C, H, O, N and derived ratios).

The method is intended as a complementary, low-cost, and chemically interpretable supplement to spectroscopic approaches such as FTIR, Raman, and NIR, rather than a direct replacement under all real-world conditions.

In its current form, the framework should be regarded as a methodological proof-of-concept, with strongest applicability to well-controlled laboratory mixtures and aggregated plastic samples.

---

## 🔍 Key Features

- Synthetic dataset construction for polymer mixture modeling
- Environmentally informed composition distribution generation
- Multi-output regression for plastic mixture quantification
- Model comparison across multiple machine learning algorithms
- Random Forest–based training, interpretation, and visualization
- Learning curve analysis and robustness assessment
- PCA-based visualization of elemental feature space
- Experimental dataset validation

---

## 📂 Repository Structure & Scripts

### Dataset Generation
- `generate_synthetic_dataset.py`  
  Generate synthetic plastic mixture datasets for model training and testing.

- `generate_environmental_distribution_dataset.py`  
  Construct datasets with composition distributions adjusted according to typical environmental occurrence frequencies.

### Model Comparison

- `train_and_compare_ml_models.py`  
  Train and compare multiple machine learning models (e.g., DT, RF, SVR, GBR).
  
### Model Training

- `RF_model.py`  
  Train a Random Forest model for multi-output regression of plastic compositions.

- `rf_results_plot.py`  
  Visualize Random Forest prediction results and residual distributions.
  
### Model Evaluation & Analysis
- `model_validation.py`  
  Validate trained models using experimental datasets.

- `learning_curve_analysis.py`  
  Analyze learning behavior with respect to training set size and number of trees.

- `elemental_noise_robustness.py`  
  Evaluate model robustness under simulated elemental measurement noise.

### Visualization & Interpretation
- `pca_polymers_visualization.py`  
  Perform PCA on polymer elemental features and visualize clustering behavior.

---

## ⚙️ Requirements

- **Python**: 3.9  
- **Core dependencies**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `openpyxl`
