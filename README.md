# elevatelabs-task7
# Breast Cancer Classification Using SVMs

## Overview
This project implements Support Vector Machines (SVMs) for binary classification using the Breast Cancer Dataset. The goal is to classify tumor samples as **Benign** or **Malignant** based on extracted features. Both **linear** and **non-linear (RBF kernel)** SVM models are trained and evaluated. Decision boundaries are visualized using two selected features.

## Dataset
- **Name:** Breast Cancer Dataset  
- **Source:** Click [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset](click here to download dataset) to download.  
- **Features:** Various tumor-related attributes including **radius_mean** and **texture_mean**.  
- **Target Variable:** Diagnosis (**Benign = 0**, **Malignant = 1**).  

## Tools Used
- **Scikit-learn** – Model training and evaluation  
- **NumPy** – Data manipulation  
- **Matplotlib** – Visualization  
- **Mlxtend** – Decision boundary plotting  

## Project Workflow
### 1. Data Preparation
- Load the dataset (`breast-cancer.csv`).  
- Drop unnecessary columns (e.g., **ID**).  
- Encode target variable (**diagnosis**).  
- Standardize numerical features using **StandardScaler**.  
- Split data into **training** and **testing sets**.  

### 2. Train SVM Models
- Train an **SVM with a linear kernel**.  
- Train an **SVM with an RBF kernel** (non-linear decision boundary).  
- Visualize **decision boundaries** using two selected features (**radius_mean** and **texture_mean**).  

### 3. Hyperparameter Tuning
- Perform **GridSearchCV** to optimize:  
  - `C` (Regularization parameter)  
  - `gamma` (Kernel coefficient for RBF kernel)  
- Select the **best hyperparameters** based on cross-validation accuracy.  

### 4. Model Evaluation
- Compute **cross-validation accuracy** for both linear and RBF models.  
- Train the final **SVM model with optimal parameters**.  
- Evaluate the final model on the **test set**.  

### 5. Decision Boundary Visualization
- Generate **plots** for linear and RBF kernel decision boundaries.  
- Use **mesh grids** for contour visualization.  

## Results
- **Cross-validation accuracy (2D Features - Linear Kernel):** `88.35%`  
- **Cross-validation accuracy (2D Features - RBF Kernel):** `89.89%`  
- **Final test accuracy with best hyperparameters:** `97.37%`  
- **Decision boundary plots** showcasing classification regions.  

## How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib mlxtend
