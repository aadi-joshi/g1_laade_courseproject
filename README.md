# Disease Prediction System
A machine learning-based system for predicting diseases based on symptoms using multiple classification algorithms and advanced linear algebra techniques.

## Overview
This project implements a disease prediction system using multiple machine learning classifiers enhanced with dimensionality reduction and linear algebra techniques. It combines predictions from Support Vector Machines (SVM), Naive Bayes, and Random Forest algorithms, utilizing PCA and eigendecomposition for robust feature analysis and disease predictions based on patient symptoms.

## Key Machine Learning & Data Science Concepts

### Classification Algorithms Used
1. **Support Vector Machine (SVM)**
   - A supervised learning algorithm that finds the optimal hyperplane to separate different classes
   - Effective for high-dimensional spaces
   - Uses kernel trick for non-linear classification

2. **Naive Bayes**
   - Probabilistic classifier based on Bayes' theorem
   - Assumes independence between features
   - Particularly effective for text classification and medical diagnosis

3. **Random Forest**
   - Ensemble learning method using multiple decision trees
   - Reduces overfitting through majority voting
   - Provides feature importance rankings

### Advanced Linear Algebra Implementation

#### Principal Component Analysis (PCA)
- Implemented using scikit-learn's PCA with 95% variance retention
- Reduces dimensionality while preserving essential symptom patterns
- Key components:
  ```python
  pca = PCA(n_components=0.95)
  X_pca = pca.fit_transform(X)
  ```
- Benefits:
  - Noise reduction in symptom data
  - Improved computational efficiency
  - Preservation of critical symptom relationships

#### Correlation Matrix Analysis
- Implements robust correlation analysis with numerical stability checks
- Features:
  - Handles missing values and infinite correlations
  - Ensures matrix symmetry and positive semi-definiteness
  - Example implementation:
    ```python
    correlation_matrix = np.nan_to_num(np.corrcoef(X.T))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    ```

#### Eigendecomposition
- Analyzes feature importance through eigenvalue decomposition
- Implementation includes:
  - Eigenvalue calculation for feature importance ranking
  - Eigenvector analysis for principal directions
  ```python
  eigenvalues, eigenvectors = eigh(correlation_matrix)
  sorted_idx = eigenvalues.argsort()[::-1]
  ```

#### Singular Value Decomposition (SVD)
- Used for latent feature analysis
- Implementation:
  ```python
  U, S, Vt = svd(X, full_matrices=False)
  ```
- Applications:
  - Decomposition of symptom matrix into fundamental patterns
  - Additional dimensionality reduction technique
  - Numerical stability assessment through condition number

### Feature Analysis and Importance

#### Variance Analysis
- Zero variance feature removal
- Variance-based feature selection:
  ```python
  variance = X.var()
  non_zero_var_cols = variance[variance != 0].index
  ```

#### Confidence Weighting
- Eigenvalue-based confidence scoring
- Implementation:
  ```python
  confidence_weights = np.nan_to_num(eigenvalues / np.sum(eigenvalues))
  ```

### Model Evaluation and Prediction
- Combines multiple models with PCA-transformed data
- Implements confidence-weighted prediction
- Returns comprehensive prediction metrics including:
  - Individual model predictions
  - Feature importance scores
  - Variance explained ratios

## Libraries Used

### Core Libraries
- **numpy**: Array operations and numerical computations
- **pandas**: Data manipulation and analysis
- **scipy**: Scientific computing and linear algebra operations

### Machine Learning Libraries
- **scikit-learn**:
  - Classification algorithms
  - PCA implementation
  - Model evaluation metrics

## Advanced Features

### Symptom Pattern Analysis
- Implements `analyze_symptom_patterns()` function
- Returns:
  - Principal components
  - Variance explained ratios
  - Feature importance metrics
  - Singular values
  - Condition number for numerical stability

### Error Handling
- Robust error handling in prediction pipeline
- Validation of matrix operations
- Safe PCA transformation with error checking

## Usage
```python
# Basic disease prediction
predictions = predictDisease("Vomiting,Fatigue,Anxiety,Lethargy")

# Get all available symptoms
symptoms = get_all_symptoms()

# Analyze symptom patterns
patterns = analyze_symptom_patterns()
```

## Data Requirements
- Training data in CSV format
- Features should represent binary symptom presence
- Last column should contain disease labels
- Data should be complete (no missing values) for PCA analysis

## Notes and Limitations
- PCA assumes linear relationships between symptoms
- Matrix operations require sufficient memory for large datasets
- System performance depends on:
  - Quality and quantity of training data
  - Feature variance and correlation structure
  - Numerical stability of matrix operations

## Future Improvements
- Implementation of sparse matrix operations for large datasets
- Alternative dimensionality reduction techniques (t-SNE, UMAP)
- Advanced eigendecomposition methods for better feature analysis
- Integration of non-linear PCA variants