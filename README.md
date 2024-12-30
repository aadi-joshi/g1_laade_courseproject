# Disease Prediction System
A machine learning-based system for predicting diseases based on symptoms using multiple classification algorithms.

## Overview
This project implements a disease prediction system using multiple machine learning classifiers. It combines predictions from Support Vector Machines (SVM), Naive Bayes, and Random Forest algorithms to make robust disease predictions based on patient symptoms.

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

### Model Evaluation
- Uses accuracy score as the primary metric
- Implements cross-validation scoring
- Employs ensemble voting through mode calculation of predictions

### Data Preprocessing
- Label encoding for categorical disease labels
- Feature engineering for symptom representation
- Train-test splitting for model validation

## Libraries Used

### Core Libraries
- **numpy**: Array operations and numerical computations
- **pandas**: Data manipulation and analysis
  - DataFrame operations
  - CSV file handling
  - Data preprocessing

### Machine Learning Libraries
- **scikit-learn**:
  - `LabelEncoder`: Converting categorical labels to numerical form
  - `train_test_split`: Dataset splitting
  - `SVC`: Support Vector Classification
  - `GaussianNB`: Naive Bayes classification
  - `RandomForestClassifier`: Random Forest implementation
  - `accuracy_score`: Performance metric calculation

### Statistical Libraries
- **scipy.stats**: Statistical functions
- **statistics**: Basic statistical operations like mode calculation

## Potential Optimizations Using PCA and Linear Algebra

### Principal Component Analysis (PCA)
1. **Dimensionality Reduction**
   - Could reduce the symptom feature space while preserving variance
   - Potentially improve computational efficiency
   - Help identify key symptom patterns

2. **Feature Selection**
   - Identify most significant symptom combinations
   - Remove redundant or highly correlated symptoms

### Linear Algebra Optimizations
1. **Matrix Operations**
   - Optimize symptom vector operations using sparse matrices
   - Implement efficient matrix multiplication for feature transformations

2. **Eigenvalue Decomposition**
   - Analyze feature importance through eigenvalue analysis
   - Improve feature selection through eigenvector analysis

3. **SVD (Singular Value Decomposition)**
   - Could be used for noise reduction in symptom data
   - Alternative approach to dimensionality reduction

## Usage
```python
# Example usage
predictions = predictDisease("Vomiting,Fatigue,Anxiety,Lethargy")
```

## Model Performance
The system uses ensemble voting from three different classifiers to make predictions. Each model contributes its prediction, and the final prediction is determined by taking the mode of all predictions.

## Data Requirements
- Training data should be provided in CSV format
- Each row should represent a case with symptoms as features
- The last column should contain the disease label

## Notes
- The system assumes symptoms are independent (Naive Bayes assumption)
- Current implementation uses binary symptom representation (present/absent)
- Model performance depends on the quality and quantity of training data