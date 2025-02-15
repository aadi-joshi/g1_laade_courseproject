# Disease Prediction System - Mathematical Foundations

A machine learning system for disease prediction emphasizing Principal Component Analysis (PCA) and linear algebra concepts.

## Mathematical Foundations

### Core Components

1. **Principal Component Analysis (PCA)**
   - **Objective**: Orthogonal transformation of correlated features into linearly uncorrelated principal components
   - **Mathematical Process**:
     1. **Covariance Matrix**:
        ```math
        C = \frac{1}{n-1}X^TX
        ```
        Where X is the mean-centered data matrix
     2. **Eigen Decomposition**:
        ```math
        Cv = \lambda v
        ```
        Solved using `scipy.linalg.eigh` for Hermitian matrices
     3. **Variance Retention**:
        ```math
        \text{Explained Variance Ratio} = \frac{\lambda_i}{\sum{\lambda}}
        ```
        Components selected to retain 95% total variance
     4. **Projection**:
        ```math
        X_{pca} = XW
        ```
        Where W contains eigenvectors (principal components)

2. **Correlation Matrix Analysis**
   - Adjusted to be positive semi-definite:
     ```math
     C_{adj} = C + |\lambda_{min}|I \quad \text{if} \ \lambda_{min} < 0
     ```
   - Ensures valid eigenvalue decomposition

3. **Feature Importance**
   - Calculated from normalized eigenvalues:
     ```math
     w_i = \frac{\lambda_i}{\sum{\lambda}}
     ```
   - Represents each principal component's contribution

## Implementation Details

### Key Components

1. **Data Preparation**
   - Zero-variance feature elimination:
     ```math
     \sigma^2 = \frac{1}{n}\sum(x_i - \mu)^2 > 0
     ```
   - Label encoding for categorical prognosis

2. **PCA Implementation**
   - Retains 95% variance (n_components=0.95)
   - Components sorted by descending eigenvalues
   - Projection maintains Mahalanobis distance:
     ```math
     d(x,y) = \sqrt{(x-y)^TC^{-1}(x-y)}
     ```

3. **Model Training**
   - Three classifiers trained on PCA-transformed data:
     - Support Vector Machine (SVM)
     - Gaussian Naive Bayes
     - Random Forest
   - Ensemble prediction using mode of individual predictions

4. **Symptom Analysis**
   - Feature projection:
     ```math
     z = xW_k \quad (W_k = \text{top }k\text{ eigenvectors})
     ```
   - Similarity scoring using:
     ```math
     s = zW_k^T
     ```

## How to Use

1. **Install Requirements**
```bash
pip install numpy pandas scikit-learn scipy flask
```

2. **Run Application**
```bash
python app.py
```

3. **Access Interface**
Visit `http://localhost:5000` in your browser

## Mathematical Results Interpretation

### Prediction Output
- **Feature Importance**: First 5 eigenvalues (relative importance of principal directions)
- **Variance Explained**: Cumulative explained variance ratio for each component
- **Principal Components**: Loading factors for original features in PCA space

### Analysis Metrics
```python
{
  "principal_components": "Orthogonal basis vectors for symptom space",
  "variance_explained": "Information content distribution",
  "feature_importance": "Dominant symptom combinations",
  "covariance_rank": "True dimensionality of symptom data"
}
```

## Theoretical Insights

1. **Dimensionality Reduction**
   - Original 132 symptoms → k components (k ≈ 10-15 for 95% variance)
   - Johnson-Lindenstrauss Lemma guarantees distance preservation

2. **Numerical Stability**
   - Condition Number: 
     ```math
     \kappa(C) = \frac{\lambda_{max}}{\lambda_{min}}
     ```
     Controlled via correlation matrix adjustment

3. **Geometric Interpretation**
   - PCA performs rotation of symptom space axes
   - First principal component maximizes variance:
     ```math
     w_{(1)} = \arg\max_{\|w\|=1} \text{Var}(Xw)
     ```

## Conclusion

This project demonstrates the power of linear algebra in medical diagnostics:
- PCA enables efficient symptom pattern recognition
- Eigen decomposition reveals fundamental disease axes
- Matrix analysis ensures numerical stability
- Dimensionality reduction maintains predictive power while reducing complexity

The mathematical framework provides both computational efficiency and theoretical insights into symptom-disease relationships.
