# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import statistics
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from numpy.linalg import svd

# Reading the train.csv
DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)

# Encoding and data preparation
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:,:-1]
y = data.iloc[:, -1]

# Handle zero variance features
# Convert numpy boolean array to list of column names for proper pandas indexing
variance = X.var()
non_zero_var_cols = variance[variance != 0].index
X = X[non_zero_var_cols]

# Apply PCA for dimensionality reduction
# This helps reduce noise and capture most important patterns in symptoms
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X)

# Calculate correlation matrix for feature importance analysis
# This helps understand relationships between symptoms
correlation_matrix = np.nan_to_num(np.corrcoef(X.T), nan=0.0, posinf=1.0, neginf=-1.0)

# Ensure matrix is symmetric and positive semi-definite
correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
min_eig = np.min(np.real(np.linalg.eigvals(correlation_matrix)))
if min_eig < 0:
    correlation_matrix -= min_eig * np.eye(*correlation_matrix.shape)

# Calculate eigenvectors and eigenvalues of correlation matrix
# These reveal principal directions of symptom variation
eigenvalues, eigenvectors = eigh(correlation_matrix)

# Sort eigenvectors by eigenvalues to identify most important feature directions
sorted_idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# Perform SVD for latent feature analysis
# This decomposes symptom matrix into fundamental patterns
U, S, Vt = svd(X, full_matrices=False)

# Modified training data using PCA
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=24)

def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Initialize and train models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X_pca, y)
final_nb_model.fit(X_pca, y)
final_rf_model.fit(X_pca, y)

# Reading the test data
test_data = pd.read_csv("Testing.csv").dropna(axis=1)
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

symptoms = X.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    # Use the filtered columns for input data creation
    input_data = [0] * len(non_zero_var_cols)
    for symptom in symptoms:
        if symptom in data_dict["symptom_index"] and data_dict["symptom_index"][symptom] < len(non_zero_var_cols):
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
    
    input_data = np.array(input_data).reshape(1, -1)
    input_df = pd.DataFrame(input_data, columns=non_zero_var_cols)
    
    try:
        # Safe PCA transform with error handling
        input_pca = pca.transform(input_df)
        
        # Safe matrix operations with validation
        principal_features = np.dot(input_df, eigenvectors)
        similarity_scores = np.dot(principal_features, eigenvectors.T)
        
        # Make predictions
        rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_pca)[0]]
        nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_pca)[0]]
        svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_pca)[0]]
        
        # Safe confidence weight calculation
        confidence_weights = np.nan_to_num(eigenvalues / np.sum(eigenvalues), nan=0.0)
        
        final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
        
        predictions = {
            "rf_model_prediction": rf_prediction,
            "naive_bayes_prediction": nb_prediction,
            "svm_model_prediction": svm_prediction,
            "final_prediction": final_prediction,
            "feature_importance": confidence_weights[:5].tolist(),
            "variance_explained": pca.explained_variance_ratio_.tolist()
        }
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        predictions = {
            "error": "An error occurred during prediction",
            "details": str(e)
        }
    
    return predictions

def get_all_symptoms():
    return list(symptom_index.keys())

# Add function to analyze symptom patterns
def analyze_symptom_patterns():
    """
    Analyzes underlying patterns in symptom data using linear algebra concepts
    Returns: Dictionary containing various analysis metrics
    """
    analysis = {
        "principal_components": pca.components_[:5],  # Top 5 principal components
        "variance_explained": pca.explained_variance_ratio_,
        "feature_importance": np.abs(eigenvectors[:, :5]),  # Top 5 eigenvectors
        "singular_values": S[:5],  # Top 5 singular values
        "condition_number": S[0] / S[-1]  # Numerical stability metric
    }
    return analysis
