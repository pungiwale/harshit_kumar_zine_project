"""
stage_two.py

to apply smote and adaysn
"""
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Load training and testing data
train_f=input("Enter the training file name")
train_data = pd.read_csv(train_f)
test_f=input("Enter the testing file name")
test_data = pd.read_csv(test_f)
# Separate features and target variable
X_train = train_data.drop(columns=['attack_cat'])
y_train = train_data['attack_cat']
X_test = test_data.drop(columns=['attack_cat'])
y_test = test_data['attack_cat']
 #Create an imputer object with a mean filling strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the training data and transform the data
X_train = imputer.fit_transform(X_train)
# Utility functions for SMOTE and ADASYN
def smote_from_scratch(X, y, target_class, n_samples, k_neighbors=1):
    X_minority = X[y == target_class]
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X_minority)
    synthetic_samples = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(X_minority))
        sample = X_minority[idx]
        neighbors = nn.kneighbors([sample], return_distance=False)[0]
        neighbor_idx = np.random.choice(neighbors)
        neighbor = X_minority[neighbor_idx]
        diff = neighbor - sample
        synthetic_sample = sample + np.random.rand() * diff
        synthetic_samples.append(synthetic_sample)
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.full(len(X_synthetic), target_class)
    X_resampled = np.vstack([X, X_synthetic])
    y_resampled = np.hstack([y, y_synthetic])
    return X_resampled, y_resampled
def adasyn_from_scratch(X, y, target_class, n_samples, k_neighbors=5):
    X_minority = X[y == target_class]
    X_majority = X[y != target_class]
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X)
    difficulty_ratios = []
    for sample in X_minority:
        neighbors = nn.kneighbors([sample], return_distance=False)[0]
        majority_count = sum(y[neighbors] != target_class)
        difficulty_ratios.append(majority_count / k_neighbors)
    difficulty_ratios = np.array(difficulty_ratios)
    difficulty_ratios /= difficulty_ratios.sum()
    synthetic_samples = []
    for i, ratio in enumerate(difficulty_ratios):
        n_to_generate = int(ratio * n_samples)
        for _ in range(n_to_generate):
            neighbors = nn.kneighbors([X_minority[i]], return_distance=False)[0]
            neighbor_idx = np.random.choice(neighbors)
            neighbor = X[neighbor_idx]
            diff = neighbor - X_minority[i]
            synthetic_sample = X_minority[i] + np.random.rand() * diff
    synthetic_samples.append(synthetic_sample)
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.full(len(X_synthetic), target_class)
    X_resampled = np.vstack([X, X_synthetic])
    y_resampled = np.hstack([y, y_synthetic])
    return X_resampled, y_resampled
# Find the minority class and calculate the number of samples to generate
target_class = y_train.value_counts().idxmin()
n_samples = y_train.value_counts().max() - y_train.value_counts()[target_class]
"""
# Apply SMOTE
X_train_smote, y_train_smote = smote_from_scratch(X_train.values, y_train.values, target_class, n_samples)
# Apply ADASYN
X_train_adasyn, y_train_adasyn = adasyn_from_scratch(X_train.values, y_train.values, target_class, n_samples)
"""
# Apply SMOTE
X_train_smote, y_train_smote = smote_from_scratch(X_train, y_train, target_class, n_samples)
# Apply ADASYN
X_train_adasyn, y_train_adasyn = adasyn_from_scratch(X_train, y_train, target_class, n_samples)
# Verify the lengths
print(f"Length of X_train_smote: {len(X_train_smote)}")
print(f"Length of y_train_smote: {len(y_train_smote)}")
# Utility function for evaluation and plotting
def plot_roc_curve(y_true, y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(pd.get_dummies(y_true).values.ravel(), y_pred_prob.ravel())
    auc_score = roc_auc_score(pd.get_dummies(y_true).values, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC: {auc_score:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC Curve")
    plt.show()
# to evaluate the models
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    print(f"--- {model_name} ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    plot_roc_curve(y_test, y_pred_prob, model_name)
# Train and evaluate models with SMOTE data
print("Using SMOTE Data")
tree_model = DecisionTreeClassifier(random_state=42)
evaluate_model(tree_model, X_train_smote, y_train_smote, X_test, y_test, "Decision Tree (SMOTE)")
forest_model = RandomForestClassifier(random_state=42)
evaluate_model(forest_model, X_train_smote, y_train_smote, X_test, y_test, "Random Forest (SMOTE)")
nn_model = MLPClassifier(random_state=42, max_iter=500, verbose=True)
nn_model.fit(X_train_smote, y_train_smote)
plt.plot(nn_model.loss_curve_)
plt.title("Neural Network Loss Curve (SMOTE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
evaluate_model(nn_model, X_train_smote, y_train_smote, X_test, y_test, "Neural Network (SMOTE)")
# Train and evaluate models with ADASYN data
print("Using ADASYN Data")
tree_model = DecisionTreeClassifier(random_state=42)
evaluate_model(tree_model, X_train_adasyn, y_train_adasyn, X_test, y_test, "Decision Tree (ADASYN)")
forest_model = RandomForestClassifier(random_state=42)
evaluate_model(forest_model, X_train_adasyn, y_train_adasyn, X_test, y_test, "Random Forest (ADASYN)")
nn_model = MLPClassifier(random_state=42, max_iter=500, verbose=True)
nn_model.fit(X_train_adasyn, y_train_adasyn)
plt.plot(nn_model.loss_curve_)
plt.title("Neural Network Loss Curve (ADASYN)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
evaluate_model(nn_model, X_train_adasyn, y_train_adasyn, X_test, y_test, "Neural Network (ADASYN)")
