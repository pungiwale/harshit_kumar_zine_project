"""
stage one for model development
"""
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import label_binarize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Reading training data
f_train = input("Enter the training file: ")
data = pd.read_csv(f_train, on_bad_lines='skip')
x_train = data.drop("attack_cat", axis=1)
y_train = data["attack_cat"]

# Reading test data
f_test = input("Enter the test file: ")
df = pd.read_csv(f_test)
x_test = df.drop("attack_cat", axis=1)
y_test = df["attack_cat"]

# Identify non-numeric columns
non_numeric_columns = df.select_dtypes(include=['object']).columns
print("Non-numeric columns to be removed:", non_numeric_columns)

# Drop non-numeric columns
x_train = x_train.drop(columns=non_numeric_columns)
x_test = x_test.drop(columns=non_numeric_columns)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
}

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# Binarize the output labels for ROC AUC
classes = np.unique(y_train)
y_test_binarized = label_binarize(y_test, classes=classes)

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    # Classification Report
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))
    
    # AUC-ROC Curve
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(x_test)
    elif hasattr(model, "decision_function"):
        y_probs = model.decision_function(x_test)
    else:
        raise AttributeError(f"{type(model).__name__} does not support probability predictions")
    
    # Normalize if necessary
    if not np.allclose(y_probs.sum(axis=1), 1):
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    
    # Calculate AUC for each class
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_probs[:, i])
        auc = roc_auc_score(y_test_binarized[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f"{name} (Class {class_label}, AUC = {auc:.2f})")
    
    # Plot AUC-ROC Curve
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curves')
    plt.legend()
    plt.show()
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {name}")
    plt.show()
