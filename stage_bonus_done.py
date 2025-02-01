#stage_bonus.py
"""
stage_bonus
"""
"""
boss i tried to add deepfool but it showwd than 
clwverhans does not have deepfool so i commented it
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from cleverhans.tf2.attacks import fast_gradient_method, carlini_wagner_l2 #, deepfool
from sklearn.metrics import roc_curve, auc
import stage_one


# from tensorflow.keras.models import load_model

# Load models from stage_one.py
nn_model = stage_one.models["Neural Network"]
dt_model = stage_one.models["Decision Tree"]
rf_model = stage_one.models["Random Forest"]
# Load test dataset
test_data = pd.read_csv("test_data.csv")

# Separate features and target variable
x_test = test_data.drop(columns=["attack_cat"]).values.astype(np.float32)
y_test = test_data["attack_cat"].values
# to implement FGSM
def generate_fgsm_examples(model, X, eps=0.01):
    x_tensor = tf.convert_to_tensor(X)
    adv_x = fast_gradient_method(model, x_tensor, eps, np.inf)
    return adv_x.numpy()
#...
x_adv_fgsm = generate_fgsm_examples(nn_model, x_test)
#to implement CW
def generate_cw_examples(model, X):
    x_tensor = tf.convert_to_tensor(X)
    adv_x = carlini_wagner_l2(model, x_tensor, targeted=False)
    return adv_x.numpy()
#...
x_adv_cw = generate_cw_examples(nn_model, x_test)
# to implement Deepfool
"""
def generate_deepfool_examples(model, X):
    x_tensor = tf.convert_to_tensor(X)
    adv_x = deepfool(model, x_tensor)
    return adv_x.numpy()

x_adv_deepfool = generate_deepfool_examples(nn_model, x_test)
"""
#to evaluate models over the adversaries attack
def evaluate_model(model, X, y, model_name="Neural Network", attack_name="Clean"):
    # Get model predictions
    if model_name == "Neural Network":
        y_pred = model.predict(X).argmax(axis=1)  # Convert softmax to class labels
    else:  # For Decision Tree & Random Forest
        y_pred = model.predict(X)

    # Compute accuracy
    acc = np.mean(y_pred == y)
    print(f"\n{model_name} Performance on {attack_name} Data:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=np.unique(y),
    yticklabels=np.unique(y))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix - {attack_name} Data")
    plt.show()

    # ROC Curve (only for Neural Network, as tree models don’t have probability outputs)
    if model_name == "Neural Network":
        y_prob = model.predict(X)
        fpr, tpr, _ = roc_curve(y, y_prob[:, 1], pos_label=1)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{attack_name} AUC: {auc(fpr, tpr):.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve - {attack_name} Data")
        plt.legend()
        plt.show()

# Evaluate on clean data
evaluate_model(nn_model, x_test, y_test, "Neural Network", "Clean")
evaluate_model(dt_model, x_test, y_test, "Decision Tree", "Clean")
evaluate_model(rf_model, x_test, y_test, "Random Forest", "Clean")

# Evaluate on FGSM adversarial examples
evaluate_model(nn_model, x_adv_fgsm, y_test, "Neural Network", "FGSM")
evaluate_model(dt_model, x_adv_fgsm, y_test, "Decision Tree", "FGSM")
evaluate_model(rf_model, x_adv_fgsm, y_test, "Random Forest", "FGSM")

# Evaluate on CW adversarial examples
evaluate_model(nn_model, x_adv_cw, y_test, "Neural Network", "CW")
evaluate_model(dt_model, x_adv_cw, y_test, "Decision Tree", "CW")
evaluate_model(rf_model, x_adv_cw, y_test, "Random Forest", "CW")
"""
# Evaluate on DeepFool adversarial examples
evaluate_model(nn_model, x_adv_deepfool, y_test, "Neural Network", "DeepFool")
evaluate_model(dt_model, x_adv_deepfool, y_test, "Decision Tree", "DeepFool")
evaluate_model(rf_model, x_adv_deepfool, y_test, "Random Forest", "DeepFool")
"""
