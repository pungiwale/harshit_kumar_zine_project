#stage_three.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# ---------------------------------
# STEP 1: Load and Preprocess Data
# ---------------------------------
f=input("path_to_your_dataset.csv")
df = pd.read_csv(f)

# Encode target variable
encoder = LabelEncoder()
df["attack_cat"] = encoder.fit_transform(df["attack_cat"])

# Extract features and target
features = df.drop(columns=["attack_cat"])
target = df["attack_cat"]

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# -------------------------------
# STEP 2: Build GAN for Data Generation
# -------------------------------
latent_dim = 100  # Size of noise vector

# Generator model
def build_generator():
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(features_scaled.shape[1], activation='tanh')  # Output shape same as feature count
    ])
    return model

# Discriminator model
def build_discriminator():
    model = Sequential([
        Dense(256, input_dim=features_scaled.shape[1]),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification (real/fake)
    ])
    return model

# Compile GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Combined GAN model
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
generated_sample = generator(gan_input)
gan_output = discriminator(generated_sample)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

# -------------------------------
# STEP 3: Train GAN
# -------------------------------
batch_size = 64
epochs = 5000
losses = []

for epoch in range(epochs):
    # Train Discriminator
    idx = np.random.randint(0, features_scaled.shape[0], batch_size)
    real_samples = features_scaled[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_samples = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)) * 0.9)
    d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    losses.append((d_loss[0], g_loss))

    if epoch % 500 == 0:
        print(f"Epoch {epoch} - D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")

# Plot GAN Loss Curve
plt.plot([loss[0] for loss in losses], label="Discriminator Loss")
plt.plot([loss[1] for loss in losses], label="Generator Loss")
plt.legend()
plt.title("GAN Training Loss")
plt.show()

# -------------------------------
# STEP 4: Generate Synthetic Data
# -------------------------------
num_samples = 5000
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_data = generator.predict(noise)
synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)
synthetic_df = pd.DataFrame(synthetic_data_rescaled, columns=features.columns)
synthetic_df.to_csv("synthetic_UNSW-NB15.csv", index=False)

# T-test Analysis
t_stats, p_values = ttest_ind(features_scaled, synthetic_data, equal_var=False)
plt.hist(p_values, bins=50, alpha=0.7)
plt.title("T-test P-values for Synthetic vs Real Data")
plt.show()

# -------------------------------
# STEP 5: Train Classification Models
# -------------------------------
# Combine real and synthetic data
combined_features = pd.concat([df.drop(columns=["attack_cat"]), synthetic_df], axis=0)
combined_target = np.concatenate([target, np.random.choice(target, size=len(synthetic_df))])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_target, test_size=0.2, random_state=42)

# Train Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Train Neural Network
y_train_nn = to_categorical(y_train)
y_test_nn = to_categorical(y_test)

nn_model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(len(encoder.classes_), activation="softmax")
])
nn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = nn_model.fit(X_train, y_train_nn, epochs=20, batch_size=32, validation_data=(X_test, y_test_nn), verbose=1)

# -------------------------------
# STEP 6: Evaluate Models
# -------------------------------
# Plot NN Loss Curve
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Neural Network Loss Curve")
plt.show()

# Compute ROC-AUC
from sklearn.preprocessing import label_binarize
y_test_binarized = label_binarize(y_test, classes=np.arange(len(encoder.classes_)))
y_pred_dt = dt_model.predict_proba(X_test)
y_pred_rf = rf_model.predict_proba(X_test)
y_pred_nn = nn_model.predict(X_test)

roc_auc_dt = roc_auc_score(y_test_binarized, y_pred_dt, average='macro', multi_class='ovr')
roc_auc_rf = roc_auc_score(y_test_binarized, y_pred_rf, average='macro', multi_class='ovr')
roc_auc_nn = roc_auc_score(y_test_binarized, y_pred_nn, average='macro', multi_class='ovr')

print(f"ROC AUC (Decision Tree): {roc_auc_dt:.4f}")
print(f"ROC AUC (Random Forest): {roc_auc_rf:.4f}")
print(f"ROC AUC (Neural Network): {roc_auc_nn:.4f}")

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15,5))
for ax, model_name, y_pred in zip(axes, ["Decision Tree", "Random Forest", "Neural Network"],
                                  [dt_model.predict(X_test), rf_model.predict(X_test), np.argmax(y_pred_nn, axis=1)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
plt.show()

# Classification Reports
print("Decision Tree:\n", classification_report(y_test, dt_model.predict(X_test)))
print("Random Forest:\n", classification_report(y_test, rf_model.predict(X_test)))
print("Neural Network:\n", classification_report(y_test, np.argmax(y_pred_nn, axis=1)))
