#!/usr/bin/env python
"""
Kepler Exoplanet 3-Class Detection - Local Training Script
Trains Genesis CNN, XGBoost, and RandomForest models + Ensemble
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("Kepler Exoplanet 3-Class Detection - Local Training")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python: {sys.version.split()[0]}")
print("=" * 80)

# Load Data
print("\n[1/9] Loading Data...")
features = pd.read_csv('koi_lightcurve_features_no_label.csv')
labels_full = pd.read_csv('q1_q17_dr25_koi.csv')
print(f"  Features: {features.shape}")
print(f"  Labels: {labels_full.shape}")

# Extract disposition column
if 'koi_disposition' in labels_full.columns:
    disposition_col = 'koi_disposition'
elif 'disposition' in labels_full.columns:
    disposition_col = 'disposition'
else:
    print("ERROR: No disposition column found!")
    sys.exit(1)

labels = labels_full[[disposition_col]].copy()
labels.columns = ['disposition']

# Align features and labels by ID
feature_id_col = None
for col in ['kepoi_name', 'kepid', 'koi_id']:
    if col in features.columns:
        feature_id_col = col
        break

if feature_id_col is None:
    features_aligned = features.iloc[:len(labels)]
else:
    label_id_col = None
    for col in ['kepoi_name', 'kepid', 'koi_id']:
        if col in labels_full.columns:
            label_id_col = col
            break

    if label_id_col:
        merged = pd.merge(
            features, labels_full[[label_id_col, disposition_col]],
            left_on=feature_id_col, right_on=label_id_col, how='inner'
        )
        features_aligned = merged.drop(columns=[label_id_col, disposition_col, feature_id_col])
        labels = merged[[disposition_col]]
        labels.columns = ['disposition']
    else:
        features_aligned = features

print(f"  Aligned Data: {features_aligned.shape}")
print(f"  Label Distribution:")
print(labels['disposition'].value_counts())

# Remove ALL non-numeric columns
print(f"  Removing non-numeric columns...")
numeric_cols = features_aligned.select_dtypes(include=[np.number]).columns
features_aligned = features_aligned[numeric_cols]

print(f"  Final Features: {features_aligned.shape}")

# Encode Labels
print("\n  Applying One-Hot Encoding...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels['disposition'])
label_mapping = {i: name for i, name in enumerate(label_encoder.classes_)}
print(f"  Label Mapping: {label_mapping}")

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))
print(f"  One-Hot Shape: {y_onehot.shape}")

# Train/Test Split
print("\n[2/9] Train/Test Split (75%/25%)...")
combined = pd.concat([
    features_aligned.reset_index(drop=True),
    pd.DataFrame(y_onehot, columns=[f'class_{i}' for i in range(y_onehot.shape[1])])
], axis=1)

combined_shuffled = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

feature_cols = [col for col in combined_shuffled.columns if not col.startswith('class_')]
label_cols = [col for col in combined_shuffled.columns if col.startswith('class_')]

X = combined_shuffled[feature_cols].values
y = combined_shuffled[label_cols].values
y_labels = np.argmax(y, axis=1)

X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y_labels
)

y_train_labels = np.argmax(y_train_raw, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print(f"  Train: X={X_train_raw.shape}, y={y_train_raw.shape}")
print(f"  Test: X={X_test.shape}, y={y_test.shape}")
print(f"  Train Classes: {np.bincount(y_train_labels)}")
print(f"  Test Classes: {np.bincount(y_test_labels)}")

# Handle Missing Values
print("\n[3/9] Handling Missing Values...")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')  # Use median to fill NaN
X_train_imputed = imputer.fit_transform(X_train_raw)
X_test_imputed = imputer.transform(X_test)
joblib.dump(imputer, 'models/feature_imputer.pkl')
print("  Imputer saved: models/feature_imputer.pkl")

# Feature Scaling
print("\n[4/9] Feature Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
joblib.dump(scaler, 'models/feature_scaler.pkl')
print("  Scaler saved: models/feature_scaler.pkl")

# SMOTE
print("\n[5/9] Applying SMOTE...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train, y_train_labels_balanced = smote.fit_resample(X_train_scaled, y_train_labels)
print(f"  After SMOTE: {X_train.shape}")
print(f"  Balanced Classes: {np.bincount(y_train_labels_balanced)}")

y_train = np.zeros((len(y_train_labels_balanced), y_onehot.shape[1]))
for i, label in enumerate(y_train_labels_balanced):
    y_train[i, label] = 1

# Train XGBoost
print("\n[6/9] Training XGBoost...")
xgb_start = time.time()

class_counts = np.bincount(y_train_labels_balanced)
sample_weights = np.array([1.0 / class_counts[label] for label in y_train_labels_balanced])
sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.1,
    random_state=RANDOM_STATE, tree_method='hist', n_jobs=-1, eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train_labels_balanced, sample_weight=sample_weights, verbose=False)

xgb_time = time.time() - xgb_start
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_xgb_proba = xgb_model.predict_proba(X_test_scaled)
xgb_acc = accuracy_score(y_test_labels, y_pred_xgb)
xgb_f1 = f1_score(y_test_labels, y_pred_xgb, average='weighted')

print(f"  Training Time: {xgb_time:.2f}s")
print(f"  Test Accuracy: {xgb_acc:.4f}")
print(f"  F1-Score: {xgb_f1:.4f}")
xgb_model.save_model('models/xgboost_3class.json')
print("  Model saved: models/xgboost_3class.json")

# Train Random Forest
print("\n[7/9] Training Random Forest...")
rf_start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=20, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0
)
rf_model.fit(X_train, y_train_labels_balanced)

rf_time = time.time() - rf_start
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_rf_proba = rf_model.predict_proba(X_test_scaled)
rf_acc = accuracy_score(y_test_labels, y_pred_rf)
rf_f1 = f1_score(y_test_labels, y_pred_rf, average='weighted')

print(f"  Training Time: {rf_time:.2f}s")
print(f"  Test Accuracy: {rf_acc:.4f}")
print(f"  F1-Score: {rf_f1:.4f}")
joblib.dump(rf_model, 'models/random_forest_3class.pkl')
print("  Model saved: models/random_forest_3class.pkl")

# Train Genesis CNN
print("\n[8/9] Training Genesis CNN...")
cnn_trained = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    cnn_start = time.time()

    class_weights = {}
    for i in range(len(class_counts)):
        class_weights[i] = len(y_train_labels_balanced) / (len(class_counts) * class_counts[i])

    def build_cnn(input_dim, num_classes):
        model = models.Sequential([
            layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
            layers.Conv1D(64, 50, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 50, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(16),
            layers.Dropout(0.25),
            layers.Conv1D(128, 12, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(128, 12, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.AveragePooling1D(8),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    cnn_model = build_cnn(X_train.shape[1], y_train.shape[1])
    print("  Model architecture created")

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)

    print("  Training (this may take a while)...")
    history = cnn_model.fit(
        X_train, y_train, validation_data=(X_test_scaled, y_test),
        epochs=50, batch_size=32, class_weight=class_weights,
        callbacks=[early_stop, reduce_lr], verbose=2
    )

    cnn_time = time.time() - cnn_start
    y_pred_cnn_proba = cnn_model.predict(X_test_scaled, verbose=0)
    y_pred_cnn = np.argmax(y_pred_cnn_proba, axis=1)
    cnn_acc = accuracy_score(y_test_labels, y_pred_cnn)
    cnn_f1 = f1_score(y_test_labels, y_pred_cnn, average='weighted')

    print(f"  Training Time: {cnn_time:.2f}s")
    print(f"  Test Accuracy: {cnn_acc:.4f}")
    print(f"  F1-Score: {cnn_f1:.4f}")
    print(f"  Epochs: {len(history.history['loss'])}")

    cnn_model.save('models/genesis_cnn_3class.keras')
    print("  Model saved: models/genesis_cnn_3class.keras")
    cnn_trained = True

except Exception as e:
    print(f"  TensorFlow error: {e}")
    print("  Skipping CNN training")
    cnn_acc = 0.0
    cnn_f1 = 0.0
    cnn_time = 0.0

# Ensemble Model
print("\n[9/9] Creating Ensemble (Voting)...")

from sklearn.base import BaseEstimator, ClassifierMixin

class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self._estimator_type = "classifier"
        self.classes_ = model.classes_
    def fit(self, X, y):
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)

estimators = [('xgb', XGBWrapper(xgb_model)), ('rf', rf_model)]
ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=[1, 1])
ensemble.fit(X_train, y_train_labels_balanced)

y_pred_ensemble = ensemble.predict(X_test_scaled)
y_pred_ensemble_proba = ensemble.predict_proba(X_test_scaled)
ensemble_acc = accuracy_score(y_test_labels, y_pred_ensemble)
ensemble_f1 = f1_score(y_test_labels, y_pred_ensemble, average='weighted')

print(f"  Test Accuracy: {ensemble_acc:.4f}")
print(f"  F1-Score: {ensemble_f1:.4f}")
joblib.dump(ensemble, 'models/ensemble_voting_3class.pkl')
print("  Model saved: models/ensemble_voting_3class.pkl")

# Save Metadata
print("\n[10/10] Generating Reports...")
metadata = {
    "created_at": datetime.now().isoformat(),
    "label_mapping": label_mapping,
    "num_classes": len(label_mapping),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "feature_dim": X_train.shape[1],
    "models": {
        "xgboost": {
            "accuracy": float(xgb_acc), "f1_score": float(xgb_f1),
            "training_time_sec": float(xgb_time), "file": "xgboost_3class.json"
        },
        "random_forest": {
            "accuracy": float(rf_acc), "f1_score": float(rf_f1),
            "training_time_sec": float(rf_time), "file": "random_forest_3class.pkl"
        },
        "ensemble_voting": {
            "accuracy": float(ensemble_acc), "f1_score": float(ensemble_f1),
            "file": "ensemble_voting_3class.pkl"
        }
    }
}

if cnn_trained:
    metadata["models"]["genesis_cnn"] = {
        "accuracy": float(cnn_acc), "f1_score": float(cnn_f1),
        "training_time_sec": float(cnn_time), "file": "genesis_cnn_3class.keras"
    }

with open('models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print("  Metadata saved: models/metadata.json")

# Confusion Matrices
print("  Generating confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices - 3-Class Kepler Exoplanet Detection', fontsize=16, fontweight='bold')

models_viz = [
    ('XGBoost', y_pred_xgb, 'Blues'),
    ('Random Forest', y_pred_rf, 'Greens'),
    ('Ensemble', y_pred_ensemble, 'Purples')
]

for ax, (name, y_pred, cmap) in zip(axes, models_viz):
    cm = confusion_matrix(y_test_labels, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=[label_mapping[i] for i in range(len(label_mapping))],
                yticklabels=[label_mapping[i] for i in range(len(label_mapping))],
                cbar_kws={'label': 'Count'})
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("  Saved: figures/confusion_matrices.png")

# Performance Comparison
print("  Generating performance comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

model_names = ['XGBoost', 'Random Forest', 'Ensemble']
accuracies = [xgb_acc, rf_acc, ensemble_acc]
f1_scores = [xgb_f1, rf_f1, ensemble_f1]
colors = ['#3498db', '#2ecc71', '#9b59b6']

bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.05)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

bars2 = ax2.bar(model_names, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, f1_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/performance_comparison.png', dpi=300, bbox_inches='tight')
print("  Saved: figures/performance_comparison.png")

# Summary
print("\n" + "=" * 80)
print("*** TRAINING COMPLETED! ***")
print("=" * 80)
print("\n*** Model Performance ***")
print(f"  XGBoost       - Acc: {xgb_acc:.4f}, F1: {xgb_f1:.4f}, Time: {xgb_time:.2f}s")
print(f"  Random Forest - Acc: {rf_acc:.4f}, F1: {rf_f1:.4f}, Time: {rf_time:.2f}s")
if cnn_trained:
    print(f"  Genesis CNN   - Acc: {cnn_acc:.4f}, F1: {cnn_f1:.4f}, Time: {cnn_time:.2f}s")
print(f"  Ensemble      - Acc: {ensemble_acc:.4f}, F1: {ensemble_f1:.4f}")

best_acc = max(xgb_acc, rf_acc, ensemble_acc)
if best_acc == ensemble_acc:
    best_model = "Ensemble"
elif best_acc == xgb_acc:
    best_model = "XGBoost"
else:
    best_model = "Random Forest"

print(f"\n*** BEST MODEL: {best_model} (Accuracy: {best_acc:.4f}) ***")

print("\n*** Output Files ***")
print("  models/")
print("    - xgboost_3class.json")
print("    - random_forest_3class.pkl")
if cnn_trained:
    print("    - genesis_cnn_3class.keras")
print("    - ensemble_voting_3class.pkl")
print("    - feature_scaler.pkl")
print("    - metadata.json")
print("  figures/")
print("    - confusion_matrices.png")
print("    - performance_comparison.png")

print("\n" + "=" * 80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("\n*** Classification Reports ***")
print("\n[XGBoost]")
print(classification_report(y_test_labels, y_pred_xgb,
                          target_names=[label_mapping[i] for i in range(len(label_mapping))]))

print("\n[Random Forest]")
print(classification_report(y_test_labels, y_pred_rf,
                          target_names=[label_mapping[i] for i in range(len(label_mapping))]))

print("\n[Ensemble]")
print(classification_report(y_test_labels, y_pred_ensemble,
                          target_names=[label_mapping[i] for i in range(len(label_mapping))]))

print("\n*** ALL TASKS COMPLETED! Models ready for inference. ***")
