#!/usr/bin/env python
"""
Complete Ensemble Creation and Visualization
"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import json

print("=" * 80)
print("Completing Ensemble Model and Visualizations")
print("=" * 80)

# Create figures directory
os.makedirs('figures', exist_ok=True)

# Load data for evaluation
print("\n[1/5] Loading Data...")
features_df = pd.read_csv('koi_lightcurve_features_no_label.csv')
labels_df = pd.read_csv('q1_q17_dr25_koi.csv')

# Align data
features_aligned = features_df.merge(
    labels_df[['kepoi_name', 'koi_disposition']],
    left_on=features_df.columns[0],
    right_on='kepoi_name',
    how='inner'
)
features_aligned = features_aligned.drop(columns=['kepoi_name', 'koi_disposition'])

# Remove non-numeric columns
numeric_cols = features_aligned.select_dtypes(include=[np.number]).columns
features_aligned = features_aligned[numeric_cols]

# Prepare labels
y_onehot = pd.get_dummies(labels_df.loc[features_aligned.index, 'koi_disposition']).values
label_mapping = {i: name for i, name in enumerate(pd.get_dummies(labels_df['koi_disposition']).columns)}

# Split data
# Create properly named columns
feature_col_names = [f'feature_{i}' for i in range(features_aligned.shape[1])]
label_col_names = [f'class_{i}' for i in range(y_onehot.shape[1])]

combined = pd.concat([
    pd.DataFrame(features_aligned.values, columns=feature_col_names),
    pd.DataFrame(y_onehot, columns=label_col_names)
], axis=1)

combined_shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)
feature_cols = [col for col in combined_shuffled.columns if col.startswith('feature_')]
label_cols = [col for col in combined_shuffled.columns if col.startswith('class_')]

X = combined_shuffled[feature_cols].values
y = combined_shuffled[label_cols].values
y_labels = np.argmax(y, axis=1)

X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y_labels
)

y_test_labels = np.argmax(y_test, axis=1)

# Load preprocessors
print("\n[2/5] Loading Preprocessors...")
imputer = joblib.load('models/feature_imputer.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

# Recreate training data for ensemble fitting
from imblearn.over_sampling import SMOTE
y_train_labels = np.argmax(y_train_raw, axis=1)
X_train_imputed = imputer.transform(X_train_raw)
X_train_scaled = scaler.transform(X_train_imputed)

smote = SMOTE(random_state=42)
X_train, y_train_labels_balanced = smote.fit_resample(X_train_scaled, y_train_labels)

# Load models
print("\n[3/5] Loading Models...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('models/xgboost_3class.json')

rf_model = joblib.load('models/random_forest_3class.pkl')

# Create custom ensemble class
print("\n[4/5] Creating Ensemble...")

class SimpleEnsemble:
    """Simple ensemble that averages predictions from multiple models"""

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        # Get probabilities from all models
        all_probas = []
        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            all_probas.append(proba * weight)

        # Average weighted probabilities
        avg_proba = np.mean(all_probas, axis=0)
        return avg_proba

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

# Create ensemble with equal weights
ensemble = SimpleEnsemble(
    models=[xgb_model, rf_model],
    weights=[1.0, 1.0]
)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test_scaled)
ensemble_acc = accuracy_score(y_test_labels, y_pred_ensemble)
ensemble_f1 = f1_score(y_test_labels, y_pred_ensemble, average='weighted')

print(f"  Test Accuracy: {ensemble_acc:.4f}")
print(f"  F1-Score: {ensemble_f1:.4f}")

# Save ensemble
joblib.dump(ensemble, 'models/ensemble_voting_3class.pkl')
print("  Model saved: models/ensemble_voting_3class.pkl")

# Generate predictions for all models
print("\n[5/5] Generating Visualizations...")

models = {
    'XGBoost': xgb_model,
    'Random Forest': rf_model,
    'Ensemble': ensemble
}

# Create confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test_labels, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=list(label_mapping.values()),
                yticklabels=list(label_mapping.values()))
    axes[idx].set_title(f'{name} Confusion Matrix')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('figures/confusion_matrices.png', dpi=150, bbox_inches='tight')
print("  Saved: figures/confusion_matrices.png")

# Create performance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
accuracies = []
f1_scores = []
model_names = []

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_labels, y_pred)
    f1 = f1_score(y_test_labels, y_pred, average='weighted')

    accuracies.append(acc)
    f1_scores.append(f1)
    model_names.append(name)

axes[0].bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

axes[1].bar(model_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1].set_title('Model F1-Score Comparison')
axes[1].set_ylabel('F1-Score')
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(f1_scores):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('figures/performance_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved: figures/performance_comparison.png")

# Create metadata
print("\n[6/6] Generating Metadata...")

# Get model file sizes
model_files = {
    'xgboost': 'models/xgboost_3class.json',
    'random_forest': 'models/random_forest_3class.pkl',
    'genesis_cnn': 'models/genesis_cnn_3class.keras',
    'ensemble': 'models/ensemble_voting_3class.pkl'
}

metadata = {
    "created_at": datetime.now().isoformat(),
    "label_mapping": label_mapping,
    "num_classes": len(label_mapping),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "feature_dim": X_test.shape[1],
    "models": {}
}

for model_name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_labels, y_pred)
    f1 = f1_score(y_test_labels, y_pred, average='weighted')

    key = model_name.lower().replace(' ', '_')
    file_path = model_files.get(key, '')

    metadata["models"][key] = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "file": file_path,
        "file_size_mb": round(os.path.getsize(file_path) / (1024**2), 2) if os.path.exists(file_path) else 0
    }

# Save metadata
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("  Saved: models/metadata.json")

print("\n" + "=" * 80)
print("COMPLETE! All models, visualizations, and metadata generated.")
print("=" * 80)
print("\nGenerated Files:")
print("  - models/ensemble_voting_3class.pkl")
print("  - models/metadata.json")
print("  - figures/confusion_matrices.png")
print("  - figures/performance_comparison.png")
print("\nModel Performance:")
for model_name, info in metadata["models"].items():
    print(f"  {model_name}: Accuracy={info['accuracy']:.4f}, F1={info['f1_score']:.4f}")
print("=" * 80)
