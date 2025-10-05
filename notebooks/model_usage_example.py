"""
Quick Start Example: Using Three-Class Exoplanet Detection Models

This script demonstrates how to use the model configuration module
to build, train, and evaluate all three models plus ensemble methods.

Author: ML Architecture Team
Version: 1.0.0
Date: 2025-10-05
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.model_configs import (
    ModelConfig,
    GenesisCNNConfig,
    XGBoostConfig,
    RandomForestConfig,
    EnsembleConfig,
    KerasClassifierWrapper,
    MetadataManager,
    create_all_models,
    load_all_models
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
import joblib


# ===========================
# STEP 1: LOAD AND PREPROCESS DATA
# ===========================

def load_and_preprocess_data():
    """Load and preprocess Kepler exoplanet data"""

    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 80)

    # Load preprocessed data (assumed to exist from preprocessing script)
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')

    print(f"\n✅ Data loaded:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")

    # Convert one-hot encoded labels to class indices
    y_train_indices = y_train.idxmax(axis=1).map({
        'FALSE POSITIVE': 0,
        'CANDIDATE': 1,
        'CONFIRMED': 2
    }).values

    y_test_indices = y_test.idxmax(axis=1).map({
        'FALSE POSITIVE': 0,
        'CANDIDATE': 1,
        'CONFIRMED': 2
    }).values

    # Check class distribution
    print(f"\n📊 Training set class distribution:")
    unique, counts = np.unique(y_train_indices, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {ModelConfig.CLASS_NAMES[cls]}: {count} ({count/len(y_train_indices)*100:.1f}%)")

    # Feature scaling
    print(f"\n🔄 Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    ModelConfig.ensure_model_dir()
    joblib.dump(scaler, ModelConfig.SCALER_PATH)
    print(f"  ✅ Scaler saved to {ModelConfig.SCALER_PATH}")

    # Apply SMOTE to balance classes
    print(f"\n⚖️ Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=ModelConfig.RANDOM_STATE, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_scaled,
        y_train_indices
    )

    print(f"  Before SMOTE: {X_train_scaled.shape[0]} samples")
    print(f"  After SMOTE: {X_train_balanced.shape[0]} samples")

    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"    {ModelConfig.CLASS_NAMES[cls]}: {count}")

    return (
        X_train_balanced, y_train_balanced,
        X_test_scaled, y_test_indices,
        scaler
    )


# ===========================
# STEP 2: BUILD AND TRAIN MODELS
# ===========================

def train_all_models(X_train, y_train, X_test, y_test):
    """Train all three models"""

    print("\n" + "=" * 80)
    print("STEP 2: TRAINING MODELS")
    print("=" * 80)

    results = {}

    # Initialize configs
    cnn_config = GenesisCNNConfig()
    xgb_config = XGBoostConfig()
    rf_config = RandomForestConfig()

    # ----------------------
    # Model 1: Genesis CNN
    # ----------------------
    print("\n[1/3] Training Genesis CNN...")
    print("-" * 80)

    genesis_model = cnn_config.build_model()
    print(f"  Model parameters: {genesis_model.count_params():,}")

    # Prepare data for CNN
    X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)
    y_train_cat = to_categorical(y_train, ModelConfig.N_CLASSES)
    y_test_cat = to_categorical(y_test, ModelConfig.N_CLASSES)

    # Compute class weights
    class_weights = cnn_config.get_class_weights(y_train)
    print(f"  Class weights: {class_weights}")

    # Train
    start_time = time.time()
    history = genesis_model.fit(
        X_train_cnn, y_train_cat,
        validation_data=(X_test_cnn, y_test_cat),
        epochs=cnn_config.epochs,
        batch_size=cnn_config.batch_size,
        class_weight=class_weights,
        callbacks=cnn_config.get_callbacks(),
        verbose=1
    )
    training_time = time.time() - start_time

    # Save model
    cnn_config.save_model(genesis_model)

    # Evaluate
    y_pred_proba = genesis_model.predict(X_test_cnn, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    results['genesis_cnn'] = evaluate_model(
        y_test, y_pred, y_pred_proba,
        'Genesis CNN',
        training_time
    )

    # ----------------------
    # Model 2: XGBoost
    # ----------------------
    print("\n[2/3] Training XGBoost...")
    print("-" * 80)

    xgb_model = xgb_config.build_model()
    print(f"  Estimators: {xgb_model.n_estimators}")
    print(f"  Max depth: {xgb_model.max_depth}")

    # Compute sample weights
    sample_weights = xgb_config.get_sample_weights(y_train)

    # Train
    start_time = time.time()
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        verbose=False
    )
    training_time = time.time() - start_time

    # Save model
    xgb_config.save_model(xgb_model)

    # Evaluate
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)

    results['xgboost'] = evaluate_model(
        y_test, y_pred, y_pred_proba,
        'XGBoost',
        training_time
    )

    # ----------------------
    # Model 3: Random Forest
    # ----------------------
    print("\n[3/3] Training Random Forest...")
    print("-" * 80)

    rf_model = rf_config.build_model()
    print(f"  Estimators: {rf_model.n_estimators}")
    print(f"  Max depth: {rf_model.max_depth}")
    print(f"  Class weight: {rf_model.class_weight}")

    # Train
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Save model
    rf_config.save_model(rf_model)

    # Evaluate
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)

    # OOB Score
    print(f"  OOB Score: {rf_model.oob_score_:.4f}")

    results['random_forest'] = evaluate_model(
        y_test, y_pred, y_pred_proba,
        'Random Forest',
        training_time
    )
    results['random_forest']['oob_score'] = rf_model.oob_score_

    return genesis_model, xgb_model, rf_model, results


# ===========================
# STEP 3: BUILD ENSEMBLE MODELS
# ===========================

def train_ensemble_models(genesis_model, xgb_model, rf_model, X_train, y_train, X_test, y_test):
    """Train ensemble models"""

    print("\n" + "=" * 80)
    print("STEP 3: TRAINING ENSEMBLE MODELS")
    print("=" * 80)

    ensemble_config = EnsembleConfig()
    results = {}

    # Wrap Keras model for scikit-learn compatibility
    genesis_wrapper = KerasClassifierWrapper(model=genesis_model)

    # ----------------------
    # Ensemble 1: Voting Classifier
    # ----------------------
    print("\n[1/2] Training Voting Classifier...")
    print("-" * 80)
    print(f"  Voting weights: {ensemble_config.voting_weights}")

    voting_clf = ensemble_config.build_voting_classifier(
        genesis_wrapper, xgb_model, rf_model
    )

    start_time = time.time()
    voting_clf.fit(X_train, y_train)
    training_time = time.time() - start_time

    ensemble_config.save_voting_model(voting_clf)

    y_pred = voting_clf.predict(X_test)
    y_pred_proba = voting_clf.predict_proba(X_test)

    results['ensemble_voting'] = evaluate_model(
        y_test, y_pred, y_pred_proba,
        'Ensemble (Voting)',
        training_time
    )

    # ----------------------
    # Ensemble 2: Stacking Classifier
    # ----------------------
    print("\n[2/2] Training Stacking Classifier...")
    print("-" * 80)
    print(f"  Cross-validation folds: {ensemble_config.stacking_cv}")
    print(f"  Meta-learner: Logistic Regression")

    stacking_clf = ensemble_config.build_stacking_classifier(
        genesis_wrapper, xgb_model, rf_model
    )

    start_time = time.time()
    stacking_clf.fit(X_train, y_train)
    training_time = time.time() - start_time

    ensemble_config.save_stacking_model(stacking_clf)

    y_pred = stacking_clf.predict(X_test)
    y_pred_proba = stacking_clf.predict_proba(X_test)

    results['ensemble_stacking'] = evaluate_model(
        y_test, y_pred, y_pred_proba,
        'Ensemble (Stacking)',
        training_time
    )

    return voting_clf, stacking_clf, results


# ===========================
# EVALUATION HELPER
# ===========================

def evaluate_model(y_true, y_pred, y_pred_proba, model_name, training_time):
    """Evaluate model and return metrics"""

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    # ROC-AUC (one-vs-rest for multi-class)
    y_true_onehot = to_categorical(y_true, ModelConfig.N_CLASSES)
    roc_auc = roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovr', average='weighted')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None)

    # Print results
    print(f"\n  ✅ {model_name} Results:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Precision (weighted): {precision:.4f}")
    print(f"    Recall (weighted): {recall:.4f}")
    print(f"    F1-Score (weighted): {f1:.4f}")
    print(f"    ROC-AUC (OVR): {roc_auc:.4f}")
    print(f"    Training time: {training_time:.1f}s")

    print(f"\n    Per-Class Metrics:")
    for i, class_name in ModelConfig.CLASS_NAMES.items():
        print(f"      {class_name}:")
        print(f"        Precision: {precision_per_class[i]:.4f}")
        print(f"        Recall: {recall_per_class[i]:.4f}")
        print(f"        F1-Score: {f1_per_class[i]:.4f}")
        print(f"        Support: {support_per_class[i]}")

    # Return structured results
    return {
        'test_accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'roc_auc_ovr': roc_auc,
        'training_time_seconds': training_time,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            ModelConfig.CLASS_NAMES[i]: {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i],
                'support': int(support_per_class[i])
            }
            for i in range(ModelConfig.N_CLASSES)
        }
    }


# ===========================
# STEP 4: SAVE METADATA
# ===========================

def save_metadata(all_results, dataset_info, preprocessing_info):
    """Save comprehensive metadata"""

    print("\n" + "=" * 80)
    print("STEP 4: SAVING METADATA")
    print("=" * 80)

    metadata = MetadataManager.create_metadata(
        model_performances=all_results,
        dataset_info=dataset_info,
        preprocessing_info=preprocessing_info
    )

    MetadataManager.save_metadata(metadata)

    print(f"\n✅ Metadata saved successfully!")
    print(f"  Location: {ModelConfig.METADATA_PATH}")


# ===========================
# MAIN EXECUTION
# ===========================

def main():
    """Main execution pipeline"""

    print("\n" + "=" * 80)
    print("THREE-CLASS EXOPLANET DETECTION - COMPLETE TRAINING PIPELINE")
    print("=" * 80)

    # Step 1: Load and preprocess data
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data()

    # Step 2: Train base models
    genesis_model, xgb_model, rf_model, base_results = train_all_models(
        X_train, y_train, X_test, y_test
    )

    # Step 3: Train ensemble models
    voting_clf, stacking_clf, ensemble_results = train_ensemble_models(
        genesis_model, xgb_model, rf_model,
        X_train, y_train, X_test, y_test
    )

    # Combine results
    all_results = {**base_results, **ensemble_results}

    # Step 4: Save metadata
    dataset_info = {
        "name": "Kepler Q1-Q17 DR25 KOI",
        "total_samples": len(X_train) + len(X_test),
        "n_features": ModelConfig.N_FEATURES,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "validation_split": 0.0
    }

    preprocessing_info = {
        "smote_applied": True,
        "smote_config": {
            "sampling_strategy": "auto",
            "k_neighbors": 5,
            "random_state": ModelConfig.RANDOM_STATE
        },
        "feature_scaling": {
            "method": "StandardScaler",
            "scaler_path": str(ModelConfig.SCALER_PATH)
        },
        "train_test_split": {
            "test_size": 0.25,
            "random_state": 42,
            "stratify": True
        }
    }

    save_metadata(all_results, dataset_info, preprocessing_info)

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - MODEL COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'Time (s)':<12}")
    print("-" * 80)

    for model_name, metrics in all_results.items():
        print(f"{model_name:<25} "
              f"{metrics['test_accuracy']:<12.4f} "
              f"{metrics['f1_weighted']:<12.4f} "
              f"{metrics['roc_auc_ovr']:<12.4f} "
              f"{metrics['training_time_seconds']:<12.1f}")

    print("\n" + "=" * 80)
    print("All models saved successfully!")
    print(f"Model directory: {ModelConfig.MODEL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
