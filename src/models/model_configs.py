"""
Model Configuration Module for Three-Class Exoplanet Detection

This module provides configuration classes and factory functions for building
three machine learning models (Genesis CNN, XGBoost, RandomForest) and ensemble
methods for detecting exoplanets in Kepler light curve data.

Author: ML Architecture Team
Version: 1.0.0
Date: 2025-10-05
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout, Flatten, MaxPooling1D,
    AveragePooling1D, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# XGBoost
import xgboost as xgb

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# Utilities
import joblib


# ===========================
# GLOBAL CONFIGURATIONS
# ===========================

class ModelConfig:
    """Global model configuration constants"""

    # Data dimensions
    N_FEATURES = 784
    N_CLASSES = 3

    # Class names
    CLASS_NAMES = {
        0: "FALSE_POSITIVE",
        1: "CANDIDATE",
        2: "CONFIRMED"
    }

    # Random seed
    RANDOM_STATE = 42

    # Model directories
    MODEL_DIR = Path("models")
    METADATA_PATH = MODEL_DIR / "metadata.json"

    # File paths
    GENESIS_CNN_PATH = MODEL_DIR / "genesis_cnn_three_class.keras"
    XGBOOST_PATH = MODEL_DIR / "xgboost_three_class.json"
    RANDOM_FOREST_PATH = MODEL_DIR / "random_forest_three_class.pkl"
    ENSEMBLE_VOTING_PATH = MODEL_DIR / "ensemble_voting_three_class.pkl"
    ENSEMBLE_STACKING_PATH = MODEL_DIR / "ensemble_stacking_three_class.pkl"
    SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"

    @classmethod
    def ensure_model_dir(cls):
        """Create model directory if it doesn't exist"""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ===========================
# GENESIS CNN CONFIGURATION
# ===========================

class GenesisCNNConfig:
    """Configuration for Genesis CNN three-class model"""

    def __init__(self):
        self.input_shape = (ModelConfig.N_FEATURES, 1)
        self.n_classes = ModelConfig.N_CLASSES

        # Architecture parameters
        self.conv1_filters = 64
        self.conv1_kernel = 50
        self.conv2_filters = 128
        self.conv2_kernel = 12
        self.pool1_size = 16
        self.pool2_size = 8
        self.dense1_units = 256
        self.dense2_units = 128

        # Regularization
        self.dropout_rate_1 = 0.25
        self.dropout_rate_2 = 0.3
        self.dropout_rate_3 = 0.4
        self.dropout_rate_4 = 0.3

        # Training parameters
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.batch_size = 32
        self.epochs = 50

        # Callbacks
        self.early_stopping_patience = 7
        self.early_stopping_monitor = 'val_loss'
        self.reduce_lr_patience = 3
        self.reduce_lr_factor = 0.5
        self.reduce_lr_min_lr = 1e-6

    def get_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced dataset

        Args:
            y_train: Training labels (1D array of class indices)

        Returns:
            Dictionary mapping class index to weight
        """
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        return {i: weight for i, weight in enumerate(class_weights)}

    def build_model(self) -> keras.Model:
        """
        Build Genesis CNN architecture for three-class classification

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Feature Extraction Block 1
            Conv1D(
                self.conv1_filters,
                self.conv1_kernel,
                padding='same',
                activation='relu',
                input_shape=self.input_shape,
                name='conv1d_1'
            ),
            BatchNormalization(name='batch_norm_1'),
            Conv1D(
                self.conv1_filters,
                self.conv1_kernel,
                padding='same',
                activation='relu',
                name='conv1d_2'
            ),
            BatchNormalization(name='batch_norm_2'),
            MaxPooling1D(pool_size=self.pool1_size, name='max_pool_1'),
            Dropout(self.dropout_rate_1, name='dropout_1'),

            # Feature Extraction Block 2
            Conv1D(
                self.conv2_filters,
                self.conv2_kernel,
                padding='same',
                activation='relu',
                name='conv1d_3'
            ),
            BatchNormalization(name='batch_norm_3'),
            Conv1D(
                self.conv2_filters,
                self.conv2_kernel,
                padding='same',
                activation='relu',
                name='conv1d_4'
            ),
            BatchNormalization(name='batch_norm_4'),
            AveragePooling1D(pool_size=self.pool2_size, name='avg_pool_1'),
            Dropout(self.dropout_rate_2, name='dropout_2'),

            # Flatten
            Flatten(name='flatten'),

            # Classification Head
            Dense(self.dense1_units, activation='relu', name='dense_1'),
            BatchNormalization(name='batch_norm_5'),
            Dropout(self.dropout_rate_3, name='dropout_3'),

            Dense(self.dense2_units, activation='relu', name='dense_2'),
            BatchNormalization(name='batch_norm_6'),
            Dropout(self.dropout_rate_4, name='dropout_4'),

            # Output layer (3 classes)
            Dense(self.n_classes, activation='softmax', name='output')
        ], name='GenesisExoplanetCNN')

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.loss,
            metrics=self.metrics
        )

        return model

    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        """
        Get training callbacks

        Returns:
            List of Keras callbacks
        """
        early_stop = EarlyStopping(
            monitor=self.early_stopping_monitor,
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor=self.early_stopping_monitor,
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.reduce_lr_min_lr,
            verbose=1
        )

        return [early_stop, reduce_lr]

    def save_model(self, model: keras.Model):
        """Save model to disk"""
        ModelConfig.ensure_model_dir()
        model.save(ModelConfig.GENESIS_CNN_PATH)
        print(f"✅ Genesis CNN saved to {ModelConfig.GENESIS_CNN_PATH}")

    @staticmethod
    def load_model() -> keras.Model:
        """Load saved model from disk"""
        if not ModelConfig.GENESIS_CNN_PATH.exists():
            raise FileNotFoundError(f"Model not found: {ModelConfig.GENESIS_CNN_PATH}")

        model = load_model(ModelConfig.GENESIS_CNN_PATH)
        print(f"✅ Genesis CNN loaded from {ModelConfig.GENESIS_CNN_PATH}")
        return model


# ===========================
# XGBOOST CONFIGURATION
# ===========================

class XGBoostConfig:
    """Configuration for XGBoost three-class model"""

    def __init__(self):
        # Model parameters
        self.objective = 'multi:softprob'
        self.num_class = ModelConfig.N_CLASSES
        self.n_estimators = 200
        self.max_depth = 7
        self.learning_rate = 0.05
        self.subsample = 0.8
        self.colsample_bytree = 0.8
        self.colsample_bylevel = 0.8
        self.min_child_weight = 3
        self.gamma = 0.1
        self.reg_alpha = 0.05  # L1 regularization
        self.reg_lambda = 1.0  # L2 regularization
        self.random_state = ModelConfig.RANDOM_STATE

        # GPU acceleration (set to None if GPU not available)
        self.tree_method = 'gpu_hist'  # Use 'hist' for CPU
        self.gpu_id = 0
        self.predictor = 'gpu_predictor'  # Use 'cpu_predictor' for CPU

    def get_sample_weights(self, y_train: np.ndarray) -> np.ndarray:
        """
        Compute sample weights for imbalanced dataset

        Args:
            y_train: Training labels (1D array of class indices)

        Returns:
            Array of sample weights
        """
        return compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )

    def build_model(self) -> xgb.XGBClassifier:
        """
        Build XGBoost classifier for three-class classification

        Returns:
            XGBoost classifier instance
        """
        try:
            # Try GPU configuration
            model = xgb.XGBClassifier(
                objective=self.objective,
                num_class=self.num_class,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                colsample_bylevel=self.colsample_bylevel,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                tree_method=self.tree_method,
                gpu_id=self.gpu_id,
                predictor=self.predictor,
                eval_metric='mlogloss'
            )
        except Exception as e:
            print(f"⚠️ GPU not available, falling back to CPU: {e}")
            # Fallback to CPU configuration
            model = xgb.XGBClassifier(
                objective=self.objective,
                num_class=self.num_class,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                colsample_bylevel=self.colsample_bylevel,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                tree_method='hist',
                eval_metric='mlogloss'
            )

        return model

    def save_model(self, model: xgb.XGBClassifier):
        """Save model to disk (JSON format)"""
        ModelConfig.ensure_model_dir()
        model.save_model(str(ModelConfig.XGBOOST_PATH))
        print(f"✅ XGBoost saved to {ModelConfig.XGBOOST_PATH}")

    @staticmethod
    def load_model() -> xgb.XGBClassifier:
        """Load saved model from disk"""
        if not ModelConfig.XGBOOST_PATH.exists():
            raise FileNotFoundError(f"Model not found: {ModelConfig.XGBOOST_PATH}")

        model = xgb.XGBClassifier()
        model.load_model(str(ModelConfig.XGBOOST_PATH))
        print(f"✅ XGBoost loaded from {ModelConfig.XGBOOST_PATH}")
        return model


# ===========================
# RANDOM FOREST CONFIGURATION
# ===========================

class RandomForestConfig:
    """Configuration for RandomForest three-class model"""

    def __init__(self):
        # Model parameters
        self.n_estimators = 300
        self.max_depth = 15
        self.min_samples_split = 5
        self.min_samples_leaf = 2
        self.max_features = 'sqrt'  # sqrt(784) ≈ 28
        self.bootstrap = True
        self.oob_score = True  # Out-of-bag evaluation
        self.class_weight = 'balanced'  # Automatic class balancing
        self.random_state = ModelConfig.RANDOM_STATE
        self.n_jobs = -1  # Use all CPU cores
        self.verbose = 0
        self.criterion = 'gini'

    def build_model(self) -> RandomForestClassifier:
        """
        Build RandomForest classifier for three-class classification

        Returns:
            RandomForest classifier instance
        """
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            criterion=self.criterion
        )

        return model

    def save_model(self, model: RandomForestClassifier):
        """Save model to disk (pickle format)"""
        ModelConfig.ensure_model_dir()
        joblib.dump(model, ModelConfig.RANDOM_FOREST_PATH)
        print(f"✅ RandomForest saved to {ModelConfig.RANDOM_FOREST_PATH}")

    @staticmethod
    def load_model() -> RandomForestClassifier:
        """Load saved model from disk"""
        if not ModelConfig.RANDOM_FOREST_PATH.exists():
            raise FileNotFoundError(f"Model not found: {ModelConfig.RANDOM_FOREST_PATH}")

        model = joblib.load(ModelConfig.RANDOM_FOREST_PATH)
        print(f"✅ RandomForest loaded from {ModelConfig.RANDOM_FOREST_PATH}")
        return model


# ===========================
# ENSEMBLE CONFIGURATION
# ===========================

class EnsembleConfig:
    """Configuration for ensemble models"""

    def __init__(self):
        self.random_state = ModelConfig.RANDOM_STATE

        # Voting weights (CNN gets higher weight due to better performance)
        self.voting_weights = [2, 1, 1]  # [CNN, XGBoost, RF]

        # Stacking meta-learner parameters
        self.meta_learner_max_iter = 1000
        self.meta_learner_solver = 'lbfgs'
        self.meta_learner_multi_class = 'multinomial'
        self.stacking_cv = 5

    def build_voting_classifier(
        self,
        genesis_model,
        xgb_model,
        rf_model
    ) -> VotingClassifier:
        """
        Build soft voting classifier ensemble

        Args:
            genesis_model: Wrapped Keras model (scikit-learn compatible)
            xgb_model: XGBoost classifier
            rf_model: RandomForest classifier

        Returns:
            VotingClassifier instance
        """
        voting_clf = VotingClassifier(
            estimators=[
                ('genesis_cnn', genesis_model),
                ('xgboost', xgb_model),
                ('random_forest', rf_model)
            ],
            voting='soft',
            weights=self.voting_weights,
            n_jobs=-1
        )

        return voting_clf

    def build_stacking_classifier(
        self,
        genesis_model,
        xgb_model,
        rf_model
    ) -> StackingClassifier:
        """
        Build stacking classifier ensemble

        Args:
            genesis_model: Wrapped Keras model (scikit-learn compatible)
            xgb_model: XGBoost classifier
            rf_model: RandomForest classifier

        Returns:
            StackingClassifier instance
        """
        # Meta-learner: Logistic Regression
        meta_learner = LogisticRegression(
            multi_class=self.meta_learner_multi_class,
            solver=self.meta_learner_solver,
            class_weight='balanced',
            max_iter=self.meta_learner_max_iter,
            random_state=self.random_state
        )

        stacking_clf = StackingClassifier(
            estimators=[
                ('genesis_cnn', genesis_model),
                ('xgboost', xgb_model),
                ('random_forest', rf_model)
            ],
            final_estimator=meta_learner,
            cv=self.stacking_cv,
            stack_method='predict_proba',
            n_jobs=-1
        )

        return stacking_clf

    def save_voting_model(self, model: VotingClassifier):
        """Save voting classifier to disk"""
        ModelConfig.ensure_model_dir()
        joblib.dump(model, ModelConfig.ENSEMBLE_VOTING_PATH)
        print(f"✅ Voting Ensemble saved to {ModelConfig.ENSEMBLE_VOTING_PATH}")

    def save_stacking_model(self, model: StackingClassifier):
        """Save stacking classifier to disk"""
        ModelConfig.ensure_model_dir()
        joblib.dump(model, ModelConfig.ENSEMBLE_STACKING_PATH)
        print(f"✅ Stacking Ensemble saved to {ModelConfig.ENSEMBLE_STACKING_PATH}")

    @staticmethod
    def load_voting_model() -> VotingClassifier:
        """Load voting ensemble from disk"""
        if not ModelConfig.ENSEMBLE_VOTING_PATH.exists():
            raise FileNotFoundError(f"Model not found: {ModelConfig.ENSEMBLE_VOTING_PATH}")

        model = joblib.load(ModelConfig.ENSEMBLE_VOTING_PATH)
        print(f"✅ Voting Ensemble loaded from {ModelConfig.ENSEMBLE_VOTING_PATH}")
        return model

    @staticmethod
    def load_stacking_model() -> StackingClassifier:
        """Load stacking ensemble from disk"""
        if not ModelConfig.ENSEMBLE_STACKING_PATH.exists():
            raise FileNotFoundError(f"Model not found: {ModelConfig.ENSEMBLE_STACKING_PATH}")

        model = joblib.load(ModelConfig.ENSEMBLE_STACKING_PATH)
        print(f"✅ Stacking Ensemble loaded from {ModelConfig.ENSEMBLE_STACKING_PATH}")
        return model


# ===========================
# KERAS WRAPPER (for ensemble compatibility)
# ===========================

from sklearn.base import BaseEstimator, ClassifierMixin

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for Keras model
    Allows Keras model to be used in ensemble methods
    """

    def __init__(self, model: keras.Model = None):
        self.model = model
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y, **kwargs):
        """Fit method (model is already trained)"""
        return self

    def predict(self, X):
        """Predict class labels"""
        # Reshape for CNN input
        if len(X.shape) == 2:
            X_reshaped = X.reshape(-1, X.shape[1], 1)
        else:
            X_reshaped = X

        probas = self.model.predict(X_reshaped, verbose=0)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        """Predict class probabilities"""
        # Reshape for CNN input
        if len(X.shape) == 2:
            X_reshaped = X.reshape(-1, X.shape[1], 1)
        else:
            X_reshaped = X

        return self.model.predict(X_reshaped, verbose=0)


# ===========================
# METADATA MANAGEMENT
# ===========================

class MetadataManager:
    """Manage model metadata and performance metrics"""

    @staticmethod
    def create_metadata(
        model_performances: Dict[str, Dict[str, Any]],
        dataset_info: Dict[str, Any],
        preprocessing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata JSON

        Args:
            model_performances: Dictionary of model performance metrics
            dataset_info: Dataset information
            preprocessing_info: Preprocessing configuration

        Returns:
            Metadata dictionary
        """
        metadata = {
            "model_metadata": {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "framework_versions": {
                    "tensorflow": tf.__version__,
                    "keras": keras.__version__,
                    "xgboost": xgb.__version__,
                    "sklearn": "1.5.2",  # Update if needed
                    "python": "3.11.9"
                },
                "dataset_info": dataset_info
            },

            "label_mapping": {
                "class_names": ModelConfig.CLASS_NAMES,
                "encoding": {
                    "type": "one_hot",
                    "shape": [ModelConfig.N_CLASSES]
                }
            },

            "model_performance": model_performances,

            "preprocessing": preprocessing_info,

            "deployment": {
                "model_files": {
                    "genesis_cnn": str(ModelConfig.GENESIS_CNN_PATH),
                    "xgboost": str(ModelConfig.XGBOOST_PATH),
                    "random_forest": str(ModelConfig.RANDOM_FOREST_PATH),
                    "ensemble_voting": str(ModelConfig.ENSEMBLE_VOTING_PATH),
                    "ensemble_stacking": str(ModelConfig.ENSEMBLE_STACKING_PATH),
                    "scaler": str(ModelConfig.SCALER_PATH)
                },
                "api_endpoint": "/predict/exoplanet",
                "input_format": {
                    "type": "json",
                    "schema": {
                        "features": "array of 784 floats"
                    }
                }
            }
        }

        return metadata

    @staticmethod
    def save_metadata(metadata: Dict[str, Any]):
        """Save metadata to JSON file"""
        ModelConfig.ensure_model_dir()

        with open(ModelConfig.METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved to {ModelConfig.METADATA_PATH}")

    @staticmethod
    def load_metadata() -> Dict[str, Any]:
        """Load metadata from JSON file"""
        if not ModelConfig.METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata not found: {ModelConfig.METADATA_PATH}")

        with open(ModelConfig.METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        print(f"✅ Metadata loaded from {ModelConfig.METADATA_PATH}")
        return metadata


# ===========================
# FACTORY FUNCTIONS
# ===========================

def create_all_models() -> Dict[str, Any]:
    """
    Factory function to create all models

    Returns:
        Dictionary containing all model instances and configs
    """
    # Initialize configs
    cnn_config = GenesisCNNConfig()
    xgb_config = XGBoostConfig()
    rf_config = RandomForestConfig()
    ensemble_config = EnsembleConfig()

    # Build models
    genesis_model = cnn_config.build_model()
    xgb_model = xgb_config.build_model()
    rf_model = rf_config.build_model()

    print("\n✅ All models created successfully!")
    print(f"  - Genesis CNN: {genesis_model.count_params():,} parameters")
    print(f"  - XGBoost: {xgb_model.n_estimators} estimators")
    print(f"  - RandomForest: {rf_model.n_estimators} estimators")

    return {
        'models': {
            'genesis_cnn': genesis_model,
            'xgboost': xgb_model,
            'random_forest': rf_model
        },
        'configs': {
            'genesis_cnn': cnn_config,
            'xgboost': xgb_config,
            'random_forest': rf_config,
            'ensemble': ensemble_config
        }
    }


def load_all_models() -> Dict[str, Any]:
    """
    Load all saved models from disk

    Returns:
        Dictionary containing all loaded models
    """
    models = {
        'genesis_cnn': GenesisCNNConfig.load_model(),
        'xgboost': XGBoostConfig.load_model(),
        'random_forest': RandomForestConfig.load_model(),
        'ensemble_voting': EnsembleConfig.load_voting_model(),
        'ensemble_stacking': EnsembleConfig.load_stacking_model()
    }

    print("\n✅ All models loaded successfully!")
    return models


# ===========================
# MAIN USAGE EXAMPLE
# ===========================

if __name__ == "__main__":
    print("=" * 80)
    print("Three-Class Exoplanet Detection - Model Configuration Module")
    print("=" * 80)

    # Create all models
    print("\n[1/3] Creating models...")
    model_dict = create_all_models()

    # Print Genesis CNN summary
    print("\n[2/3] Genesis CNN Architecture:")
    print("-" * 80)
    model_dict['models']['genesis_cnn'].summary()

    # Print configuration details
    print("\n[3/3] Configuration Summary:")
    print("-" * 80)
    print(f"Input shape: {ModelConfig.N_FEATURES} features")
    print(f"Output classes: {ModelConfig.N_CLASSES}")
    print(f"Class names: {list(ModelConfig.CLASS_NAMES.values())}")
    print(f"Model directory: {ModelConfig.MODEL_DIR}")

    print("\n" + "=" * 80)
    print("Configuration module ready for use!")
    print("=" * 80)
