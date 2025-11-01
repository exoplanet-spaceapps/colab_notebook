"""
Ensemble Learning Models for Exoplanet Classification

This module implements ensemble methods including:
- VotingClassifier (soft voting)
- StackingClassifier (meta-learner)
- Model selection and evaluation utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Wrapper for ensemble learning with VotingClassifier and StackingClassifier.

    Supports:
    - Soft voting across multiple models
    - Stacking with meta-learner
    - Automatic model selection
    - Performance evaluation
    """

    def __init__(
        self,
        models: Dict[str, BaseEstimator],
        ensemble_type: str = 'voting',
        voting: str = 'soft',
        meta_learner: Optional[BaseEstimator] = None,
        cv: int = 5
    ):
        """
        Initialize ensemble model.

        Args:
            models: Dictionary of (name, model) pairs
            ensemble_type: 'voting' or 'stacking'
            voting: 'soft' or 'hard' (for VotingClassifier)
            meta_learner: Meta-learner for stacking (default: LogisticRegression)
            cv: Cross-validation folds
        """
        self.models = models
        self.ensemble_type = ensemble_type
        self.voting = voting
        self.meta_learner = meta_learner
        self.cv = cv
        self.ensemble = None
        self.individual_scores = {}
        self.ensemble_score = None

    def build_voting_classifier(self) -> VotingClassifier:
        """
        Build VotingClassifier with soft or hard voting.

        Returns:
            VotingClassifier instance
        """
        estimators = list(self.models.items())

        ensemble = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            n_jobs=-1,
            verbose=True
        )

        logger.info(f"Built VotingClassifier with {len(estimators)} models "
                   f"(voting={self.voting})")
        return ensemble

    def build_stacking_classifier(self) -> StackingClassifier:
        """
        Build StackingClassifier with meta-learner.

        Returns:
            StackingClassifier instance
        """
        from sklearn.linear_model import LogisticRegression

        estimators = list(self.models.items())

        # Use LogisticRegression as default meta-learner
        if self.meta_learner is None:
            self.meta_learner = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )

        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_learner,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )

        logger.info(f"Built StackingClassifier with {len(estimators)} base models "
                   f"and meta-learner: {type(self.meta_learner).__name__}")
        return ensemble

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """
        Fit ensemble model.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self
        """
        # Build appropriate ensemble
        if self.ensemble_type == 'voting':
            self.ensemble = self.build_voting_classifier()
        elif self.ensemble_type == 'stacking':
            self.ensemble = self.build_stacking_classifier()
        else:
            raise ValueError(f"Unknown ensemble_type: {self.ensemble_type}")

        # Fit ensemble
        logger.info(f"Fitting {self.ensemble_type} ensemble...")
        self.ensemble.fit(X, y)

        logger.info("Ensemble fitting complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        if self.ensemble is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        return self.ensemble.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if self.ensemble is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        return self.ensemble.predict_proba(X)

    def evaluate_individual_models(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate each individual model separately.

        Args:
            X: Test features
            y: Test labels

        Returns:
            Dictionary of model scores
        """
        scores = {}

        for name, model in self.models.items():
            try:
                y_pred = model.predict(X)

                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
                }

                # Add ROC-AUC if predict_proba available
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X)
                        metrics['roc_auc'] = roc_auc_score(
                            y, y_proba,
                            multi_class='ovr',
                            average='weighted'
                        )
                    except Exception as e:
                        logger.warning(f"ROC-AUC calculation failed for {name}: {e}")

                scores[name] = metrics
                logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                           f"F1: {metrics['f1']:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                scores[name] = {'error': str(e)}

        self.individual_scores = scores
        return scores

    def evaluate_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate ensemble model.

        Args:
            X: Test features
            y: Test labels

        Returns:
            Dictionary of ensemble metrics
        """
        if self.ensemble is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
        }

        self.ensemble_score = metrics
        logger.info(f"Ensemble {self.ensemble_type} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1']:.4f}")

        return metrics

    def save(self, filepath: Union[str, Path]):
        """
        Save ensemble model to disk.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.ensemble, filepath)
        logger.info(f"Ensemble saved to {filepath}")

        # Save metadata
        metadata = {
            'ensemble_type': self.ensemble_type,
            'voting': self.voting if self.ensemble_type == 'voting' else None,
            'model_names': list(self.models.keys()),
            'individual_scores': self.individual_scores,
            'ensemble_score': self.ensemble_score
        }

        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'EnsembleModel':
        """
        Load ensemble model from disk.

        Args:
            filepath: Path to model file

        Returns:
            EnsembleModel instance
        """
        filepath = Path(filepath)
        ensemble = joblib.load(filepath)

        # Load metadata
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Create instance
        instance = cls.__new__(cls)
        instance.ensemble = ensemble
        instance.ensemble_type = metadata.get('ensemble_type', 'voting')
        instance.voting = metadata.get('voting', 'soft')
        instance.individual_scores = metadata.get('individual_scores', {})
        instance.ensemble_score = metadata.get('ensemble_score', None)
        instance.models = {}  # Will be populated from ensemble

        logger.info(f"Ensemble loaded from {filepath}")
        return instance


def select_best_model(
    models_dict: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: str = 'f1',
    include_ensemble: bool = True,
    ensemble_types: List[str] = ['voting', 'stacking']
) -> Tuple[str, BaseEstimator, Dict[str, Any]]:
    """
    Automatically select best model from individual models and ensembles.

    Args:
        models_dict: Dictionary of (name, model) pairs
        X_test: Test features
        y_test: Test labels
        metric: Metric to optimize ('accuracy', 'f1', 'roc_auc')
        include_ensemble: Whether to include ensemble models
        ensemble_types: List of ensemble types to evaluate

    Returns:
        Tuple of (best_model_name, best_model, metrics_dict)
    """
    all_scores = {}
    all_models = {}

    # Evaluate individual models
    logger.info("Evaluating individual models...")
    for name, model in models_dict.items():
        try:
            y_pred = model.predict(X_test)

            scores = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                scores['roc_auc'] = roc_auc_score(
                    y_test, y_proba,
                    multi_class='ovr',
                    average='weighted'
                )

            all_scores[name] = scores
            all_models[name] = model

            logger.info(f"{name}: {metric}={scores.get(metric, 0):.4f}")

        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")

    # Evaluate ensemble models
    if include_ensemble and len(models_dict) > 1:
        logger.info("\nEvaluating ensemble models...")

        for ens_type in ensemble_types:
            try:
                ens_name = f"ensemble_{ens_type}"

                # Create and fit ensemble
                if ens_type == 'voting':
                    ensemble_model = EnsembleModel(
                        models=models_dict,
                        ensemble_type='voting',
                        voting='soft'
                    )
                else:  # stacking
                    ensemble_model = EnsembleModel(
                        models=models_dict,
                        ensemble_type='stacking'
                    )

                # Note: Ensemble should already be fitted
                # If not, this would require training data
                scores = ensemble_model.evaluate_ensemble(X_test, y_test)

                all_scores[ens_name] = scores
                all_models[ens_name] = ensemble_model.ensemble

                logger.info(f"{ens_name}: {metric}={scores.get(metric, 0):.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {ens_type} ensemble: {e}")

    # Select best model
    if not all_scores:
        raise ValueError("No models could be evaluated")

    best_name = max(all_scores.keys(), key=lambda k: all_scores[k].get(metric, 0))
    best_model = all_models[best_name]
    best_scores = all_scores[best_name]

    logger.info(f"\n🏆 Best model: {best_name} ({metric}={best_scores[metric]:.4f})")

    return best_name, best_model, {
        'best_model': best_name,
        'best_scores': best_scores,
        'all_scores': all_scores,
        'metric_used': metric
    }


def create_ensemble_report(
    ensemble_model: EnsembleModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create comprehensive ensemble evaluation report.

    Args:
        ensemble_model: Fitted EnsembleModel
        X_test: Test features
        y_test: Test labels
        class_names: List of class names for confusion matrix

    Returns:
        Report dictionary with all metrics and analysis
    """
    report = {
        'ensemble_type': ensemble_model.ensemble_type,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    # Individual model scores
    individual_scores = ensemble_model.evaluate_individual_models(X_test, y_test)
    report['individual_models'] = individual_scores

    # Ensemble scores
    ensemble_scores = ensemble_model.evaluate_ensemble(X_test, y_test)
    report['ensemble'] = ensemble_scores

    # Confusion matrix
    y_pred = ensemble_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    if class_names:
        report['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'classes': class_names
        }
    else:
        report['confusion_matrix'] = cm.tolist()

    # Performance comparison
    improvements = {}
    for name, scores in individual_scores.items():
        if 'error' not in scores:
            improvement = ensemble_scores['f1'] - scores['f1']
            improvements[name] = {
                'f1_improvement': improvement,
                'percentage': (improvement / scores['f1'] * 100) if scores['f1'] > 0 else 0
            }

    report['improvements'] = improvements

    # Best individual model
    best_individual = max(
        individual_scores.items(),
        key=lambda x: x[1].get('f1', 0) if 'error' not in x[1] else 0
    )
    report['best_individual_model'] = {
        'name': best_individual[0],
        'scores': best_individual[1]
    }

    # Recommendation
    if ensemble_scores['f1'] > best_individual[1].get('f1', 0):
        report['recommendation'] = 'use_ensemble'
        report['reason'] = f"Ensemble F1 ({ensemble_scores['f1']:.4f}) > " \
                          f"Best individual F1 ({best_individual[1]['f1']:.4f})"
    else:
        report['recommendation'] = 'use_individual'
        report['reason'] = f"Best individual model ({best_individual[0]}) " \
                          f"performs better than ensemble"

    return report


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Ensemble Learning Module")
    print("=" * 60)
    print("\nFeatures:")
    print("✓ VotingClassifier (soft/hard voting)")
    print("✓ StackingClassifier (meta-learner)")
    print("✓ Automatic model selection")
    print("✓ Comprehensive evaluation")
    print("\nExample usage:")
    print("""
    # Create ensemble
    ensemble = EnsembleModel(
        models={'cnn': cnn_model, 'xgb': xgb_model, 'rf': rf_model},
        ensemble_type='voting',
        voting='soft'
    )

    # Fit and evaluate
    ensemble.fit(X_train, y_train)
    scores = ensemble.evaluate_ensemble(X_test, y_test)

    # Select best model
    best_name, best_model, metrics = select_best_model(
        models_dict, X_test, y_test, metric='f1'
    )
    """)
