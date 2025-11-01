"""
Tests for Ensemble Learning Module
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.ensemble import (
    EnsembleModel,
    select_best_model,
    create_ensemble_report
)


@pytest.fixture
def sample_data():
    """Create sample classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_models(sample_data):
    """Create and train multiple models."""
    X_train, X_test, y_train, y_test = sample_data

    models = {
        'rf': RandomForestClassifier(n_estimators=50, random_state=42),
        'lr': LogisticRegression(max_iter=1000, random_state=42),
        'dt': DecisionTreeClassifier(random_state=42)
    }

    # Train all models
    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


class TestEnsembleModel:
    """Test EnsembleModel class."""

    def test_voting_classifier(self, trained_models, sample_data):
        """Test VotingClassifier creation and training."""
        X_train, X_test, y_train, y_test = sample_data

        ensemble = EnsembleModel(
            models=trained_models,
            ensemble_type='voting',
            voting='soft'
        )

        ensemble.fit(X_train, y_train)

        assert ensemble.ensemble is not None
        assert ensemble.ensemble_type == 'voting'
        assert ensemble.voting == 'soft'

    def test_stacking_classifier(self, trained_models, sample_data):
        """Test StackingClassifier creation and training."""
        X_train, X_test, y_train, y_test = sample_data

        ensemble = EnsembleModel(
            models=trained_models,
            ensemble_type='stacking',
            cv=3
        )

        ensemble.fit(X_train, y_train)

        assert ensemble.ensemble is not None
        assert ensemble.ensemble_type == 'stacking'
        assert ensemble.meta_learner is not None

    def test_predict(self, trained_models, sample_data):
        """Test prediction method."""
        X_train, X_test, y_train, y_test = sample_data

        ensemble = EnsembleModel(
            models=trained_models,
            ensemble_type='voting'
        )
        ensemble.fit(X_train, y_train)

        predictions = ensemble.predict(X_test)

        assert predictions.shape[0] == X_test.shape[0]
        assert len(np.unique(predictions)) <= 3  # 3 classes

    def test_predict_proba(self, trained_models, sample_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = sample_data

        ensemble = EnsembleModel(
            models=trained_models,
            ensemble_type='voting',
            voting='soft'
        )
        ensemble.fit(X_train, y_train)

        probabilities = ensemble.predict_proba(X_test)

        assert probabilities.shape[0] == X_test.shape[0]
        assert probabilities.shape[1] == 3  # 3 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_evaluate_individual_models(self, trained_models, sample_data):
        """Test individual model evaluation."""
        X_train, X_test, y_train, y_test = sample_data

        ensemble = EnsembleModel(models=trained_models)
        ensemble.fit(X_train, y_train)

        scores = ensemble.evaluate_individual_models(X_test, y_test)

        assert len(scores) == len(trained_models)
        for name, metrics in scores.items():
            assert 'accuracy' in metrics
            assert 'f1' in metrics
            assert metrics['accuracy'] >= 0 and metrics['accuracy'] <= 1

    def test_evaluate_ensemble(self, trained_models, sample_data):
        """Test ensemble evaluation."""
        X_train, X_test, y_train, y_test = sample_data

        ensemble = EnsembleModel(models=trained_models)
        ensemble.fit(X_train, y_train)

        metrics = ensemble.evaluate_ensemble(X_test, y_test)

        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert metrics['accuracy'] >= 0 and metrics['accuracy'] <= 1

    def test_save_load(self, trained_models, sample_data, tmp_path):
        """Test model save and load."""
        X_train, X_test, y_train, y_test = sample_data

        ensemble = EnsembleModel(models=trained_models)
        ensemble.fit(X_train, y_train)

        # Save
        model_path = tmp_path / "ensemble_model.pkl"
        ensemble.save(model_path)

        assert model_path.exists()

        # Load
        loaded_ensemble = EnsembleModel.load(model_path)

        # Test predictions are same
        original_pred = ensemble.predict(X_test)
        loaded_pred = loaded_ensemble.predict(X_test)

        assert np.array_equal(original_pred, loaded_pred)


class TestModelSelection:
    """Test model selection utilities."""

    def test_select_best_model(self, trained_models, sample_data):
        """Test automatic model selection."""
        X_train, X_test, y_train, y_test = sample_data

        best_name, best_model, metrics = select_best_model(
            trained_models,
            X_test,
            y_test,
            metric='f1',
            include_ensemble=False  # Skip ensemble for speed
        )

        assert best_name in trained_models.keys()
        assert best_model is not None
        assert 'all_scores' in metrics
        assert 'best_scores' in metrics

    def test_select_best_model_with_ensemble(self, trained_models, sample_data):
        """Test model selection including ensemble."""
        X_train, X_test, y_train, y_test = sample_data

        # Fit models first for ensemble
        for model in trained_models.values():
            if not hasattr(model, 'classes_'):
                model.fit(X_train, y_train)

        best_name, best_model, metrics = select_best_model(
            trained_models,
            X_test,
            y_test,
            metric='accuracy',
            include_ensemble=True,
            ensemble_types=['voting']
        )

        assert best_name is not None
        assert 'all_scores' in metrics
        assert len(metrics['all_scores']) >= len(trained_models)


class TestEnsembleReport:
    """Test ensemble reporting."""

    def test_create_ensemble_report(self, trained_models, sample_data):
        """Test ensemble report generation."""
        X_train, X_test, y_train, y_test = sample_data

        ensemble = EnsembleModel(models=trained_models)
        ensemble.fit(X_train, y_train)

        report = create_ensemble_report(
            ensemble,
            X_test,
            y_test,
            class_names=['Class A', 'Class B', 'Class C']
        )

        assert 'ensemble_type' in report
        assert 'individual_models' in report
        assert 'ensemble' in report
        assert 'confusion_matrix' in report
        assert 'recommendation' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
