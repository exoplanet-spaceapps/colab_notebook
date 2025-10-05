"""
Inference Interface for Exoplanet Classification

Provides unified prediction interface with:
- Probability predictions
- Class name mapping
- Metadata integration
- Batch processing
- Result formatting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json
import logging
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Unified inference interface for exoplanet classification models.

    Features:
    - Probability and class predictions
    - Metadata integration (class names, feature names)
    - Batch processing
    - Result formatting
    - Confidence thresholds
    """

    def __init__(
        self,
        model: BaseEstimator,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize inference engine.

        Args:
            model: Trained classifier
            metadata: Dictionary with class_names, feature_names, etc.
            metadata_path: Path to metadata JSON file
        """
        self.model = model
        self.metadata = metadata or {}

        # Load metadata from file if provided
        if metadata_path:
            self.load_metadata(metadata_path)

        # Extract class names
        self.class_names = self.metadata.get('class_names', None)
        if self.class_names is None and hasattr(model, 'classes_'):
            self.class_names = model.classes_.tolist()

        # Extract feature names
        self.feature_names = self.metadata.get('feature_names', None)

        logger.info(f"InferenceEngine initialized with model: {type(model).__name__}")
        if self.class_names:
            logger.info(f"Classes: {self.class_names}")

    def load_metadata(self, filepath: Union[str, Path]):
        """
        Load metadata from JSON file.

        Args:
            filepath: Path to metadata file
        """
        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"Metadata file not found: {filepath}")
            return

        with open(filepath, 'r') as f:
            self.metadata = json.load(f)

        logger.info(f"Metadata loaded from {filepath}")

    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = False,
        threshold: Optional[float] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on input data.

        Args:
            X: Input features (n_samples, n_features)
            return_proba: Whether to return probabilities
            threshold: Confidence threshold (if None, use argmax)

        Returns:
            Predictions (and probabilities if return_proba=True)
        """
        if return_proba:
            proba = self.model.predict_proba(X)

            if threshold is not None:
                # Apply confidence threshold
                max_proba = np.max(proba, axis=1)
                pred = np.argmax(proba, axis=1)
                pred[max_proba < threshold] = -1  # Uncertain predictions
            else:
                pred = np.argmax(proba, axis=1)

            return pred, proba
        else:
            return self.model.predict(X)

    def predict_with_metadata(
        self,
        X: np.ndarray,
        return_proba: bool = True,
        return_class_names: bool = True,
        threshold: Optional[float] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Predict with rich metadata and formatting.

        Args:
            X: Input features
            return_proba: Include probability predictions
            return_class_names: Map class indices to names
            threshold: Confidence threshold
            top_k: Number of top predictions to return per sample

        Returns:
            Dictionary with predictions, probabilities, and metadata
        """
        results = {
            'n_samples': X.shape[0],
            'predictions': []
        }

        # Get predictions and probabilities
        if return_proba:
            pred_indices, proba = self.predict(X, return_proba=True, threshold=threshold)
        else:
            pred_indices = self.predict(X)
            proba = None

        # Process each sample
        for i in range(X.shape[0]):
            sample_result = {
                'sample_index': i,
                'predicted_class_index': int(pred_indices[i])
            }

            # Add class name
            if return_class_names and self.class_names:
                if pred_indices[i] == -1:
                    sample_result['predicted_class_name'] = 'UNCERTAIN'
                else:
                    sample_result['predicted_class_name'] = self.class_names[pred_indices[i]]

            # Add probabilities
            if proba is not None:
                sample_proba = proba[i]
                sample_result['confidence'] = float(np.max(sample_proba))

                # Top-k predictions
                top_k_indices = np.argsort(sample_proba)[::-1][:top_k]
                top_k_probs = sample_proba[top_k_indices]

                top_k_results = []
                for idx, prob in zip(top_k_indices, top_k_probs):
                    entry = {
                        'class_index': int(idx),
                        'probability': float(prob)
                    }
                    if self.class_names:
                        entry['class_name'] = self.class_names[idx]
                    top_k_results.append(entry)

                sample_result['top_k_predictions'] = top_k_results
                sample_result['probabilities'] = sample_proba.tolist()

            results['predictions'].append(sample_result)

        # Add summary statistics
        if proba is not None:
            confidences = [p['confidence'] for p in results['predictions']]
            results['summary'] = {
                'mean_confidence': float(np.mean(confidences)),
                'min_confidence': float(np.min(confidences)),
                'max_confidence': float(np.max(confidences)),
                'std_confidence': float(np.std(confidences))
            }

            if threshold is not None:
                n_uncertain = np.sum(pred_indices == -1)
                results['summary']['n_uncertain'] = int(n_uncertain)
                results['summary']['uncertainty_rate'] = float(n_uncertain / X.shape[0])

        # Add class distribution
        unique, counts = np.unique(pred_indices[pred_indices != -1], return_counts=True)
        class_dist = []
        for cls, count in zip(unique, counts):
            entry = {
                'class_index': int(cls),
                'count': int(count),
                'proportion': float(count / X.shape[0])
            }
            if self.class_names:
                entry['class_name'] = self.class_names[cls]
            class_dist.append(entry)

        results['class_distribution'] = class_dist

        # Add metadata
        results['metadata'] = {
            'model_type': type(self.model).__name__,
            'n_classes': len(self.class_names) if self.class_names else None,
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'threshold': threshold
        }

        return results

    def predict_single(
        self,
        X: np.ndarray,
        return_proba: bool = True,
        return_class_names: bool = True,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Predict on a single sample with detailed output.

        Args:
            X: Single sample features (1D array)
            return_proba: Include probabilities
            return_class_names: Include class names
            top_k: Number of top predictions

        Returns:
            Prediction dictionary for single sample
        """
        # Reshape if needed
        if X.ndim == 1:
            X = X.reshape(1, -1)

        results = self.predict_with_metadata(
            X,
            return_proba=return_proba,
            return_class_names=return_class_names,
            top_k=top_k
        )

        return results['predictions'][0]

    def predict_batch(
        self,
        X: np.ndarray,
        batch_size: int = 1000,
        return_proba: bool = True
    ) -> Dict[str, Any]:
        """
        Predict on large dataset in batches.

        Args:
            X: Input features
            batch_size: Number of samples per batch
            return_proba: Include probabilities

        Returns:
            Combined results from all batches
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        all_predictions = []
        all_probabilities = [] if return_proba else None

        logger.info(f"Processing {n_samples} samples in {n_batches} batches...")

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            X_batch = X[start_idx:end_idx]

            if return_proba:
                pred, proba = self.predict(X_batch, return_proba=True)
                all_predictions.extend(pred)
                all_probabilities.append(proba)
            else:
                pred = self.predict(X_batch)
                all_predictions.extend(pred)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{n_batches} batches")

        all_predictions = np.array(all_predictions)

        results = {
            'predictions': all_predictions,
            'n_samples': n_samples,
            'batch_size': batch_size,
            'n_batches': n_batches
        }

        if return_proba:
            results['probabilities'] = np.vstack(all_probabilities)

        # Add class distribution
        unique, counts = np.unique(all_predictions, return_counts=True)
        class_dist = {}
        for cls, count in zip(unique, counts):
            key = self.class_names[cls] if self.class_names else f"class_{cls}"
            class_dist[key] = {
                'count': int(count),
                'proportion': float(count / n_samples)
            }

        results['class_distribution'] = class_dist

        return results

    def explain_prediction(
        self,
        X: np.ndarray,
        feature_importance: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Explain prediction with feature contributions.

        Args:
            X: Single sample features
            feature_importance: Feature importance scores (from model)

        Returns:
            Explanation dictionary
        """
        # Get prediction
        prediction = self.predict_single(X, return_proba=True, top_k=3)

        explanation = {
            'prediction': prediction,
            'features': {}
        }

        # Add feature values
        if X.ndim == 1:
            X = X.reshape(1, -1)

        feature_values = X[0]

        for i, value in enumerate(feature_values):
            feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"

            feature_info = {
                'value': float(value),
                'index': i
            }

            if feature_importance is not None and i < len(feature_importance):
                feature_info['importance'] = float(feature_importance[i])

            explanation['features'][feature_name] = feature_info

        # Sort features by importance if available
        if feature_importance is not None:
            top_features = sorted(
                explanation['features'].items(),
                key=lambda x: x[1].get('importance', 0),
                reverse=True
            )[:10]

            explanation['top_features'] = {k: v for k, v in top_features}

        return explanation

    def save_predictions(
        self,
        predictions: Dict[str, Any],
        filepath: Union[str, Path],
        format: str = 'json'
    ):
        """
        Save predictions to file.

        Args:
            predictions: Prediction results
            filepath: Output file path
            format: Output format ('json', 'csv', 'parquet')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(predictions, f, indent=2)

        elif format == 'csv':
            # Convert to DataFrame
            if 'predictions' in predictions and isinstance(predictions['predictions'], list):
                df = pd.DataFrame(predictions['predictions'])
            else:
                df = pd.DataFrame([predictions])

            df.to_csv(filepath, index=False)

        elif format == 'parquet':
            if 'predictions' in predictions and isinstance(predictions['predictions'], list):
                df = pd.DataFrame(predictions['predictions'])
            else:
                df = pd.DataFrame([predictions])

            df.to_parquet(filepath, index=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Predictions saved to {filepath}")


def create_inference_report(
    inference_results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Create human-readable inference report.

    Args:
        inference_results: Results from predict_with_metadata()
        output_path: Optional path to save report

    Returns:
        Report text
    """
    report_lines = [
        "=" * 80,
        "EXOPLANET CLASSIFICATION - INFERENCE REPORT",
        "=" * 80,
        ""
    ]

    # Model info
    metadata = inference_results.get('metadata', {})
    report_lines.extend([
        "MODEL INFORMATION:",
        f"  Model Type: {metadata.get('model_type', 'Unknown')}",
        f"  Number of Classes: {metadata.get('n_classes', 'Unknown')}",
        f"  Classes: {', '.join(map(str, metadata.get('class_names', [])))}" if metadata.get('class_names') else "",
        ""
    ])

    # Summary statistics
    if 'summary' in inference_results:
        summary = inference_results['summary']
        report_lines.extend([
            "PREDICTION SUMMARY:",
            f"  Total Samples: {inference_results['n_samples']}",
            f"  Mean Confidence: {summary.get('mean_confidence', 0):.4f}",
            f"  Min Confidence: {summary.get('min_confidence', 0):.4f}",
            f"  Max Confidence: {summary.get('max_confidence', 0):.4f}",
            f"  Std Confidence: {summary.get('std_confidence', 0):.4f}",
        ])

        if 'n_uncertain' in summary:
            report_lines.append(
                f"  Uncertain Predictions: {summary['n_uncertain']} "
                f"({summary['uncertainty_rate']*100:.2f}%)"
            )

        report_lines.append("")

    # Class distribution
    if 'class_distribution' in inference_results:
        report_lines.extend([
            "CLASS DISTRIBUTION:",
        ])

        for entry in inference_results['class_distribution']:
            class_name = entry.get('class_name', f"Class {entry['class_index']}")
            count = entry['count']
            proportion = entry['proportion']
            report_lines.append(
                f"  {class_name}: {count} samples ({proportion*100:.2f}%)"
            )

        report_lines.append("")

    # Sample predictions
    if 'predictions' in inference_results:
        n_samples = min(5, len(inference_results['predictions']))
        report_lines.extend([
            f"SAMPLE PREDICTIONS (first {n_samples}):",
        ])

        for i, pred in enumerate(inference_results['predictions'][:n_samples]):
            class_name = pred.get('predicted_class_name', f"Class {pred['predicted_class_index']}")
            confidence = pred.get('confidence', 0)

            report_lines.append(
                f"  Sample {i+1}: {class_name} (confidence: {confidence:.4f})"
            )

            if 'top_k_predictions' in pred:
                report_lines.append("    Top predictions:")
                for top_pred in pred['top_k_predictions'][:3]:
                    top_class = top_pred.get('class_name', f"Class {top_pred['class_index']}")
                    top_prob = top_pred['probability']
                    report_lines.append(f"      - {top_class}: {top_prob:.4f}")

        report_lines.append("")

    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Report saved to {output_path}")

    return report_text


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Inference Engine Module")
    print("=" * 60)
    print("\nFeatures:")
    print("✓ Probability predictions")
    print("✓ Class name mapping")
    print("✓ Metadata integration")
    print("✓ Batch processing")
    print("✓ Confidence thresholds")
    print("\nExample usage:")
    print("""
    # Create inference engine
    engine = InferenceEngine(
        model=trained_model,
        metadata={'class_names': ['No Planet', 'Planet'], 'feature_names': [...]}
    )

    # Predict with metadata
    results = engine.predict_with_metadata(
        X_test,
        return_proba=True,
        return_class_names=True,
        threshold=0.8,
        top_k=3
    )

    # Create report
    report = create_inference_report(results)
    print(report)
    """)
