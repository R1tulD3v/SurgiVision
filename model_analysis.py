"""
Spleen Anomaly Detection Model Analysis
Standalone script for comprehensive model evaluation and visualization
Run: python model_analysis.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config
import data_splits
import inference

warnings.filterwarnings('ignore')

# Import your custom classes
from enhanced_anomaly_creator import MedicalAnomalyCreator
from spleen_anomaly_detector_fixed import Spleen3DAnomalyDetectorFixed


class SpleenModelAnalyzer:
    def __init__(self, model_path, threshold=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = Spleen3DAnomalyDetectorFixed(model_path)
        self.threshold = threshold if threshold is not None else config.DEFAULT_THRESHOLD
        self.results = []

        # Set style for professional plots
        plt.style.use('default')
        sns.set_palette("husl")

    def evaluate_model(self, seed=None, calibrate=True):
        """Evaluate the model with a leakage-aware protocol.

        - The decision threshold is calibrated on the VALIDATION split.
        - Normal performance is measured on a held-out TEST split that was never
          used for training or calibration.
        - Synthetic (intensity-manipulated) anomalies are a SANITY CHECK only,
          not a measure of real-world performance.
        """
        if seed is None:
            seed = config.RANDOM_SEED
        inference.set_global_seed(seed)

        n = len(self.detector.preprocessor.image_files)
        train_idx, val_idx, test_idx = data_splits.train_val_test_indices(n, seed=seed)
        print("🔬 Starting Model Evaluation (leakage-aware)...")
        print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

        if calibrate and val_idx:
            self.threshold, _ = self.detector.calibrate_threshold(val_idx)
        print(f"Threshold (calibrated on validation): {self.threshold:.6f}")
        print("⚠️  Synthetic anomalies are a sanity check, not real-world performance.")
        print("-" * 50)

        all_results = []

        # --- Normal cases: held-out TEST split ---
        print("📊 Evaluating Normal Cases (held-out test split)...")
        for i in test_idx:
            try:
                result = self.detector.detect_anomaly_from_training_file(i, self.threshold)
                if result:
                    all_results.append({
                        'case_id': f'normal_{i}',
                        'reconstruction_error': result['reconstruction_error'],
                        'true_label': 0,
                        'predicted_label': 1 if result['is_anomaly'] else 0,
                        'confidence': result['confidence'],
                        'case_type': 'Normal'
                    })
            except Exception as e:
                print(f"Error in normal case {i}: {e}")

        print(f"✅ Processed {len([r for r in all_results if r['true_label'] == 0])} normal cases")

        # --- Synthetic anomalies (SANITY CHECK), built from test-split bases ---
        print("🩺 Evaluating Synthetic Anomaly Cases (sanity check)...")
        anomaly_creator = MedicalAnomalyCreator(self.detector.preprocessor)
        pathology_types = ["Large Spleen Cyst", "Spleen Infarct", "Spleen Laceration",
                           "Hyperdense Mass", "Multiple Metastases"]
        pathology_map = {"Large Spleen Cyst": 0, "Spleen Infarct": 1,
                         "Spleen Laceration": 2, "Hyperdense Mass": 3, "Multiple Metastases": 4}
        synthetic_bases = (test_idx or list(range(n)))[:5]

        for pathology in pathology_types:
            for base_idx in synthetic_bases:
                try:
                    pathological_cases = anomaly_creator.create_all_pathologies(base_index=base_idx)
                    case_idx = pathology_map.get(pathology, 0)
                    if case_idx < len(pathological_cases):
                        case = pathological_cases[case_idx]
                        masked_volume = inference.apply_spleen_mask(case['volume'], case['mask'])
                        reconstruction_error = inference.reconstruction_error(
                            self.detector.model, masked_volume, self.device
                        )
                        decision = inference.decide_anomaly(reconstruction_error, self.threshold)
                        all_results.append({
                            'case_id': f'{pathology}_{base_idx}',
                            'reconstruction_error': reconstruction_error,
                            'true_label': 1,
                            'predicted_label': 1 if decision['is_anomaly'] else 0,
                            'confidence': decision['confidence'],
                            'case_type': pathology
                        })
                except Exception as e:
                    print(f"Error in {pathology} case (base {base_idx}): {e}")

        anomaly_count = len([r for r in all_results if r['true_label'] == 1])
        print(f"✅ Processed {anomaly_count} synthetic anomaly cases")

        self.results = all_results
        return all_results

    def calculate_metrics(self):
        """Calculate comprehensive metrics"""
        y_true = [r['true_label'] for r in self.results]
        y_pred = [r['predicted_label'] for r in self.results]
        y_scores = [r['reconstruction_error'] for r in self.results]

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(set(y_true)) > 1 else (0,0,0,0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # AUC-ROC
        auc_roc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else None

        metrics = {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1, 'specificity': specificity, 'sensitivity': sensitivity,
            'auc_roc': auc_roc, 'true_positives': tp, 'true_negatives': tn,
            'false_positives': fp, 'false_negatives': fn,
            'total_cases': len(self.results)
        }

        return metrics

    def create_visualizations(self, save_path="model_analysis_plots"):
        """Create all visualization plots"""
        Path(save_path).mkdir(exist_ok=True)

        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11

        # 1. Confusion Matrix
        self._plot_confusion_matrix(save_path)

        # 2. ROC Curve
        self._plot_roc_curve(save_path)

        # 3. Precision-Recall Curve
        self._plot_precision_recall_curve(save_path)

        # 4. Error Distribution
        self._plot_error_distribution(save_path)

        # 5. Pathology Performance
        self._plot_pathology_performance(save_path)

        # 6. Metrics Summary
        self._plot_metrics_summary(save_path)

        print(f"📊 All plots saved to '{save_path}/' directory")

    def _plot_confusion_matrix(self, save_path):
        """Plot confusion matrix"""
        y_true = [r['true_label'] for r in self.results]
        y_pred = [r['predicted_label'] for r in self.results]

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix - Spleen Anomaly Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_roc_curve(self, save_path):
        """Plot ROC curve"""
        y_true = [r['true_label'] for r in self.results]
        y_scores = [r['reconstruction_error'] for r in self.results]

        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)

            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
            plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
            plt.title('ROC Curve - Spleen Anomaly Detection', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_path}/roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_precision_recall_curve(self, save_path):
        """Plot Precision-Recall curve"""
        y_true = [r['true_label'] for r in self.results]
        y_scores = [r['reconstruction_error'] for r in self.results]

        if len(set(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall Curve')
            plt.xlabel('Recall (Sensitivity)', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curve - Spleen Anomaly Detection', fontsize=16, fontweight='bold')
            plt.legend(loc="lower left", fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_path}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_error_distribution(self, save_path):
        """Plot reconstruction error distribution"""
        normal_errors = [r['reconstruction_error'] for r in self.results if r['true_label'] == 0]
        anomaly_errors = [r['reconstruction_error'] for r in self.results if r['true_label'] == 1]

        plt.figure(figsize=(12, 6))

        if normal_errors:
            plt.hist(normal_errors, bins=30, alpha=0.7, label='Normal Cases', color='blue', density=True)
        if anomaly_errors:
            plt.hist(anomaly_errors, bins=30, alpha=0.7, label='Anomaly Cases', color='red', density=True)

        plt.axvline(x=self.threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold: {self.threshold:.6f}')

        plt.xlabel('Reconstruction Error', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Reconstruction Error Distribution', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pathology_performance(self, save_path):
        """Plot performance by pathology type"""
        pathology_results = {}

        for result in self.results:
            if result['true_label'] == 1:
                pathology = result['case_type']
                if pathology not in pathology_results:
                    pathology_results[pathology] = {'correct': 0, 'total': 0}

                pathology_results[pathology]['total'] += 1
                if result['predicted_label'] == 1:
                    pathology_results[pathology]['correct'] += 1

        if pathology_results:
            pathologies = list(pathology_results.keys())
            accuracies = [pathology_results[p]['correct'] / pathology_results[p]['total'] * 100
                         for p in pathologies]

            plt.figure(figsize=(12, 6))
            bars = plt.bar(pathologies, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

            plt.xlabel('Pathology Type', fontsize=12)
            plt.ylabel('Detection Accuracy (%)', fontsize=12)
            plt.title('Detection Accuracy by Pathology Type', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 105)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_path}/pathology_performance.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_metrics_summary(self, save_path):
        """Plot metrics summary chart"""
        metrics = self.calculate_metrics()

        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
                        metrics['f1_score'], metrics['specificity']]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values,
                      color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])

        # Add value labels on bars
        for bar, val in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Metrics Summary', fontsize=16, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def print_detailed_report(self):
        """Print comprehensive analysis report"""
        metrics = self.calculate_metrics()

        print("\n" + "="*80)
        print("🔬 SPLEEN ANOMALY DETECTION MODEL ANALYSIS REPORT")
        print("="*80)

        print("\n📊 DATASET SUMMARY")
        print("-" * 40)
        print(f"Total Cases Evaluated: {metrics['total_cases']}")
        print(f"Normal Cases: {len([r for r in self.results if r['true_label'] == 0])}")
        print(f"Anomaly Cases: {len([r for r in self.results if r['true_label'] == 1])}")
        print(f"Detection Threshold: {self.threshold:.6f}")

        print("\n🎯 PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Accuracy:     {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"Precision:    {metrics['precision']:.3f}")
        print(f"Recall:       {metrics['recall']:.3f}")
        print(f"F1-Score:     {metrics['f1_score']:.3f}")
        print(f"Specificity:  {metrics['specificity']:.3f}")
        print(f"Sensitivity:  {metrics['sensitivity']:.3f}")
        if metrics['auc_roc']:
            print(f"AUC-ROC:      {metrics['auc_roc']:.3f}")

        print("\n📈 CONFUSION MATRIX")
        print("-" * 40)
        print(f"True Positives:  {metrics['true_positives']}")
        print(f"True Negatives:  {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")

        print("\n🩺 PATHOLOGY-SPECIFIC PERFORMANCE")
        print("-" * 40)
        pathology_stats = {}
        for result in self.results:
            if result['true_label'] == 1:
                pathology = result['case_type']
                if pathology not in pathology_stats:
                    pathology_stats[pathology] = {'correct': 0, 'total': 0}
                pathology_stats[pathology]['total'] += 1
                if result['predicted_label'] == 1:
                    pathology_stats[pathology]['correct'] += 1

        for pathology, stats in pathology_stats.items():
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"{pathology:<20}: {accuracy:5.1f}% ({stats['correct']}/{stats['total']})")

        print("\n💡 PERFORMANCE ASSESSMENT")
        print("-" * 40)
        if metrics['accuracy'] >= 0.95:
            print("🎉 EXCELLENT: Outstanding model performance!")
        elif metrics['accuracy'] >= 0.90:
            print("✅ VERY GOOD: High-quality model performance")
        elif metrics['accuracy'] >= 0.80:
            print("👍 GOOD: Acceptable model performance")
        elif metrics['accuracy'] >= 0.70:
            print("⚠️  MODERATE: Performance needs improvement")
        else:
            print("❌ POOR: Significant improvement required")

        # Clinical recommendations
        print("\n🏥 CLINICAL RECOMMENDATIONS")
        print("-" * 40)
        if metrics['sensitivity'] < 0.80:
            print("• Consider lowering detection threshold to improve sensitivity")
        if metrics['specificity'] < 0.80:
            print("• Consider raising detection threshold to reduce false positives")
        if metrics['accuracy'] > 0.90:
            print("• Model ready for clinical validation studies")
        else:
            print("• Additional training data and model refinement recommended")

        print("\n" + "="*80)

def main():
    """Main function to run the complete analysis"""
    model_path = config.AUTOENCODER_PATH

    if not Path(model_path).exists():
        print("❌ Model file not found!")
        print(f"Expected path: {model_path}")
        return

    print("🚀 Starting Spleen Anomaly Detection Model Analysis")
    print("="*60)

    # Initialize analyzer (threshold will be calibrated on the validation split)
    analyzer = SpleenModelAnalyzer(model_path)

    # Run leakage-aware evaluation
    analyzer.evaluate_model()

    # Generate all visualizations
    analyzer.create_visualizations(save_path="presentation_plots")

    # Print comprehensive report
    analyzer.print_detailed_report()

    print("\n✅ Analysis Complete!")
    print("📁 Visualization plots saved in 'presentation_plots/' directory")
    print("🎯 Use these plots for your presentation!")

if __name__ == "__main__":
    main()
