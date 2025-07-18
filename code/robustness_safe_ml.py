#!/usr/bin/env python3
"""
Robustness and SAFE ML Evaluation Module
========================================
Implements comprehensive robustness testing and SAFE ML paradigm evaluation
for the credit risk assessment models.

SAFE ML Components:
- Safety: Model behavior bounds and failure modes
- Accountability: Decision audit trails and explanations
- Fairness: Bias detection and mitigation
- Ethics: Compliance with ethical AI principles

Author: Omoshola S. Owolabi
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """Results from robustness evaluation"""
    test_type: str
    baseline_performance: float
    perturbed_performance: Dict[float, float]
    stability_score: float
    failure_modes: List[str]
    recommendations: List[str]


@dataclass
class SAFEMLResult:
    """Results from SAFE ML evaluation"""
    safety_score: float
    accountability_score: float
    fairness_score: float
    ethics_score: float
    overall_score: float
    detailed_metrics: Dict[str, Any]
    compliance_status: Dict[str, bool]
    recommendations: List[str]


@dataclass
class AuditTrailEntry:
    """Single entry in the audit trail"""
    timestamp: str
    model_id: str
    input_hash: str
    prediction: float
    confidence: float
    explanation: Dict[str, float]
    decision_factors: List[str]
    

class AdversarialRobustness:
    """Test model robustness against adversarial perturbations"""
    
    def __init__(self, 
                 epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.2],
                 n_iterations: int = 100):
        """
        Initialize adversarial robustness tester
        
        Args:
            epsilon_values: Perturbation magnitudes to test
            n_iterations: Number of iterations per epsilon
        """
        self.epsilon_values = epsilon_values
        self.n_iterations = n_iterations
        
    def evaluate(self, 
                model: BaseEstimator,
                X: np.ndarray,
                y: np.ndarray,
                feature_names: Optional[List[str]] = None) -> RobustnessResult:
        """
        Evaluate model robustness to adversarial perturbations
        
        Args:
            model: Trained model to evaluate
            X: Test features
            y: True labels
            feature_names: Optional feature names
            
        Returns:
            RobustnessResult with evaluation details
        """
        logger.info("Evaluating adversarial robustness...")
        
        # Baseline performance
        y_pred_base = model.predict_proba(X)[:, 1]
        baseline_auc = roc_auc_score(y, y_pred_base)
        
        perturbed_performance = {}
        failure_modes = []
        
        for epsilon in self.epsilon_values:
            aucs = []
            
            for _ in range(self.n_iterations):
                # Generate random perturbations
                perturbation = np.random.uniform(-epsilon, epsilon, X.shape)
                X_perturbed = X + perturbation
                
                # Clip to valid range if needed
                X_perturbed = np.clip(X_perturbed, X.min(), X.max())
                
                # Evaluate on perturbed data
                try:
                    y_pred_pert = model.predict_proba(X_perturbed)[:, 1]
                    auc_pert = roc_auc_score(y, y_pred_pert)
                    aucs.append(auc_pert)
                except Exception as e:
                    failure_modes.append(f"Failed at epsilon={epsilon}: {str(e)}")
                    
            if aucs:
                perturbed_performance[epsilon] = np.mean(aucs)
            else:
                perturbed_performance[epsilon] = 0.0
                
        # Calculate stability score
        performance_drops = [
            (baseline_auc - perf) / baseline_auc 
            for perf in perturbed_performance.values()
        ]
        stability_score = 1.0 - np.mean(performance_drops)
        
        # Generate recommendations
        recommendations = []
        if stability_score < 0.9:
            recommendations.append("Consider adversarial training to improve robustness")
        if max(performance_drops) > 0.1:
            recommendations.append("Model shows significant vulnerability to perturbations")
            recommendations.append("Implement input validation and sanitization")
            
        return RobustnessResult(
            test_type="Adversarial Perturbation",
            baseline_performance=baseline_auc,
            perturbed_performance=perturbed_performance,
            stability_score=stability_score,
            failure_modes=failure_modes,
            recommendations=recommendations
        )


class TemporalStability:
    """Evaluate model stability over time periods"""
    
    def __init__(self, time_windows: List[Tuple[str, int, int]] = None):
        """
        Initialize temporal stability evaluator
        
        Args:
            time_windows: List of (name, start_idx, end_idx) tuples
        """
        self.time_windows = time_windows or [
            ("Q1", 0, 3),
            ("Q2", 3, 6),
            ("Q3", 6, 9),
            ("Q4", 9, 12)
        ]
        
    def evaluate(self,
                model: BaseEstimator,
                X_temporal: List[np.ndarray],
                y_temporal: List[np.ndarray]) -> RobustnessResult:
        """
        Evaluate model performance across time windows
        
        Args:
            model: Trained model
            X_temporal: List of feature arrays for each time period
            y_temporal: List of label arrays for each time period
            
        Returns:
            RobustnessResult
        """
        logger.info("Evaluating temporal stability...")
        
        # Evaluate performance for each time window
        window_performance = {}
        failure_modes = []
        
        for window_name, start, end in self.time_windows:
            try:
                # Aggregate data for window
                X_window = np.vstack(X_temporal[start:end])
                y_window = np.hstack(y_temporal[start:end])
                
                # Evaluate model
                y_pred = model.predict_proba(X_window)[:, 1]
                auc = roc_auc_score(y_window, y_pred)
                window_performance[window_name] = auc
                
            except Exception as e:
                failure_modes.append(f"Failed on {window_name}: {str(e)}")
                window_performance[window_name] = 0.0
                
        # Calculate stability metrics
        performances = list(window_performance.values())
        if performances:
            stability_score = 1.0 - (np.std(performances) / np.mean(performances))
            baseline_performance = np.mean(performances)
        else:
            stability_score = 0.0
            baseline_performance = 0.0
            
        # Recommendations
        recommendations = []
        if stability_score < 0.85:
            recommendations.append("Model shows temporal instability")
            recommendations.append("Consider time-aware features or model updates")
        if np.std(performances) > 0.05:
            recommendations.append("High variance across time periods detected")
            
        return RobustnessResult(
            test_type="Temporal Stability",
            baseline_performance=baseline_performance,
            perturbed_performance={str(k): v for k, v in window_performance.items()},
            stability_score=stability_score,
            failure_modes=failure_modes,
            recommendations=recommendations
        )


class DistributionShiftAnalyzer:
    """Analyze model performance under distribution shifts"""
    
    def __init__(self, shift_types: List[str] = None):
        """
        Initialize distribution shift analyzer
        
        Args:
            shift_types: Types of shifts to simulate
        """
        self.shift_types = shift_types or [
            "covariate_shift",
            "label_shift", 
            "concept_drift"
        ]
        
    def simulate_covariate_shift(self, 
                                X: np.ndarray, 
                                shift_magnitude: float = 0.5) -> np.ndarray:
        """Simulate covariate shift by changing feature distributions"""
        X_shifted = X.copy()
        
        # Shift mean and variance of features
        n_features_to_shift = max(1, int(X.shape[1] * shift_magnitude))
        features_to_shift = np.random.choice(X.shape[1], n_features_to_shift, replace=False)
        
        for feat in features_to_shift:
            # Change mean
            X_shifted[:, feat] += np.random.normal(0, shift_magnitude)
            # Change variance
            X_shifted[:, feat] *= (1 + np.random.uniform(-shift_magnitude, shift_magnitude))
            
        return X_shifted
        
    def simulate_label_shift(self, 
                           y: np.ndarray, 
                           shift_magnitude: float = 0.2) -> np.ndarray:
        """Simulate label shift by changing class distribution"""
        y_shifted = y.copy()
        
        # Randomly flip some labels
        n_to_flip = int(len(y) * shift_magnitude)
        flip_indices = np.random.choice(len(y), n_to_flip, replace=False)
        y_shifted[flip_indices] = 1 - y_shifted[flip_indices]
        
        return y_shifted
        
    def evaluate(self,
                model: BaseEstimator,
                X: np.ndarray,
                y: np.ndarray) -> RobustnessResult:
        """
        Evaluate model under various distribution shifts
        
        Args:
            model: Trained model
            X: Original features
            y: Original labels
            
        Returns:
            RobustnessResult
        """
        logger.info("Evaluating distribution shift robustness...")
        
        # Baseline performance
        y_pred_base = model.predict_proba(X)[:, 1]
        baseline_auc = roc_auc_score(y, y_pred_base)
        
        shift_performance = {}
        failure_modes = []
        
        # Test different shift magnitudes
        for shift_mag in [0.1, 0.3, 0.5]:
            # Covariate shift
            X_cov_shifted = self.simulate_covariate_shift(X, shift_mag)
            try:
                y_pred = model.predict_proba(X_cov_shifted)[:, 1]
                auc = roc_auc_score(y, y_pred)
                shift_performance[f"covariate_shift_{shift_mag}"] = auc
            except Exception as e:
                failure_modes.append(f"Covariate shift {shift_mag}: {str(e)}")
                
            # Label shift (evaluate on shifted labels)
            y_label_shifted = self.simulate_label_shift(y, shift_mag)
            try:
                auc = roc_auc_score(y_label_shifted, y_pred_base)
                shift_performance[f"label_shift_{shift_mag}"] = auc
            except Exception as e:
                failure_modes.append(f"Label shift {shift_mag}: {str(e)}")
                
        # Calculate stability
        performances = list(shift_performance.values())
        stability_score = np.mean(performances) / baseline_auc if baseline_auc > 0 else 0
        
        # Recommendations
        recommendations = []
        if stability_score < 0.9:
            recommendations.append("Model vulnerable to distribution shifts")
            recommendations.append("Consider domain adaptation techniques")
            recommendations.append("Implement drift detection monitoring")
            
        return RobustnessResult(
            test_type="Distribution Shift",
            baseline_performance=baseline_auc,
            perturbed_performance=shift_performance,
            stability_score=stability_score,
            failure_modes=failure_modes,
            recommendations=recommendations
        )


class SAFEMLEvaluator:
    """
    Comprehensive SAFE ML evaluation framework
    Based on Babaei & Giudici (2025) SAFE AI paradigm
    """
    
    def __init__(self):
        """Initialize SAFE ML evaluator"""
        self.safety_criteria = {
            'prediction_bounds': (0.0, 1.0),
            'confidence_threshold': 0.7,
            'uncertainty_threshold': 0.3
        }
        
        self.fairness_metrics = [
            'demographic_parity',
            'equal_opportunity',
            'equalized_odds',
            'disparate_impact'
        ]
        
        self.accountability_requirements = [
            'explanation_available',
            'decision_traceable',
            'audit_trail_complete',
            'human_reviewable'
        ]
        
    def evaluate_safety(self,
                       model: BaseEstimator,
                       X: np.ndarray,
                       predictions: np.ndarray,
                       explanations: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate model safety criteria
        
        Args:
            model: Trained model
            X: Input features
            predictions: Model predictions
            explanations: Optional SHAP values
            
        Returns:
            Safety evaluation results
        """
        safety_metrics = {}
        
        # Check prediction bounds
        out_of_bounds = np.sum((predictions < 0) | (predictions > 1))
        safety_metrics['predictions_in_bounds'] = out_of_bounds == 0
        safety_metrics['out_of_bounds_ratio'] = out_of_bounds / len(predictions)
        
        # Check confidence levels
        confidence = np.abs(predictions - 0.5) * 2  # Distance from decision boundary
        low_confidence = np.sum(confidence < self.safety_criteria['confidence_threshold'])
        safety_metrics['low_confidence_ratio'] = low_confidence / len(predictions)
        
        # Check for extreme predictions
        extreme_predictions = np.sum((predictions < 0.05) | (predictions > 0.95))
        safety_metrics['extreme_prediction_ratio'] = extreme_predictions / len(predictions)
        
        # Explanation consistency (if available)
        if explanations is not None:
            # Check if explanations sum approximately to prediction difference
            base_value = 0.5  # Assuming balanced baseline
            exp_sum = np.sum(explanations, axis=1)
            consistency_error = np.abs((predictions - base_value) - exp_sum)
            safety_metrics['explanation_consistency'] = np.mean(consistency_error < 0.1)
            
        # Calculate overall safety score
        safety_score = np.mean([
            safety_metrics['predictions_in_bounds'],
            1 - safety_metrics['low_confidence_ratio'],
            1 - safety_metrics['extreme_prediction_ratio'],
            safety_metrics.get('explanation_consistency', 1.0)
        ])
        
        return {
            'score': safety_score,
            'metrics': safety_metrics,
            'safe': safety_score > 0.8
        }
        
    def evaluate_accountability(self,
                              model: BaseEstimator,
                              audit_trail: List[AuditTrailEntry],
                              explanations_available: bool = True) -> Dict[str, Any]:
        """
        Evaluate model accountability
        
        Args:
            model: Trained model
            audit_trail: List of audit trail entries
            explanations_available: Whether explanations are provided
            
        Returns:
            Accountability evaluation results
        """
        accountability_metrics = {}
        
        # Check audit trail completeness
        if audit_trail:
            # Verify all required fields are present
            complete_entries = sum(
                1 for entry in audit_trail
                if all([entry.timestamp, entry.model_id, entry.input_hash,
                       entry.prediction is not None, entry.confidence is not None])
            )
            accountability_metrics['audit_trail_completeness'] = complete_entries / len(audit_trail)
        else:
            accountability_metrics['audit_trail_completeness'] = 0.0
            
        # Check explanation availability
        accountability_metrics['explanations_available'] = explanations_available
        
        # Check decision traceability
        if audit_trail:
            traceable = sum(
                1 for entry in audit_trail
                if entry.decision_factors and len(entry.decision_factors) > 0
            )
            accountability_metrics['decision_traceability'] = traceable / len(audit_trail)
        else:
            accountability_metrics['decision_traceability'] = 0.0
            
        # Human reviewability
        accountability_metrics['human_reviewable'] = (
            explanations_available and 
            accountability_metrics['audit_trail_completeness'] > 0.9
        )
        
        # Calculate overall score
        accountability_score = np.mean([
            accountability_metrics['audit_trail_completeness'],
            float(accountability_metrics['explanations_available']),
            accountability_metrics['decision_traceability'],
            float(accountability_metrics['human_reviewable'])
        ])
        
        return {
            'score': accountability_score,
            'metrics': accountability_metrics,
            'accountable': accountability_score > 0.8
        }
        
    def evaluate_fairness(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         sensitive_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model fairness across protected groups
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            sensitive_features: DataFrame with sensitive attributes
            
        Returns:
            Fairness evaluation results
        """
        fairness_metrics = {}
        
        for attr in sensitive_features.columns:
            groups = sensitive_features[attr].unique()
            if len(groups) != 2:
                continue  # Skip non-binary attributes for now
                
            # Split by group
            group_0_mask = sensitive_features[attr] == groups[0]
            group_1_mask = sensitive_features[attr] == groups[1]
            
            # Demographic parity
            accept_rate_0 = np.mean(y_pred[group_0_mask])
            accept_rate_1 = np.mean(y_pred[group_1_mask])
            demographic_parity = min(accept_rate_0, accept_rate_1) / max(accept_rate_0, accept_rate_1)
            fairness_metrics[f'{attr}_demographic_parity'] = demographic_parity
            
            # Equal opportunity (TPR parity)
            tpr_0 = np.mean(y_pred[group_0_mask & (y_true == 1)])
            tpr_1 = np.mean(y_pred[group_1_mask & (y_true == 1)])
            if tpr_0 > 0 and tpr_1 > 0:
                equal_opportunity = min(tpr_0, tpr_1) / max(tpr_0, tpr_1)
                fairness_metrics[f'{attr}_equal_opportunity'] = equal_opportunity
                
            # Disparate impact
            if accept_rate_0 > 0:
                disparate_impact = accept_rate_1 / accept_rate_0
                fairness_metrics[f'{attr}_disparate_impact'] = min(disparate_impact, 1/disparate_impact)
                
        # Overall fairness score
        if fairness_metrics:
            fairness_score = np.mean(list(fairness_metrics.values()))
        else:
            fairness_score = 1.0  # No sensitive features provided
            
        return {
            'score': fairness_score,
            'metrics': fairness_metrics,
            'fair': fairness_score > 0.8
        }
        
    def evaluate_ethics(self,
                       model_purpose: str,
                       data_consent: bool,
                       transparency_level: str,
                       human_oversight: bool) -> Dict[str, Any]:
        """
        Evaluate ethical compliance
        
        Args:
            model_purpose: Stated purpose of the model
            data_consent: Whether data was collected with consent
            transparency_level: Level of transparency ('high', 'medium', 'low')
            human_oversight: Whether human oversight is implemented
            
        Returns:
            Ethics evaluation results
        """
        ethics_metrics = {}
        
        # Purpose alignment
        acceptable_purposes = [
            'credit risk assessment',
            'financial inclusion',
            'fair lending',
            'risk management'
        ]
        ethics_metrics['purpose_aligned'] = any(
            purpose in model_purpose.lower() 
            for purpose in acceptable_purposes
        )
        
        # Data ethics
        ethics_metrics['data_consent'] = data_consent
        
        # Transparency
        transparency_scores = {'high': 1.0, 'medium': 0.7, 'low': 0.3}
        ethics_metrics['transparency_score'] = transparency_scores.get(transparency_level, 0.5)
        
        # Human oversight
        ethics_metrics['human_oversight'] = human_oversight
        
        # Calculate overall ethics score
        ethics_score = np.mean([
            float(ethics_metrics['purpose_aligned']),
            float(ethics_metrics['data_consent']),
            ethics_metrics['transparency_score'],
            float(ethics_metrics['human_oversight'])
        ])
        
        return {
            'score': ethics_score,
            'metrics': ethics_metrics,
            'ethical': ethics_score > 0.8
        }
        
    def comprehensive_evaluation(self,
                               model: BaseEstimator,
                               X: np.ndarray,
                               y_true: np.ndarray,
                               predictions: np.ndarray,
                               explanations: Optional[np.ndarray] = None,
                               sensitive_features: Optional[pd.DataFrame] = None,
                               audit_trail: Optional[List[AuditTrailEntry]] = None,
                               model_metadata: Optional[Dict[str, Any]] = None) -> SAFEMLResult:
        """
        Perform comprehensive SAFE ML evaluation
        
        Args:
            model: Trained model
            X: Input features
            y_true: True labels
            predictions: Model predictions (probabilities)
            explanations: Optional SHAP values
            sensitive_features: Optional sensitive attributes
            audit_trail: Optional audit trail
            model_metadata: Optional model metadata
            
        Returns:
            SAFEMLResult with comprehensive evaluation
        """
        logger.info("Performing comprehensive SAFE ML evaluation...")
        
        # Default metadata
        if model_metadata is None:
            model_metadata = {
                'purpose': 'credit risk assessment',
                'data_consent': True,
                'transparency_level': 'high',
                'human_oversight': True
            }
            
        # Safety evaluation
        safety_eval = self.evaluate_safety(model, X, predictions, explanations)
        
        # Accountability evaluation
        accountability_eval = self.evaluate_accountability(
            model, 
            audit_trail or [],
            explanations is not None
        )
        
        # Fairness evaluation
        if sensitive_features is not None:
            y_pred_binary = (predictions > 0.5).astype(int)
            fairness_eval = self.evaluate_fairness(y_true, y_pred_binary, sensitive_features)
        else:
            fairness_eval = {'score': 1.0, 'metrics': {}, 'fair': True}
            
        # Ethics evaluation
        ethics_eval = self.evaluate_ethics(
            model_metadata.get('purpose', 'credit risk assessment'),
            model_metadata.get('data_consent', True),
            model_metadata.get('transparency_level', 'high'),
            model_metadata.get('human_oversight', True)
        )
        
        # Overall SAFE score
        overall_score = np.mean([
            safety_eval['score'],
            accountability_eval['score'],
            fairness_eval['score'],
            ethics_eval['score']
        ])
        
        # Compliance status
        compliance_status = {
            'safety_compliant': safety_eval['safe'],
            'accountability_compliant': accountability_eval['accountable'],
            'fairness_compliant': fairness_eval['fair'],
            'ethics_compliant': ethics_eval['ethical'],
            'overall_compliant': overall_score > 0.8
        }
        
        # Recommendations
        recommendations = []
        
        if safety_eval['score'] < 0.8:
            recommendations.append("Improve model safety through better calibration")
            recommendations.append("Implement prediction bounds checking")
            
        if accountability_eval['score'] < 0.8:
            recommendations.append("Enhance audit trail completeness")
            recommendations.append("Ensure all decisions are traceable")
            
        if fairness_eval['score'] < 0.8:
            recommendations.append("Address fairness disparities across groups")
            recommendations.append("Consider fairness-aware training")
            
        if ethics_eval['score'] < 0.8:
            recommendations.append("Strengthen ethical safeguards")
            recommendations.append("Increase transparency measures")
            
        # Create detailed metrics
        detailed_metrics = {
            'safety': safety_eval,
            'accountability': accountability_eval,
            'fairness': fairness_eval,
            'ethics': ethics_eval
        }
        
        return SAFEMLResult(
            safety_score=safety_eval['score'],
            accountability_score=accountability_eval['score'],
            fairness_score=fairness_eval['score'],
            ethics_score=ethics_eval['score'],
            overall_score=overall_score,
            detailed_metrics=detailed_metrics,
            compliance_status=compliance_status,
            recommendations=recommendations
        )


class RobustnessSAFEMLVisualizer:
    """Create visualizations for robustness and SAFE ML results"""
    
    @staticmethod
    def plot_robustness_results(robustness_results: List[RobustnessResult],
                               save_path: Optional[str] = None):
        """Create comprehensive robustness visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Adversarial robustness
        ax = axes[0, 0]
        adv_result = next((r for r in robustness_results if r.test_type == "Adversarial Perturbation"), None)
        if adv_result:
            epsilons = list(adv_result.perturbed_performance.keys())
            performances = list(adv_result.perturbed_performance.values())
            
            ax.plot([0] + epsilons, [adv_result.baseline_performance] + performances, 
                   'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Perturbation Magnitude (ε)')
            ax.set_ylabel('AUC Score')
            ax.set_title('Adversarial Robustness')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.5, 1.0)
            
        # 2. Temporal stability
        ax = axes[0, 1]
        temp_result = next((r for r in robustness_results if r.test_type == "Temporal Stability"), None)
        if temp_result:
            windows = list(temp_result.perturbed_performance.keys())
            performances = list(temp_result.perturbed_performance.values())
            
            ax.bar(windows, performances, alpha=0.7)
            ax.axhline(y=temp_result.baseline_performance, color='r', 
                      linestyle='--', label='Average')
            ax.set_xlabel('Time Window')
            ax.set_ylabel('AUC Score')
            ax.set_title('Temporal Stability')
            ax.legend()
            ax.set_ylim(0.5, 1.0)
            
        # 3. Distribution shift
        ax = axes[1, 0]
        dist_result = next((r for r in robustness_results if r.test_type == "Distribution Shift"), None)
        if dist_result:
            shift_types = list(dist_result.perturbed_performance.keys())
            performances = list(dist_result.perturbed_performance.values())
            
            # Group by shift type
            shift_data = {}
            for shift, perf in zip(shift_types, performances):
                shift_type = shift.split('_')[0] + '_' + shift.split('_')[1]
                magnitude = float(shift.split('_')[2])
                if shift_type not in shift_data:
                    shift_data[shift_type] = []
                shift_data[shift_type].append((magnitude, perf))
                
            for shift_type, data in shift_data.items():
                data.sort()
                mags, perfs = zip(*data)
                ax.plot(mags, perfs, 'o-', label=shift_type, linewidth=2)
                
            ax.set_xlabel('Shift Magnitude')
            ax.set_ylabel('AUC Score')
            ax.set_title('Distribution Shift Robustness')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.5, 1.0)
            
        # 4. Overall stability scores
        ax = axes[1, 1]
        test_types = [r.test_type for r in robustness_results]
        stability_scores = [r.stability_score for r in robustness_results]
        
        bars = ax.bar(range(len(test_types)), stability_scores, alpha=0.7)
        ax.set_xticks(range(len(test_types)))
        ax.set_xticklabels(test_types, rotation=45, ha='right')
        ax.set_ylabel('Stability Score')
        ax.set_title('Overall Robustness Summary')
        ax.set_ylim(0, 1.1)
        
        # Color bars by performance
        for bar, score in zip(bars, stability_scores):
            if score >= 0.9:
                bar.set_color('green')
            elif score >= 0.8:
                bar.set_color('orange')
            else:
                bar.set_color('red')
                
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, stability_scores)):
            ax.text(i, score + 0.02, f'{score:.2f}', ha='center', va='bottom')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
        
    @staticmethod
    def plot_safe_ml_results(safe_result: SAFEMLResult,
                           save_path: Optional[str] = None):
        """Create SAFE ML evaluation visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. SAFE scores radar chart
        ax = axes[0, 0]
        categories = ['Safety', 'Accountability', 'Fairness', 'Ethics']
        scores = [
            safe_result.safety_score,
            safe_result.accountability_score,
            safe_result.fairness_score,
            safe_result.ethics_score
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, scores, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, scores, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('SAFE ML Scores', fontsize=14, pad=20)
        ax.grid(True)
        
        # Add threshold circle
        threshold = [0.8] * (len(categories) + 1)
        ax.plot(angles, threshold, '--', color='red', alpha=0.5, label='Compliance Threshold')
        ax.legend()
        
        # 2. Compliance status
        ax = axes[0, 1]
        compliance_items = list(safe_result.compliance_status.keys())
        compliance_values = [int(v) for v in safe_result.compliance_status.values()]
        
        colors = ['green' if v else 'red' for v in compliance_values]
        bars = ax.barh(compliance_items, compliance_values, color=colors, alpha=0.7)
        
        ax.set_xlim(0, 1.2)
        ax.set_xlabel('Compliant')
        ax.set_title('Compliance Status')
        
        # Add text labels
        for bar, val in zip(bars, compliance_values):
            label = 'Yes' if val else 'No'
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   label, va='center')
                   
        # 3. Detailed metrics heatmap
        ax = axes[1, 0]
        
        # Extract key metrics
        metric_data = []
        metric_labels = []
        
        for component in ['safety', 'accountability', 'fairness', 'ethics']:
            if component in safe_result.detailed_metrics:
                metrics = safe_result.detailed_metrics[component]['metrics']
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metric_data.append(value)
                        metric_labels.append(f"{component[:3].upper()}: {metric[:20]}")
                        
        # Create heatmap
        if metric_data:
            metric_matrix = np.array(metric_data).reshape(-1, 1)
            im = ax.imshow(metric_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax.set_yticks(range(len(metric_labels)))
            ax.set_yticklabels(metric_labels, fontsize=8)
            ax.set_xticks([])
            ax.set_title('Detailed Metrics')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Score', rotation=270, labelpad=15)
            
        # 4. Overall summary
        ax = axes[1, 1]
        ax.text(0.5, 0.9, 'SAFE ML Evaluation Summary', 
                ha='center', va='center', fontsize=16, weight='bold',
                transform=ax.transAxes)
                
        summary_text = f"""
Overall Score: {safe_result.overall_score:.2f}

Component Scores:
• Safety: {safe_result.safety_score:.2f}
• Accountability: {safe_result.accountability_score:.2f}
• Fairness: {safe_result.fairness_score:.2f}
• Ethics: {safe_result.ethics_score:.2f}

Compliance: {'PASSED' if safe_result.compliance_status['overall_compliant'] else 'FAILED'}

Top Recommendations:
"""
        
        ax.text(0.1, 0.65, summary_text, ha='left', va='top',
                fontsize=12, transform=ax.transAxes)
                
        # Add recommendations
        y_pos = 0.25
        for i, rec in enumerate(safe_result.recommendations[:3], 1):
            ax.text(0.1, y_pos, f"{i}. {rec[:50]}...", ha='left', va='top',
                    fontsize=10, transform=ax.transAxes)
            y_pos -= 0.05
            
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def create_audit_trail_entry(model_id: str,
                           X: np.ndarray,
                           prediction: float,
                           confidence: float,
                           explanation: Dict[str, float],
                           top_features: int = 5) -> AuditTrailEntry:
    """Create an audit trail entry for a prediction"""
    # Hash input for privacy
    input_str = str(X.tolist())
    input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]
    
    # Get top decision factors
    if explanation:
        sorted_features = sorted(explanation.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)
        decision_factors = [
            f"{feat}: {val:.3f}" 
            for feat, val in sorted_features[:top_features]
        ]
    else:
        decision_factors = ["No explanation available"]
        
    return AuditTrailEntry(
        timestamp=datetime.now().isoformat(),
        model_id=model_id,
        input_hash=input_hash,
        prediction=prediction,
        confidence=confidence,
        explanation=explanation,
        decision_factors=decision_factors
    )


def demonstrate_robustness_safe_ml():
    """Demonstration of robustness and SAFE ML evaluation"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.3 > 0).astype(int)
    
    # Train a simple model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:800], y[:800])
    
    # Test data
    X_test = X[800:]
    y_test = y[800:]
    
    print("Robustness and SAFE ML Evaluation Demonstration")
    print("=" * 60)
    
    # 1. Robustness evaluation
    print("\n1. Robustness Evaluation")
    print("-" * 40)
    
    robustness_results = []
    
    # Adversarial robustness
    adv_evaluator = AdversarialRobustness()
    adv_result = adv_evaluator.evaluate(model, X_test, y_test)
    robustness_results.append(adv_result)
    print(f"   Adversarial Stability Score: {adv_result.stability_score:.3f}")
    
    # Temporal stability (simulated)
    temporal_evaluator = TemporalStability()
    X_temporal = [X_test[i:i+50] for i in range(0, 200, 50)]
    y_temporal = [y_test[i:i+50] for i in range(0, 200, 50)]
    temp_result = temporal_evaluator.evaluate(model, X_temporal, y_temporal)
    robustness_results.append(temp_result)
    print(f"   Temporal Stability Score: {temp_result.stability_score:.3f}")
    
    # Distribution shift
    dist_evaluator = DistributionShiftAnalyzer()
    dist_result = dist_evaluator.evaluate(model, X_test, y_test)
    robustness_results.append(dist_result)
    print(f"   Distribution Shift Score: {dist_result.stability_score:.3f}")
    
    # 2. SAFE ML evaluation
    print("\n2. SAFE ML Evaluation")
    print("-" * 40)
    
    # Generate predictions
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Create synthetic sensitive features
    sensitive_features = pd.DataFrame({
        'gender': np.random.choice(['M', 'F'], size=len(X_test)),
        'age_group': np.random.choice(['young', 'old'], size=len(X_test))
    })
    
    # Create audit trail
    audit_trail = []
    for i in range(min(10, len(X_test))):
        # Simulate explanations
        feature_importance = model.feature_importances_
        explanation = {f'feature_{j}': feature_importance[j] * X_test[i, j] 
                      for j in range(n_features)}
        
        entry = create_audit_trail_entry(
            model_id="RF_v1.0",
            X=X_test[i],
            prediction=predictions[i],
            confidence=abs(predictions[i] - 0.5) * 2,
            explanation=explanation
        )
        audit_trail.append(entry)
        
    # Perform SAFE ML evaluation
    safe_evaluator = SAFEMLEvaluator()
    safe_result = safe_evaluator.comprehensive_evaluation(
        model=model,
        X=X_test,
        y_true=y_test,
        predictions=predictions,
        explanations=None,  # Would use SHAP values in practice
        sensitive_features=sensitive_features,
        audit_trail=audit_trail,
        model_metadata={
            'purpose': 'Credit risk assessment for fair lending',
            'data_consent': True,
            'transparency_level': 'high',
            'human_oversight': True
        }
    )
    
    print(f"   Safety Score: {safe_result.safety_score:.3f}")
    print(f"   Accountability Score: {safe_result.accountability_score:.3f}")
    print(f"   Fairness Score: {safe_result.fairness_score:.3f}")
    print(f"   Ethics Score: {safe_result.ethics_score:.3f}")
    print(f"   Overall SAFE Score: {safe_result.overall_score:.3f}")
    print(f"   Compliance Status: {'PASSED' if safe_result.compliance_status['overall_compliant'] else 'FAILED'}")
    
    # 3. Create visualizations
    print("\n3. Generating Visualizations...")
    
    RobustnessSAFEMLVisualizer.plot_robustness_results(
        robustness_results,
        save_path='robustness_evaluation.png'
    )
    
    RobustnessSAFEMLVisualizer.plot_safe_ml_results(
        safe_result,
        save_path='safe_ml_evaluation.png'
    )
    
    print("\nEvaluation complete! Visualizations saved.")
    
    return robustness_results, safe_result


if __name__ == "__main__":
    robustness_results, safe_result = demonstrate_robustness_safe_ml()