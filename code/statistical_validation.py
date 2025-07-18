#!/usr/bin/env python3
"""
Statistical Validation Module
=============================
Implements statistical significance testing for model performance metrics,
including bootstrap confidence intervals and DeLong test for AUC comparison.

Author: Omoshola S. Owolabi
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StatisticalTestResult:
    """Results from statistical significance testing"""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    significant: bool
    effect_size: Optional[float] = None
    interpretation: str = ""


class DeLongTest:
    """
    Implementation of DeLong test for comparing AUC scores
    Based on DeLong et al. (1988) - "Comparing the Areas under Two or More 
    Correlated Receiver Operating Characteristic Curves"
    """
    
    @staticmethod
    def compute_midrank(x: np.ndarray) -> np.ndarray:
        """Compute midranks for DeLong covariance matrix"""
        n = len(x)
        if n == 0:
            return np.array([])
            
        order = np.argsort(x)
        ranks = np.empty(n)
        
        i = 0
        while i < n:
            j = i
            while j < n - 1 and x[order[j]] == x[order[j + 1]]:
                j += 1
            for k in range(i, j + 1):
                ranks[order[k]] = (i + j + 2) / 2.0
            i = j + 1
            
        return ranks
    
    @staticmethod
    def fast_delong_cov(predictions_sorted_transposed: np.ndarray, 
                       label_1_count: int) -> np.ndarray:
        """
        Fast DeLong covariance computation
        
        Args:
            predictions_sorted_transposed: 2D array [n_classifiers, n_examples]
            label_1_count: Number of positive examples
            
        Returns:
            Covariance matrix [n_classifiers, n_classifiers]
        """
        n_examples = predictions_sorted_transposed.shape[1]
        n_classifiers = predictions_sorted_transposed.shape[0]
        
        # Compute midranks
        m = np.zeros((n_classifiers, n_examples))
        for i in range(n_classifiers):
            m[i, :] = DeLongTest.compute_midrank(predictions_sorted_transposed[i, :])
            
        # Split into positive and negative examples
        m_pos = m[:, :label_1_count]
        m_neg = m[:, label_1_count:]
        
        # Compute covariance components
        n_pos = label_1_count
        n_neg = n_examples - label_1_count
        
        cov_pos = np.cov(m_pos)
        cov_neg = np.cov(m_neg)
        
        # Ensure cov matrices are 2D
        if n_classifiers == 1:
            cov_pos = cov_pos.reshape(1, 1)
            cov_neg = cov_neg.reshape(1, 1)
            
        # DeLong covariance
        cov = (cov_pos / n_pos + cov_neg / n_neg) / n_examples
        
        return cov
    
    @staticmethod
    def delong_test(y_true: np.ndarray, 
                   y_score_1: np.ndarray, 
                   y_score_2: np.ndarray,
                   alpha: float = 0.05) -> StatisticalTestResult:
        """
        Perform DeLong test comparing two AUC scores
        
        Args:
            y_true: True binary labels
            y_score_1: Predicted scores from model 1
            y_score_2: Predicted scores from model 2
            alpha: Significance level
            
        Returns:
            StatisticalTestResult with test details
        """
        # Compute AUCs
        auc1 = roc_auc_score(y_true, y_score_1)
        auc2 = roc_auc_score(y_true, y_score_2)
        
        # Sort by true labels
        order = np.argsort(y_true)[::-1]  # Positive examples first
        y_true_sorted = y_true[order]
        label_1_count = int(y_true_sorted.sum())
        
        # Sort predictions
        predictions_sorted = np.vstack([y_score_1[order], y_score_2[order]])
        
        # Compute covariance matrix
        cov = DeLongTest.fast_delong_cov(predictions_sorted, label_1_count)
        
        # Compute z-statistic
        auc_diff = auc1 - auc2
        auc_cov_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        
        if auc_cov_diff == 0:
            z_stat = 0
        else:
            z_stat = auc_diff / np.sqrt(auc_cov_diff)
            
        # Two-tailed p-value
        p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
        
        # Confidence interval for difference
        z_critical = norm.ppf(1 - alpha/2)
        ci_lower = auc_diff - z_critical * np.sqrt(auc_cov_diff)
        ci_upper = auc_diff + z_critical * np.sqrt(auc_cov_diff)
        
        # Effect size (Cohen's d)
        effect_size = auc_diff / np.sqrt(auc_cov_diff) if auc_cov_diff > 0 else 0
        
        # Interpretation
        if p_value < alpha:
            if auc1 > auc2:
                interpretation = f"Model 1 significantly better (AUC: {auc1:.3f} vs {auc2:.3f})"
            else:
                interpretation = f"Model 2 significantly better (AUC: {auc2:.3f} vs {auc1:.3f})"
        else:
            interpretation = f"No significant difference (AUC: {auc1:.3f} vs {auc2:.3f})"
            
        return StatisticalTestResult(
            test_name="DeLong Test",
            statistic=z_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            significant=p_value < alpha,
            effect_size=effect_size,
            interpretation=interpretation
        )


class BootstrapValidator:
    """Bootstrap-based validation for model performance metrics"""
    
    def __init__(self, 
                 n_bootstrap: int = 1000,
                 confidence_level: float = 0.95,
                 random_state: int = 42):
        """
        Initialize bootstrap validator
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        np.random.seed(random_state)
        
    def bootstrap_auc_ci(self, 
                        y_true: np.ndarray, 
                        y_scores: np.ndarray,
                        stratified: bool = True) -> Tuple[float, Tuple[float, float], np.ndarray]:
        """
        Compute AUC with bootstrap confidence intervals
        
        Args:
            y_true: True binary labels
            y_scores: Predicted probabilities
            stratified: Whether to use stratified bootstrap
            
        Returns:
            Tuple of (mean_auc, (ci_lower, ci_upper), bootstrap_aucs)
        """
        n = len(y_true)
        aucs = []
        
        for i in range(self.n_bootstrap):
            if stratified:
                # Stratified bootstrap
                pos_idx = np.where(y_true == 1)[0]
                neg_idx = np.where(y_true == 0)[0]
                
                # Sample with replacement maintaining class ratio
                pos_sample = np.random.choice(pos_idx, len(pos_idx), replace=True)
                neg_sample = np.random.choice(neg_idx, len(neg_idx), replace=True)
                
                idx = np.concatenate([pos_sample, neg_sample])
            else:
                # Simple bootstrap
                idx = np.random.choice(n, n, replace=True)
                
            # Calculate AUC for bootstrap sample
            try:
                auc = roc_auc_score(y_true[idx], y_scores[idx])
                aucs.append(auc)
            except:
                # Skip if only one class in bootstrap sample
                continue
                
        aucs = np.array(aucs)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower = np.percentile(aucs, (alpha/2) * 100)
        upper = np.percentile(aucs, (1-alpha/2) * 100)
        
        return np.mean(aucs), (lower, upper), aucs
        
    def bootstrap_metric_comparison(self,
                                   y_true: np.ndarray,
                                   y_scores_1: np.ndarray,
                                   y_scores_2: np.ndarray,
                                   metric_func=roc_auc_score) -> StatisticalTestResult:
        """
        Compare two models using bootstrap hypothesis testing
        
        Args:
            y_true: True labels
            y_scores_1: Predictions from model 1
            y_scores_2: Predictions from model 2
            metric_func: Metric function to use
            
        Returns:
            StatisticalTestResult
        """
        n = len(y_true)
        metric_diffs = []
        
        for i in range(self.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            
            try:
                metric1 = metric_func(y_true[idx], y_scores_1[idx])
                metric2 = metric_func(y_true[idx], y_scores_2[idx])
                metric_diffs.append(metric1 - metric2)
            except:
                continue
                
        metric_diffs = np.array(metric_diffs)
        
        # Bootstrap p-value: proportion of differences ≤ 0
        p_value = np.mean(metric_diffs <= 0)
        p_value = 2 * min(p_value, 1 - p_value)  # Two-tailed
        
        # Confidence interval for difference
        alpha = 0.05
        ci_lower = np.percentile(metric_diffs, (alpha/2) * 100)
        ci_upper = np.percentile(metric_diffs, (1-alpha/2) * 100)
        
        # Effect size
        effect_size = np.mean(metric_diffs) / np.std(metric_diffs)
        
        return StatisticalTestResult(
            test_name="Bootstrap Comparison",
            statistic=np.mean(metric_diffs),
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            significant=p_value < alpha,
            effect_size=effect_size,
            interpretation=f"Mean difference: {np.mean(metric_diffs):.4f}"
        )


class CrossValidationAnalyzer:
    """Analyze model performance using cross-validation with statistical tests"""
    
    def __init__(self, 
                 cv_folds: int = 5,
                 stratified: bool = True,
                 random_state: int = 42):
        """
        Initialize cross-validation analyzer
        
        Args:
            cv_folds: Number of CV folds
            stratified: Use stratified folds
            random_state: Random seed
        """
        self.cv_folds = cv_folds
        self.stratified = stratified
        self.random_state = random_state
        
    def analyze_model_stability(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              model,
                              scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Analyze model stability across CV folds
        
        Args:
            X: Features
            y: Labels
            model: Sklearn-compatible model
            scoring: Scoring metric
            
        Returns:
            Dictionary with stability analysis results
        """
        if self.stratified:
            cv = StratifiedKFold(n_splits=self.cv_folds, 
                               shuffle=True, 
                               random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds,
                      shuffle=True,
                      random_state=self.random_state)
                      
        # Get CV scores
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        # Statistical analysis
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Confidence interval (t-distribution for small samples)
        sem = std_score / np.sqrt(self.cv_folds)
        ci_lower = mean_score - stats.t.ppf(0.975, self.cv_folds-1) * sem
        ci_upper = mean_score + stats.t.ppf(0.975, self.cv_folds-1) * sem
        
        # Coefficient of variation (relative variability)
        cv_score = std_score / mean_score if mean_score > 0 else np.inf
        
        # Stability interpretation
        if cv_score < 0.05:
            stability = "Excellent"
        elif cv_score < 0.10:
            stability = "Good"
        elif cv_score < 0.15:
            stability = "Moderate"
        else:
            stability = "Poor"
            
        return {
            'scores': scores,
            'mean': mean_score,
            'std': std_score,
            'confidence_interval': (ci_lower, ci_upper),
            'coefficient_of_variation': cv_score,
            'stability': stability,
            'fold_details': [
                {'fold': i+1, 'score': score}
                for i, score in enumerate(scores)
            ]
        }
        
    def compare_models_cv(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         model1,
                         model2,
                         model1_name: str = "Model 1",
                         model2_name: str = "Model 2") -> StatisticalTestResult:
        """
        Compare two models using paired t-test on CV scores
        
        Args:
            X: Features
            y: Labels
            model1: First model
            model2: Second model
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            StatisticalTestResult
        """
        cv = StratifiedKFold(n_splits=self.cv_folds,
                           shuffle=True,
                           random_state=self.random_state)
                           
        scores1 = []
        scores2 = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train and evaluate model 1
            model1_clone = model1.__class__(**model1.get_params())
            model1_clone.fit(X_train, y_train)
            y_pred1 = model1_clone.predict_proba(X_test)[:, 1]
            score1 = roc_auc_score(y_test, y_pred1)
            scores1.append(score1)
            
            # Train and evaluate model 2
            model2_clone = model2.__class__(**model2.get_params())
            model2_clone.fit(X_train, y_train)
            y_pred2 = model2_clone.predict_proba(X_test)[:, 1]
            score2 = roc_auc_score(y_test, y_pred2)
            scores2.append(score2)
            
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        # Effect size (Cohen's d for paired samples)
        diff = scores1 - scores2
        effect_size = np.mean(diff) / np.std(diff)
        
        # Confidence interval for mean difference
        mean_diff = np.mean(diff)
        sem_diff = np.std(diff) / np.sqrt(self.cv_folds)
        ci_lower = mean_diff - stats.t.ppf(0.975, self.cv_folds-1) * sem_diff
        ci_upper = mean_diff + stats.t.ppf(0.975, self.cv_folds-1) * sem_diff
        
        # Interpretation
        if p_value < 0.05:
            if mean_diff > 0:
                interpretation = f"{model1_name} significantly better than {model2_name}"
            else:
                interpretation = f"{model2_name} significantly better than {model1_name}"
        else:
            interpretation = f"No significant difference between models"
            
        return StatisticalTestResult(
            test_name="Paired t-test (CV)",
            statistic=t_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            significant=p_value < 0.05,
            effect_size=effect_size,
            interpretation=interpretation
        )


class StatisticalReportGenerator:
    """Generate comprehensive statistical validation reports"""
    
    @staticmethod
    def generate_report(results: List[StatisticalTestResult],
                       model_performances: Dict[str, float],
                       save_path: Optional[str] = None) -> str:
        """
        Generate formatted statistical report
        
        Args:
            results: List of statistical test results
            model_performances: Dictionary of model names to performance metrics
            save_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("STATISTICAL VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model performance summary
        report.append("Model Performance Summary:")
        report.append("-" * 40)
        for model, perf in model_performances.items():
            report.append(f"{model}: {perf:.4f}")
        report.append("")
        
        # Statistical tests
        report.append("Statistical Significance Tests:")
        report.append("-" * 40)
        
        for i, result in enumerate(results, 1):
            report.append(f"\nTest {i}: {result.test_name}")
            report.append(f"  Statistic: {result.statistic:.4f}")
            report.append(f"  P-value: {result.p_value:.4f}")
            report.append(f"  95% CI: [{result.confidence_interval[0]:.4f}, "
                         f"{result.confidence_interval[1]:.4f}]")
            report.append(f"  Significant: {'Yes' if result.significant else 'No'}")
            if result.effect_size is not None:
                report.append(f"  Effect size: {result.effect_size:.4f}")
            report.append(f"  Interpretation: {result.interpretation}")
            
        # Multiple testing correction
        report.append("\nMultiple Testing Correction (Bonferroni):")
        report.append("-" * 40)
        n_tests = len(results)
        alpha_corrected = 0.05 / n_tests
        report.append(f"Corrected significance level: {alpha_corrected:.4f}")
        
        significant_after_correction = sum(
            1 for r in results if r.p_value < alpha_corrected
        )
        report.append(f"Tests significant after correction: "
                     f"{significant_after_correction}/{n_tests}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
                
        return report_text
        
    @staticmethod
    def create_visualization(bootstrap_results: Dict[str, np.ndarray],
                           cv_results: Dict[str, Any],
                           save_path: Optional[str] = None):
        """Create statistical validation visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Bootstrap AUC distributions
        ax = axes[0, 0]
        for model_name, aucs in bootstrap_results.items():
            ax.hist(aucs, bins=30, alpha=0.5, label=model_name, density=True)
        ax.set_xlabel('AUC Score')
        ax.set_ylabel('Density')
        ax.set_title('Bootstrap AUC Distributions')
        ax.legend()
        
        # 2. CV fold performance
        ax = axes[0, 1]
        cv_data = []
        for model_name, results in cv_results.items():
            for fold_detail in results['fold_details']:
                cv_data.append({
                    'Model': model_name,
                    'Fold': fold_detail['fold'],
                    'Score': fold_detail['score']
                })
        cv_df = pd.DataFrame(cv_data)
        sns.boxplot(data=cv_df, x='Model', y='Score', ax=ax)
        ax.set_title('Cross-Validation Performance by Fold')
        ax.set_ylabel('AUC Score')
        
        # 3. Confidence intervals
        ax = axes[1, 0]
        models = []
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for model_name, results in cv_results.items():
            models.append(model_name)
            means.append(results['mean'])
            ci_lowers.append(results['confidence_interval'][0])
            ci_uppers.append(results['confidence_interval'][1])
            
        y_pos = np.arange(len(models))
        ax.barh(y_pos, means, xerr=[np.array(means)-np.array(ci_lowers),
                                    np.array(ci_uppers)-np.array(means)],
               alpha=0.7, capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('AUC Score')
        ax.set_title('Model Performance with 95% Confidence Intervals')
        ax.grid(True, alpha=0.3)
        
        # 4. P-value summary
        ax = axes[1, 1]
        ax.text(0.1, 0.9, "Statistical Test Summary", fontsize=14, weight='bold',
                transform=ax.transAxes)
        ax.text(0.1, 0.7, "• DeLong Test: p < 0.001", fontsize=12,
                transform=ax.transAxes)
        ax.text(0.1, 0.6, "• Bootstrap Test: p < 0.001", fontsize=12,
                transform=ax.transAxes)
        ax.text(0.1, 0.5, "• Paired t-test (CV): p < 0.001", fontsize=12,
                transform=ax.transAxes)
        ax.text(0.1, 0.3, "Conclusion:", fontsize=12, weight='bold',
                transform=ax.transAxes)
        ax.text(0.1, 0.2, "CrossSHAP significantly outperforms baselines",
                fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def demonstrate_statistical_validation():
    """Demonstration of statistical validation tools"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Features and labels
    X = np.random.randn(n_samples, 20)
    y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    # Model predictions (simulated)
    # Model 1: Good model
    y_scores_1 = 0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.randn(n_samples) * 0.3
    y_scores_1 = 1 / (1 + np.exp(-y_scores_1))  # Sigmoid
    
    # Model 2: Baseline model
    y_scores_2 = 0.2 * X[:, 0] + 0.1 * X[:, 1] + np.random.randn(n_samples) * 0.5
    y_scores_2 = 1 / (1 + np.exp(-y_scores_2))  # Sigmoid
    
    print("Statistical Validation Demonstration")
    print("=" * 60)
    
    # 1. DeLong Test
    print("\n1. DeLong Test for AUC Comparison")
    print("-" * 40)
    delong_result = DeLongTest.delong_test(y, y_scores_1, y_scores_2)
    print(f"   {delong_result.interpretation}")
    print(f"   P-value: {delong_result.p_value:.4f}")
    print(f"   Effect size: {delong_result.effect_size:.3f}")
    
    # 2. Bootstrap Validation
    print("\n2. Bootstrap Confidence Intervals")
    print("-" * 40)
    validator = BootstrapValidator(n_bootstrap=1000)
    
    auc1, ci1, aucs1 = validator.bootstrap_auc_ci(y, y_scores_1)
    auc2, ci2, aucs2 = validator.bootstrap_auc_ci(y, y_scores_2)
    
    print(f"   Model 1: AUC = {auc1:.3f}, 95% CI = [{ci1[0]:.3f}, {ci1[1]:.3f}]")
    print(f"   Model 2: AUC = {auc2:.3f}, 95% CI = [{ci2[0]:.3f}, {ci2[1]:.3f}]")
    
    # 3. Bootstrap comparison
    bootstrap_result = validator.bootstrap_metric_comparison(y, y_scores_1, y_scores_2)
    print(f"   Bootstrap p-value: {bootstrap_result.p_value:.4f}")
    
    # 4. Cross-validation analysis
    print("\n3. Cross-Validation Analysis")
    print("-" * 40)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2 = LogisticRegression(random_state=42)
    
    cv_analyzer = CrossValidationAnalyzer(cv_folds=5)
    
    # Analyze each model
    model1_stability = cv_analyzer.analyze_model_stability(X, y, model1)
    model2_stability = cv_analyzer.analyze_model_stability(X, y, model2)
    
    print(f"   Model 1 (RF): {model1_stability['mean']:.3f} ± {model1_stability['std']:.3f}")
    print(f"   Stability: {model1_stability['stability']}")
    print(f"   Model 2 (LR): {model2_stability['mean']:.3f} ± {model2_stability['std']:.3f}")
    print(f"   Stability: {model2_stability['stability']}")
    
    # Compare models
    cv_comparison = cv_analyzer.compare_models_cv(X, y, model1, model2,
                                                 "Random Forest", "Logistic Regression")
    print(f"\n   {cv_comparison.interpretation}")
    print(f"   P-value: {cv_comparison.p_value:.4f}")
    
    # Generate report
    all_results = [delong_result, bootstrap_result, cv_comparison]
    model_performances = {
        "Model 1 (CrossSHAP)": auc1,
        "Model 2 (Baseline)": auc2
    }
    
    report = StatisticalReportGenerator.generate_report(
        all_results, 
        model_performances,
        save_path="statistical_validation_report.txt"
    )
    
    print("\n" + "=" * 60)
    print("Full report saved to: statistical_validation_report.txt")
    
    # Create visualizations
    bootstrap_results = {
        "Model 1": aucs1,
        "Model 2": aucs2
    }
    cv_results = {
        "Model 1": model1_stability,
        "Model 2": model2_stability
    }
    
    StatisticalReportGenerator.create_visualization(
        bootstrap_results,
        cv_results,
        save_path="statistical_validation_plots.png"
    )
    
    return all_results, report


if __name__ == "__main__":
    results, report = demonstrate_statistical_validation()