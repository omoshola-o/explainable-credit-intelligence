#!/usr/bin/env python3
"""
CrossSHAP Algorithm Implementation
==================================
This module implements the CrossSHAP algorithm for cross-domain 
Shapley value attribution as described in the paper.

Author: Omoshola S. Owolabi
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
import shap
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrossSHAPConfig:
    """Configuration for CrossSHAP algorithm"""
    n_samples: int = 100
    interaction_threshold: float = 0.1
    normalize_values: bool = True
    compute_interactions: bool = True
    weight_learning_epochs: int = 50
    weight_learning_lr: float = 0.001
    

class CrossDomainWeightLearner(nn.Module):
    """Neural network for learning cross-domain interaction weights"""
    
    def __init__(self, corp_dim: int, retail_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.corp_dim = corp_dim
        self.retail_dim = retail_dim
        
        # Cross-domain attention mechanism
        self.corp_encoder = nn.Sequential(
            nn.Linear(corp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.retail_encoder = nn.Sequential(
            nn.Linear(retail_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Interaction learning
        self.interaction_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, corp_dim * retail_dim),
            nn.Sigmoid()  # Weights between 0 and 1
        )
        
    def forward(self, X_corp: torch.Tensor, X_retail: torch.Tensor) -> torch.Tensor:
        """Compute cross-domain interaction weights"""
        # Encode domains
        corp_encoded = self.corp_encoder(X_corp)
        retail_encoded = self.retail_encoder(X_retail)
        
        # Concatenate and compute interactions
        combined = torch.cat([corp_encoded, retail_encoded], dim=-1)
        weights = self.interaction_net(combined)
        
        # Reshape to interaction matrix
        batch_size = X_corp.shape[0]
        weights = weights.view(batch_size, self.corp_dim, self.retail_dim)
        
        return weights


class CrossSHAP:
    """
    CrossSHAP: Cross-Domain Shapley Value Attribution
    
    This class implements the CrossSHAP algorithm for computing
    cross-domain feature attributions in unified credit risk models.
    """
    
    def __init__(self, 
                 corporate_model: BaseEstimator,
                 retail_model: BaseEstimator,
                 unified_model: BaseEstimator,
                 config: Optional[CrossSHAPConfig] = None):
        """
        Initialize CrossSHAP with domain-specific and unified models
        
        Args:
            corporate_model: Trained model for corporate domain
            retail_model: Trained model for retail domain
            unified_model: Unified model using both domains
            config: Configuration parameters
        """
        self.corp_model = corporate_model
        self.retail_model = retail_model
        self.unified_model = unified_model
        self.config = config or CrossSHAPConfig()
        
        # Initialize SHAP explainers
        self.corp_explainer = None
        self.retail_explainer = None
        self.unified_explainer = None
        
        # Cross-domain weight learner
        self.weight_learner = None
        
        # Store computed values
        self.corp_shap_values = None
        self.retail_shap_values = None
        self.cross_shap_values = None
        self.interaction_matrix = None
        
    def fit(self, 
            X_corp: np.ndarray, 
            X_retail: np.ndarray,
            y: np.ndarray) -> 'CrossSHAP':
        """
        Fit CrossSHAP on training data
        
        Args:
            X_corp: Corporate features (n_samples, n_corp_features)
            X_retail: Retail features (n_samples, n_retail_features)
            y: Target labels (n_samples,)
        """
        logger.info("Fitting CrossSHAP algorithm...")
        
        # Step 1: Initialize SHAP explainers
        logger.info("Step 1: Initializing SHAP explainers")
        self._initialize_explainers(X_corp, X_retail)
        
        # Step 2: Compute individual domain SHAP values
        logger.info("Step 2: Computing domain-specific SHAP values")
        self.corp_shap_values = self.corp_explainer.shap_values(X_corp)
        self.retail_shap_values = self.retail_explainer.shap_values(X_retail)
        
        # Step 3: Learn cross-domain weights
        logger.info("Step 3: Learning cross-domain interaction weights")
        self._learn_cross_weights(X_corp, X_retail, y)
        
        # Step 4: Compute interaction matrix
        logger.info("Step 4: Computing interaction matrix")
        self.interaction_matrix = self._compute_interaction_matrix(X_corp, X_retail)
        
        logger.info("CrossSHAP fitting complete")
        return self
        
    def explain(self, 
                X_corp: np.ndarray, 
                X_retail: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute CrossSHAP explanations for given instances
        
        Args:
            X_corp: Corporate features to explain
            X_retail: Retail features to explain
            
        Returns:
            Dictionary containing:
                - 'corporate': Corporate SHAP values
                - 'retail': Retail SHAP values
                - 'cross': Cross-domain SHAP values
                - 'unified': Unified SHAP values
        """
        # Compute individual SHAP values
        corp_shap = self.corp_explainer.shap_values(X_corp)
        retail_shap = self.retail_explainer.shap_values(X_retail)
        
        # Compute cross-domain values
        cross_shap = self._compute_cross_shap(
            corp_shap, retail_shap, X_corp, X_retail
        )
        
        # Compute unified SHAP values
        X_unified = np.hstack([X_corp, X_retail])
        unified_shap = self.unified_explainer.shap_values(X_unified)
        
        # Normalize if configured
        if self.config.normalize_values:
            cross_shap = self._normalize_shap_values(cross_shap)
            
        return {
            'corporate': corp_shap,
            'retail': retail_shap,
            'cross': cross_shap,
            'unified': unified_shap
        }
        
    def _initialize_explainers(self, X_corp: np.ndarray, X_retail: np.ndarray):
        """Initialize SHAP explainers for each model"""
        # Sample background data
        n_background = min(100, len(X_corp))
        idx = np.random.choice(len(X_corp), n_background, replace=False)
        
        # Corporate explainer
        background_corp = X_corp[idx]
        self.corp_explainer = shap.Explainer(
            self.corp_model.predict, background_corp
        )
        
        # Retail explainer
        background_retail = X_retail[idx]
        self.retail_explainer = shap.Explainer(
            self.retail_model.predict, background_retail
        )
        
        # Unified explainer
        background_unified = np.hstack([background_corp, background_retail])
        self.unified_explainer = shap.Explainer(
            self.unified_model.predict, background_unified
        )
        
    def _learn_cross_weights(self, 
                           X_corp: np.ndarray, 
                           X_retail: np.ndarray,
                           y: np.ndarray):
        """Learn cross-domain interaction weights using neural network"""
        n_corp_features = X_corp.shape[1]
        n_retail_features = X_retail.shape[1]
        
        # Initialize weight learner
        self.weight_learner = CrossDomainWeightLearner(
            n_corp_features, n_retail_features
        )
        
        # Convert to tensors
        X_corp_tensor = torch.FloatTensor(X_corp)
        X_retail_tensor = torch.FloatTensor(X_retail)
        y_tensor = torch.FloatTensor(y)
        
        # Training setup
        optimizer = torch.optim.Adam(
            self.weight_learner.parameters(), 
            lr=self.config.weight_learning_lr
        )
        criterion = nn.MSELoss()
        
        # Training loop
        self.weight_learner.train()
        for epoch in range(self.config.weight_learning_epochs):
            optimizer.zero_grad()
            
            # Get interaction weights
            weights = self.weight_learner(X_corp_tensor, X_retail_tensor)
            
            # Compute weighted predictions
            corp_contrib = X_corp_tensor.unsqueeze(2)
            retail_contrib = X_retail_tensor.unsqueeze(1)
            interactions = (corp_contrib * retail_contrib * weights).sum(dim=[1, 2])
            
            # Loss based on prediction accuracy
            X_unified = torch.cat([X_corp_tensor, X_retail_tensor], dim=1)
            unified_pred = torch.FloatTensor(
                self.unified_model.predict_proba(X_unified.numpy())[:, 1]
            )
            
            loss = criterion(interactions, unified_pred - y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                
        self.weight_learner.eval()
        
    def _compute_interaction_matrix(self, 
                                  X_corp: np.ndarray, 
                                  X_retail: np.ndarray) -> np.ndarray:
        """Compute the interaction matrix between corporate and retail features"""
        with torch.no_grad():
            X_corp_tensor = torch.FloatTensor(X_corp)
            X_retail_tensor = torch.FloatTensor(X_retail)
            
            # Get learned weights
            weights = self.weight_learner(X_corp_tensor, X_retail_tensor)
            interaction_matrix = weights.mean(dim=0).numpy()
            
        # Apply threshold
        interaction_matrix[interaction_matrix < self.config.interaction_threshold] = 0
        
        return interaction_matrix
        
    def _compute_cross_shap(self,
                           corp_shap: np.ndarray,
                           retail_shap: np.ndarray,
                           X_corp: np.ndarray,
                           X_retail: np.ndarray) -> np.ndarray:
        """
        Compute cross-domain SHAP values using the CrossSHAP algorithm
        
        This implements the core CrossSHAP computation:
        φ_cross[i,j] = α[i,j] * (φ_c[i] ⊗ φ_r[j]) * W[i,j]
        """
        n_samples = corp_shap.shape[0]
        n_corp_features = corp_shap.shape[1]
        n_retail_features = retail_shap.shape[1]
        
        # Get interaction weights for these samples
        with torch.no_grad():
            X_corp_tensor = torch.FloatTensor(X_corp)
            X_retail_tensor = torch.FloatTensor(X_retail)
            alpha = self.weight_learner(X_corp_tensor, X_retail_tensor).numpy()
            
        # Initialize cross-SHAP values
        cross_shap = np.zeros((n_samples, n_corp_features * n_retail_features))
        
        # Compute cross-domain attributions
        for i in range(n_samples):
            # Outer product of SHAP values
            shap_outer = np.outer(corp_shap[i], retail_shap[i])
            
            # Apply learned weights and interaction matrix
            weighted_interaction = shap_outer * alpha[i] * self.interaction_matrix
            
            # Flatten to feature vector
            cross_shap[i] = weighted_interaction.flatten()
            
        return cross_shap
        
    def _normalize_shap_values(self, shap_values: np.ndarray) -> np.ndarray:
        """Normalize SHAP values to ensure efficiency property"""
        # Ensure sum of SHAP values equals model output - expected value
        normalized = shap_values.copy()
        
        for i in range(len(normalized)):
            total = normalized[i].sum()
            if abs(total) > 1e-6:  # Avoid division by zero
                normalized[i] = normalized[i] / total
                
        return normalized
        
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance rankings from CrossSHAP analysis
        
        Returns:
            Dictionary of DataFrames with feature importance metrics
        """
        if self.cross_shap_values is None:
            raise ValueError("Must call fit() before getting feature importance")
            
        # Corporate feature importance
        corp_importance = np.abs(self.corp_shap_values).mean(axis=0)
        corp_df = pd.DataFrame({
            'feature': [f'corp_feat_{i}' for i in range(len(corp_importance))],
            'importance': corp_importance,
            'domain': 'corporate'
        }).sort_values('importance', ascending=False)
        
        # Retail feature importance
        retail_importance = np.abs(self.retail_shap_values).mean(axis=0)
        retail_df = pd.DataFrame({
            'feature': [f'retail_feat_{i}' for i in range(len(retail_importance))],
            'importance': retail_importance,
            'domain': 'retail'
        }).sort_values('importance', ascending=False)
        
        # Cross-domain importance
        cross_importance = np.abs(self.cross_shap_values).mean(axis=0)
        n_corp = self.corp_shap_values.shape[1]
        n_retail = self.retail_shap_values.shape[1]
        
        cross_features = []
        for i in range(n_corp):
            for j in range(n_retail):
                cross_features.append(f'corp_{i}_x_retail_{j}')
                
        cross_df = pd.DataFrame({
            'feature': cross_features,
            'importance': cross_importance,
            'domain': 'cross'
        }).sort_values('importance', ascending=False)
        
        return {
            'corporate': corp_df,
            'retail': retail_df,
            'cross': cross_df
        }
        
    def visualize_interactions(self, top_k: int = 20) -> Dict[str, Any]:
        """
        Create visualization data for top cross-domain interactions
        
        Args:
            top_k: Number of top interactions to visualize
            
        Returns:
            Dictionary with visualization data
        """
        if self.interaction_matrix is None:
            raise ValueError("Must call fit() before visualizing")
            
        # Get top interactions
        interaction_flat = self.interaction_matrix.flatten()
        top_indices = np.argsort(interaction_flat)[-top_k:][::-1]
        
        # Convert to feature pairs
        n_retail = self.interaction_matrix.shape[1]
        interactions = []
        
        for idx in top_indices:
            corp_idx = idx // n_retail
            retail_idx = idx % n_retail
            strength = interaction_flat[idx]
            
            interactions.append({
                'corporate_feature': f'corp_feat_{corp_idx}',
                'retail_feature': f'retail_feat_{retail_idx}',
                'strength': strength,
                'corp_idx': corp_idx,
                'retail_idx': retail_idx
            })
            
        return {
            'top_interactions': interactions,
            'interaction_matrix': self.interaction_matrix,
            'total_features': {
                'corporate': self.interaction_matrix.shape[0],
                'retail': self.interaction_matrix.shape[1]
            }
        }


def demonstrate_crossshap():
    """Demonstration of CrossSHAP algorithm usage"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_corp_features = 10
    n_retail_features = 8
    
    X_corp = np.random.randn(n_samples, n_corp_features)
    X_retail = np.random.randn(n_samples, n_retail_features)
    
    # Create synthetic models (placeholders)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Generate synthetic target
    y = (X_corp[:, 0] + X_retail[:, 0] + 
         0.5 * X_corp[:, 1] * X_retail[:, 1] + 
         np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Train domain models
    corp_model = RandomForestClassifier(n_estimators=100, random_state=42)
    corp_model.fit(X_corp, y)
    
    retail_model = RandomForestClassifier(n_estimators=100, random_state=42)
    retail_model.fit(X_retail, y)
    
    # Train unified model
    X_unified = np.hstack([X_corp, X_retail])
    unified_model = LogisticRegression(random_state=42)
    unified_model.fit(X_unified, y)
    
    # Initialize and fit CrossSHAP
    crossshap = CrossSHAP(corp_model, retail_model, unified_model)
    crossshap.fit(X_corp[:800], X_retail[:800], y[:800])
    
    # Explain test instances
    explanations = crossshap.explain(X_corp[800:810], X_retail[800:810])
    
    print("CrossSHAP Demonstration Results:")
    print(f"Corporate SHAP shape: {explanations['corporate'].shape}")
    print(f"Retail SHAP shape: {explanations['retail'].shape}")
    print(f"Cross-domain SHAP shape: {explanations['cross'].shape}")
    print(f"Unified SHAP shape: {explanations['unified'].shape}")
    
    # Get feature importance
    importance = crossshap.get_feature_importance()
    print("\nTop 5 Corporate Features:")
    print(importance['corporate'].head())
    
    print("\nTop 5 Cross-Domain Interactions:")
    print(importance['cross'].head())
    
    # Visualize interactions
    viz_data = crossshap.visualize_interactions(top_k=10)
    print("\nTop 3 Cross-Domain Interactions:")
    for interaction in viz_data['top_interactions'][:3]:
        print(f"  {interaction['corporate_feature']} <-> "
              f"{interaction['retail_feature']}: "
              f"{interaction['strength']:.3f}")


if __name__ == "__main__":
    demonstrate_crossshap()