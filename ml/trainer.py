import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
from config import config
import os

logger = logging.getLogger(__name__)

class MLTrainer:
    """Machine Learning model trainer for price prediction"""
    
    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize ML trainer
        
        Args:
            model_type: Type of model ('logistic', 'decision_tree', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.training_metrics = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
        
        logger.info(f"Initialized MLTrainer with model type: {model_type}")
    
    def create_model(self) -> Any:
        """
        Create ML model based on specified type
        
        Returns:
            Initialized model
        """
        if self.model_type == 'logistic':
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'  # Handle class imbalance
            )
        elif self.model_type == 'decision_tree':
            model = DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced'
            )
        else:
            logger.warning(f"Unknown model type {self.model_type}, using logistic regression")
            model = LogisticRegression(random_state=42, max_iter=1000)
        
        return model
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = None) -> Dict[str, Any]:
        """
        Train the ML model with time series aware split
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Test split ratio (from config if None)
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            test_size = test_size or (1 - config.TRAIN_TEST_SPLIT_RATIO)
            
            # Time series aware split
            split_idx = int(len(X) * config.TRAIN_TEST_SPLIT_RATIO)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train model
            self.model = self.create_model()
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Get prediction probabilities
            y_train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
            y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba, "Training")
            test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba, "Testing")
            
            # Cross validation
            cv_scores = self._cross_validate(X_train_scaled, y_train)
            
            # Feature importance
            feature_importance = self._get_feature_importance()
            
            # Store training results
            self.training_metrics = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'feature_importance': feature_importance,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logger.info(f"Model training completed. Test accuracy: {test_metrics['accuracy']:.3f}")
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {}
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_proba: np.ndarray, dataset_name: str) -> Dict[str, Any]:
        """Calculate model performance metrics"""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_proba),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            
            # Calculate additional metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            logger.info(f"{dataset_name} Metrics - Accuracy: {metrics['accuracy']:.3f}, "
                       f"ROC-AUC: {metrics['roc_auc']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {dataset_name}: {str(e)}")
            return {}
    
    def _cross_validate(self, X: np.ndarray, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """Perform time series cross validation"""
        try:
            # Use TimeSeriesSplit for proper time series validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            cv_scores = cross_val_score(self.model, X, y, cv=tscv, scoring='accuracy')
            cv_roc_scores = cross_val_score(self.model, X, y, cv=tscv, scoring='roc_auc')
            
            cv_metrics = {
                'accuracy_mean': cv_scores.mean(),
                'accuracy_std': cv_scores.std(),
                'roc_auc_mean': cv_roc_scores.mean(),
                'roc_auc_std': cv_roc_scores.std(),
                'individual_scores': cv_scores.tolist()
            }
            
            logger.info(f"Cross-validation accuracy: {cv_metrics['accuracy_mean']:.3f} "
                       f"(±{cv_metrics['accuracy_std']:.3f})")
            
            return cv_metrics
            
        except Exception as e:
            logger.error(f"Error in cross validation: {str(e)}")
            return {}
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based models
                importance_values = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # Linear models
                importance_values = abs(self.model.coef_[0])
            else:
                logger.warning("Model doesn't support feature importance")
                return {}
            
            # Create feature importance dictionary
            feature_importance = dict(zip(self.feature_names, importance_values))
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            # Log top features
            top_features = list(feature_importance.keys())[:5]
            logger.info(f"Top 5 features: {top_features}")
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def save_model(self, filepath: str = None) -> bool:
        """
        Save the trained model and scaler
        
        Args:
            filepath: Path to save the model (default from config)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            filepath = filepath or config.MODEL_SAVE_PATH
            
            if self.model is None or self.scaler is None:
                logger.error("No trained model to save")
                return False
            
            # Create model package
            model_package = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'training_metrics': self.training_metrics
            }
            
            # Save using joblib
            joblib.dump(model_package, filepath)
            logger.info(f"Model saved to {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str = None) -> bool:
        """
        Load a saved model
        
        Args:
            filepath: Path to load the model from (default from config)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            filepath = filepath or config.MODEL_SAVE_PATH
            
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return False
            
            # Load model package
            model_package = joblib.load(filepath)
            
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.model_type = model_package['model_type']
            self.training_metrics = model_package.get('training_metrics', {})
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the trained model"""
        if not self.training_metrics:
            return {}
        
        summary = {
            'model_type': self.model_type,
            'training_samples': self.training_metrics.get('training_samples', 0),
            'test_samples': self.training_metrics.get('test_samples', 0),
            'test_accuracy': self.training_metrics.get('test_metrics', {}).get('accuracy', 0),
            'test_roc_auc': self.training_metrics.get('test_metrics', {}).get('roc_auc', 0),
            'cv_accuracy_mean': self.training_metrics.get('cv_scores', {}).get('accuracy_mean', 0),
            'feature_count': len(self.feature_names),
            'top_features': list(self.training_metrics.get('feature_importance', {}).keys())[:5]
        }
        
        return summary
