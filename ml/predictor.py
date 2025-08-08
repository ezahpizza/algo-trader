import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from ml.trainer import MLTrainer
from config import config

logger = logging.getLogger(__name__)

class MLPredictor:
    """ML model predictor for next-day price movement"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize ML predictor
        
        Args:
            model_path: Path to saved model (will load if exists)
        """
        self.trainer = MLTrainer(model_type=config.MODEL_TYPE)
        self.model_loaded = False
        
        if model_path and self.trainer.load_model(model_path):
            self.model_loaded = True
            logger.info("Model loaded successfully")
        else:
            logger.info("No model loaded, will need training")
    
    def predict_next_day_movement(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict next day price movement
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.model_loaded:
                logger.error("No model loaded for prediction")
                return {}
            
            # Get the latest features (most recent row)
            latest_features = features.iloc[-1:][self.trainer.feature_names]
            
            # Handle missing features
            latest_features = latest_features.fillna(0)
            
            # Scale features
            features_scaled = self.trainer.scaler.transform(latest_features)
            
            # Make prediction
            prediction = self.trainer.model.predict(features_scaled)[0]
            prediction_proba = self.trainer.model.predict_proba(features_scaled)[0]
            
            # Get confidence (probability of predicted class)
            confidence = prediction_proba[prediction]
            probability_positive = prediction_proba[1]  # Probability of positive movement
            
            prediction_result = {
                'prediction': int(prediction),
                'prediction_label': 'UP' if prediction == 1 else 'DOWN',
                'confidence': float(confidence),
                'probability_positive': float(probability_positive),
                'probability_negative': float(prediction_proba[0]),
                'timestamp': features.index[-1],
                'features_used': len(self.trainer.feature_names)
            }
            
            logger.info(f"Prediction: {prediction_result['prediction_label']} "
                       f"(confidence: {confidence:.3f})")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {}
    
    def predict_batch(self, features: pd.DataFrame, 
                     start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Make predictions for a batch of data
        
        Args:
            features: DataFrame with engineered features
            start_date: Start date for predictions
            end_date: End date for predictions
            
        Returns:
            DataFrame with predictions
        """
        try:
            if not self.model_loaded:
                logger.error("No model loaded for prediction")
                return pd.DataFrame()
            
            # Filter data by date range if provided
            if start_date or end_date:
                if start_date:
                    features = features[features.index >= start_date]
                if end_date:
                    features = features[features.index <= end_date]
            
            # Prepare features
            feature_data = features[self.trainer.feature_names].fillna(0)
            
            # Scale features
            features_scaled = self.trainer.scaler.transform(feature_data)
            
            # Make predictions
            predictions = self.trainer.model.predict(features_scaled)
            prediction_probas = self.trainer.model.predict_proba(features_scaled)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'prediction': predictions,
                'prediction_label': ['UP' if p == 1 else 'DOWN' for p in predictions],
                'confidence': [max(proba) for proba in prediction_probas],
                'probability_positive': prediction_probas[:, 1],
                'probability_negative': prediction_probas[:, 0]
            }, index=features.index)
            
            logger.info(f"Made predictions for {len(results)} data points")
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            return pd.DataFrame()
    
    def evaluate_predictions(self, features: pd.DataFrame, 
                           actual_returns: pd.Series) -> Dict[str, Any]:
        """
        Evaluate prediction accuracy against actual returns
        
        Args:
            features: DataFrame with engineered features
            actual_returns: Series with actual next-day returns
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.model_loaded:
                logger.error("No model loaded for evaluation")
                return {}
            
            # Make predictions
            predictions_df = self.predict_batch(features)
            
            if predictions_df.empty:
                return {}
            
            # Align predictions with actual returns
            aligned_data = pd.concat([predictions_df, actual_returns], axis=1, join='inner')
            aligned_data.columns = list(predictions_df.columns) + ['actual_return']
            
            # Create actual binary labels
            aligned_data['actual_label'] = (aligned_data['actual_return'] > 0).astype(int)
            
            # Calculate metrics
            accuracy = (aligned_data['prediction'] == aligned_data['actual_label']).mean()
            
            # Directional accuracy for different confidence levels
            confidence_thresholds = [0.5, 0.6, 0.7, 0.8]
            confidence_metrics = {}
            
            for threshold in confidence_thresholds:
                high_confidence = aligned_data[aligned_data['confidence'] >= threshold]
                if len(high_confidence) > 0:
                    conf_accuracy = (high_confidence['prediction'] == 
                                   high_confidence['actual_label']).mean()
                    confidence_metrics[f'accuracy_conf_{threshold}'] = conf_accuracy
                    confidence_metrics[f'samples_conf_{threshold}'] = len(high_confidence)
            
            # Calculate returns if following predictions
            strategy_returns = []
            for idx, row in aligned_data.iterrows():
                if row['prediction'] == 1:  # Predicted UP
                    strategy_returns.append(row['actual_return'])
                else:  # Predicted DOWN - short or stay out
                    strategy_returns.append(-row['actual_return'])  # Assuming short
            
            strategy_cumulative_return = np.sum(strategy_returns)
            
            evaluation = {
                'overall_accuracy': accuracy,
                'total_predictions': len(aligned_data),
                'correct_predictions': int((aligned_data['prediction'] == 
                                          aligned_data['actual_label']).sum()),
                'strategy_cumulative_return': strategy_cumulative_return,
                'average_confidence': aligned_data['confidence'].mean(),
                'confidence_metrics': confidence_metrics,
                'prediction_distribution': aligned_data['prediction'].value_counts().to_dict()
            }
            
            logger.info(f"Evaluation complete. Accuracy: {accuracy:.3f}, "
                       f"Total predictions: {len(aligned_data)}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {str(e)}")
            return {}
    
    def get_feature_contributions(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature contributions for the latest prediction
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Dictionary with feature contributions
        """
        try:
            if not self.model_loaded:
                logger.error("No model loaded")
                return {}
            
            # Get latest features
            latest_features = features.iloc[-1:][self.trainer.feature_names].fillna(0)
            
            # Get feature importance from model
            feature_importance = self.trainer.training_metrics.get('feature_importance', {})
            
            # Calculate weighted contributions
            contributions = {}
            for feature in self.trainer.feature_names:
                feature_value = latest_features[feature].iloc[0]
                importance = feature_importance.get(feature, 0)
                contributions[feature] = float(feature_value * importance)
            
            # Sort by absolute contribution
            contributions = dict(sorted(contributions.items(), 
                                      key=lambda x: abs(x[1]), reverse=True))
            
            return contributions
            
        except Exception as e:
            logger.error(f"Error getting feature contributions: {str(e)}")
            return {}
    
    def is_model_ready(self) -> bool:
        """Check if model is loaded and ready for predictions"""
        return self.model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model_loaded:
            return {'status': 'No model loaded'}
        
        return {
            'status': 'Model loaded',
            'model_type': self.trainer.model_type,
            'feature_count': len(self.trainer.feature_names),
            'training_metrics': self.trainer.training_metrics.get('test_metrics', {}),
            'features': self.trainer.feature_names
        }
