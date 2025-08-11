"""
Machine Learning Model Module
Implements ML models for predicting next-day stock movements
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from indicators import add_all_indicators
from utils import setup_logging

# Setup logging
logger = setup_logging(__name__)

class MLPredictor:
    """
    Machine Learning predictor for stock price movements
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize ML predictor
        
        Args:
            model_type: Type of model ('decision_tree', 'logistic_regression', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"ML Predictor initialized with model type: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with prepared features
        """
        try:
            logger.info("Preparing features for ML model")
            
            # Ensure we have all indicators
            if 'rsi' not in df.columns:
                df = add_all_indicators(df)
            
            # Create feature DataFrame
            features_df = df.copy()
            
            # Price-based features
            features_df['price_change'] = df['close'].pct_change()
            features_df['high_low_ratio'] = df['high'] / df['low']
            features_df['close_open_ratio'] = df['close'] / df['open']
            
            # Volume features
            if 'volume' in df.columns:
                features_df['volume_ma'] = df['volume'].rolling(20).mean()
                features_df['volume_ratio'] = df['volume'] / features_df['volume_ma']
                features_df['volume_change'] = df['volume'].pct_change()
            else:
                features_df['volume_ratio'] = 1.0
                features_df['volume_change'] = 0.0
            
            # Technical indicator features
            features_df['rsi_normalized'] = features_df['rsi'] / 100
            features_df['ma_ratio'] = features_df['sma_20'] / features_df['sma_50']
            features_df['price_to_sma20'] = df['close'] / features_df['sma_20']
            features_df['price_to_sma50'] = df['close'] / features_df['sma_50']
            
            # MACD features
            features_df['macd_signal_ratio'] = features_df['macd'] / features_df['macd_signal']
            features_df['macd_histogram_normalized'] = features_df['macd_histogram'] / df['close']
            
            # Bollinger Bands features
            features_df['bb_position'] = (df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
            features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / features_df['bb_middle']
            
            # Volatility features
            features_df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            # Momentum features
            features_df['momentum_5'] = df['close'] / df['close'].shift(5)
            features_df['momentum_10'] = df['close'] / df['close'].shift(10)
            
            # Create target variable (next day movement)
            features_df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1
            features_df['target'] = (features_df['next_day_return'] > 0).astype(int)  # 1 for up, 0 for down
            
            # Select feature columns
            self.feature_columns = [
                'rsi_normalized', 'ma_ratio', 'price_to_sma20', 'price_to_sma50',
                'macd_signal_ratio', 'macd_histogram_normalized', 'bb_position', 'bb_width',
                'volatility', 'momentum_5', 'momentum_10', 'volume_ratio', 'volume_change',
                'price_change', 'high_low_ratio', 'close_open_ratio'
            ]
            
            # Handle infinite and NaN values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            logger.info(f"Features prepared. Shape: {features_df.shape}, Features: {len(self.feature_columns)}")
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return df
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train the ML model
        
        Args:
            df: DataFrame with prepared features
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info("Training ML model")
            
            # Prepare features
            features_df = self.prepare_features(df)
            
            # Remove rows with NaN targets (last row)
            features_df = features_df.dropna(subset=['target'])
            
            if len(features_df) < 50:
                logger.warning("Insufficient data for training")
                return {'error': 'Insufficient data'}
            
            # Prepare X and y
            X = features_df[self.feature_columns]
            y = features_df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Make predictions
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            
            # Feature importance (for tree-based models)
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            results = {
                'model_type': self.model_type,
                'train_accuracy': round(train_accuracy, 4),
                'test_accuracy': round(test_accuracy, 4),
                'cv_mean_accuracy': round(cv_scores.mean(), 4),
                'cv_std_accuracy': round(cv_scores.std(), 4),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_columns),
                'feature_importance': feature_importance,
                'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
            }
            
            logger.info(f"Model trained successfully. Test accuracy: {test_accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {'error': str(e)}
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Make predictions on new data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with predictions
        """
        try:
            if not self.is_trained:
                logger.error("Model is not trained yet")
                return {'error': 'Model not trained'}
            
            # Prepare features
            features_df = self.prepare_features(df)
            
            # Get the latest data point
            latest_data = features_df[self.feature_columns].iloc[-1:].values
            
            # Handle NaN values
            if np.isnan(latest_data).any():
                logger.warning("NaN values in prediction data, filling with zeros")
                latest_data = np.nan_to_num(latest_data)
            
            # Scale features
            latest_data_scaled = self.scaler.transform(latest_data)
            
            # Make prediction
            prediction = self.model.predict(latest_data_scaled)[0]
            prediction_proba = self.model.predict_proba(latest_data_scaled)[0]
            
            result = {
                'prediction': int(prediction),
                'prediction_label': 'UP' if prediction == 1 else 'DOWN',
                'confidence': float(max(prediction_proba)),
                'probability_up': float(prediction_proba[1]),
                'probability_down': float(prediction_proba[0]),
                'timestamp': df.index[-1]
            }
            
            logger.info(f"Prediction: {result['prediction_label']} with confidence {result['confidence']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        try:
            if not self.is_trained:
                logger.error("Cannot save untrained model")
                return
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

def train_models_for_multiple_stocks(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Train ML models for multiple stocks
    
    Args:
        stock_data: Dictionary with stock data
        
    Returns:
        Dictionary with training results for each stock
    """
    results = {}
    
    for symbol, df in stock_data.items():
        try:
            logger.info(f"Training model for {symbol}")
            
            # Try different model types
            model_types = ['random_forest', 'decision_tree', 'logistic_regression']
            best_model = None
            best_accuracy = 0
            best_results = None
            
            for model_type in model_types:
                predictor = MLPredictor(model_type=model_type)
                training_results = predictor.train_model(df)
                
                if 'error' not in training_results:
                    test_accuracy = training_results['test_accuracy']
                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_model = predictor
                        best_results = training_results
            
            if best_model:
                # Save the best model
                model_path = f"models/{symbol}_{best_model.model_type}_model.pkl"
                os.makedirs("models", exist_ok=True)
                best_model.save_model(model_path)
                
                # Make prediction on latest data
                prediction = best_model.predict(df)
                best_results['latest_prediction'] = prediction
                
                results[symbol] = best_results
                logger.info(f"Best model for {symbol}: {best_model.model_type} with accuracy {best_accuracy:.4f}")
            else:
                logger.warning(f"Failed to train any model for {symbol}")
                
        except Exception as e:
            logger.error(f"Error training models for {symbol}: {str(e)}")
            continue
    
    return results

# Example usage and testing
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    
    print("Testing ML models...")
    
    # Fetch sample data
    fetcher = DataFetcher()
    stock_data = fetcher.fetch_multiple_stocks(period="1y")  # More data for ML
    
    if stock_data:
        # Train models for all stocks
        ml_results = train_models_for_multiple_stocks(stock_data)
        
        print(f"\n=== ML MODEL RESULTS ===")
        for symbol, results in ml_results.items():
            print(f"\n{symbol}:")
            print(f"  Model Type: {results['model_type']}")
            print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"  CV Accuracy: {results['cv_mean_accuracy']:.4f} ± {results['cv_std_accuracy']:.4f}")
            
            if 'latest_prediction' in results:
                pred = results['latest_prediction']
                print(f"  Latest Prediction: {pred['prediction_label']} ({pred['confidence']:.3f})")
            
            if results.get('feature_importance'):
                print("  Top Features:")
                for feature, importance in list(results['feature_importance'].items())[:5]:
                    print(f"    {feature}: {importance:.4f}")
    else:
        print("No stock data available for testing")
