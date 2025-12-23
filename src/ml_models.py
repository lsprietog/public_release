import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class IVIMRegressor:
    """
    Machine Learning wrapper for estimating IVIM/DKI parameters from diffusion MRI signals.
    
    This class provides a unified interface for training and applying various regression models
    (Random Forest, Extra Trees, MLP, etc.) to map signal attenuation curves directly to 
    tissue parameters (D, f, D*, K), bypassing iterative non-linear least squares fitting.
    
    Supported architectures:
    - 'random_forest': Robust baseline, handles noise well.
    - 'extra_trees': Often faster and slightly more accurate than RF. In our experiments, this model showed superior robustness to noise.
    - 'mlp': Multi-layer Perceptron for capturing complex non-linear mappings.
    - 'xgboost': Gradient boosting (requires xgboost package).
    - 'svr': Support Vector Regression.
    """
    
    def __init__(self, model_type='extra_trees', params=None):
        self.model_type = model_type
        self.params = params if params else {}
        self.model = self._build_model()
        
    def _build_model(self):
        if self.model_type == 'random_forest':
            # Default params from paper/notebook
            n_estimators = self.params.get('n_estimators', 100)
            return RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            
        elif self.model_type == 'extra_trees':
            n_estimators = self.params.get('n_estimators', 100)
            return ExtraTreesRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            
        elif self.model_type == 'mlp':
            hidden_layer_sizes = self.params.get('hidden_layer_sizes', (100, 50))
            return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=500, random_state=42)
            
        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=-1, random_state=42)
            except ImportError:
                print("XGBoost not installed. Falling back to Random Forest.")
                return RandomForestRegressor(n_estimators=100, random_state=42)
                
        elif self.model_type == 'svr':
            C = self.params.get('C', 100)
            return SVR(C=C)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X, y, test_size=0.2, verbose=True):
        """
        Trains the regression model using the provided signal-parameter pairs.
        
        Args:
            X: Input feature matrix (Normalized Signal vs b-values). Shape: [n_samples, n_b_values]
            y: Target parameter vector (e.g., Diffusion Coefficient D). Shape: [n_samples]
            test_size: Fraction of data to reserve for validation (default: 0.2).
            verbose: If True, prints training progress and validation metrics.
            
        Returns:
            Dictionary containing validation metrics (MAE, MSE, RMSE, R2).
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if verbose:
            print(f"Training {self.model_type} on {len(X_train)} samples...")
            
        self.model.fit(X_train, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test)
        metrics = self._evaluate(y_test, predictions)
        
        if verbose:
            print("--- Validation Metrics ---")
            for k, v in metrics.items():
                print(f"{k}: {v:.6f}")
                
        return metrics

    def predict(self, X):
        """Predicts parameters for new data."""
        return self.model.predict(X)
    
    def _evaluate(self, y_true, y_pred):
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
    
    def save(self, filepath):
        """Saves the trained model to disk."""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
        
    def load(self, filepath):
        """Loads a trained model from disk."""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")

def load_training_data(data_dir, dataset_name='MR701'):
    """
    Helper to load X and Y CSV files from the data directory.
    Expected format: Data_X2_{dataset}.csv and Data_Y_{dataset}.csv
    """
    x_path = os.path.join(data_dir, f'Data_X2_{dataset_name}.csv')
    y_path = os.path.join(data_dir, f'Data_Y_{dataset_name}.csv')
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files not found for {dataset_name} in {data_dir}")
        
    X = np.loadtxt(x_path)
    Y = np.loadtxt(y_path) # Assuming Y contains [D, f, D*, K] columns or similar
    
    return X, Y
