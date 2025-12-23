import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ml_models import IVIMRegressor
from ivim_model import calculate_ivim_params

def generate_synthetic_training_data(n_samples=1000):
    """
    Generates synthetic data for training ML models.
    """
    print(f"Generating {n_samples} synthetic samples...")
    
    # b-values
    b_values = np.array([0, 10, 20, 30, 50, 80, 100, 200, 400, 800, 1000])
    
    # Random parameters
    # D: 0.0005 to 0.003
    D = np.random.uniform(0.0005, 0.003, n_samples)
    # f: 0.05 to 0.3
    f = np.random.uniform(0.05, 0.3, n_samples)
    # D*: 0.005 to 0.05
    D_star = np.random.uniform(0.005, 0.05, n_samples)
    # K: 0 to 1.5
    K = np.random.uniform(0, 1.5, n_samples)
    
    X = []
    for i in range(n_samples):
        # Generate signal
        term_diff = np.exp(-b_values * D[i] + (1/6) * (b_values**2) * (D[i]**2) * K[i])
        term_perf = np.exp(-b_values * D_star[i])
        S0 = 1000
        signal = S0 * (f[i] * term_perf + (1 - f[i]) * term_diff)
        
        # Add noise
        noise = np.random.normal(0, 0.02 * S0, size=len(b_values))
        signal_noisy = signal + noise
        signal_noisy[signal_noisy < 0] = 0
        
        # Normalize
        signal_norm = signal_noisy / np.max(signal_noisy)
        X.append(signal_norm)
        
    X = np.array(X)
    # Targets: Let's predict D for this example
    y = D 
    
    return X, y, b_values

def main():
    print("=== ML Model Training Demo ===")
    
    # 1. Get Data (Synthetic for demo, but structure allows loading real data)
    # In a real scenario, you would use:
    # X, Y = load_training_data('data/MR701')
    # y = Y[:, 0] # D column
    
    X, y, b_values = generate_synthetic_training_data()
    
    # 2. Initialize Model
    # We use Random Forest as it was one of the best performers in the paper
    regressor = IVIMRegressor(model_type='random_forest', params={'n_estimators': 50})
    
    # 3. Train
    print("\nTraining Random Forest to predict Diffusion Coefficient (D)...")
    metrics = regressor.train(X, y)
    
    # 4. Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    regressor.save('models/rf_diffusion_model.joblib')
    
    # 5. Compare with Analytical Fit on a few samples
    print("\n--- Comparison: ML vs Analytical ---")
    indices = np.random.choice(len(X), 5)
    
    for idx in indices:
        signal = X[idx]
        true_val = y[idx]
        
        # ML Prediction
        ml_pred = regressor.predict([signal])[0]
        
        # Analytical Fit
        # Note: calculate_ivim_params expects raw signal, but we normalized. 
        # It handles normalization internally, so passing normalized is fine (S0=1).
        _, d_ana, _, _, _ = calculate_ivim_params(b_values, signal, gof=0.9)
        
        print(f"True D: {true_val:.6f} | ML: {ml_pred:.6f} | Analytical: {d_ana:.6f}")

if __name__ == "__main__":
    main()
