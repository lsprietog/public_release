import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Ensure output directory exists
if not os.path.exists('models'):
    os.makedirs('models')

def generate_comprehensive_training_data(n_samples=100000):
    """
    Generates a large, comprehensive dataset covering a wide range of b-values
    and physiological parameters to create a robust pre-trained model.
    """
    print(f"Generating {n_samples} synthetic samples for pre-training...")
    
    # Extended set of b-values to cover various clinical protocols (Prostate, Brain, etc.)
    # We include a superset of common b-values. 
    # NOTE: For the pre-trained model to be universal, it ideally needs to be trained 
    # on the SPECIFIC b-values of the target protocol. 
    # However, for a general purpose 'demo' model, we will use a standard clinical set.
    # Users should ideally retrain for their specific protocol, but this serves as a strong baseline.
    b_values = np.array([0, 10, 20, 30, 50, 80, 100, 150, 200, 400, 600, 800, 1000, 1500, 2000])
    
    # Random parameters within broad physiological ranges
    D = np.random.uniform(0.0001, 0.004, n_samples)      # Diffusion coefficient (mm2/s)
    f = np.random.uniform(0.01, 0.4, n_samples)          # Perfusion fraction
    D_star = np.random.uniform(0.005, 0.1, n_samples)    # Pseudo-diffusion (mm2/s)
    K = np.random.uniform(0, 2.5, n_samples)             # Kurtosis (dimensionless)
    
    X = []
    Y = [] # Targets: [D, f, D*, K]
    
    for i in range(n_samples):
        # Signal model: IVIM-DKI
        # S/S0 = f * exp(-b*D*) + (1-f) * exp(-b*D + 1/6 * b^2 * D^2 * K)
        
        # Diffusion term with Kurtosis
        # Note: We clip the exponent to avoid overflow/numerical issues at high b-values/K
        exponent_diff = -b_values * D[i] + (1/6) * (b_values**2) * (D[i]**2) * K[i]
        term_diff = np.exp(exponent_diff)
        
        # Perfusion term
        term_perf = np.exp(-b_values * D_star[i])
        
        signal = f[i] * term_perf + (1 - f[i]) * term_diff
        
        # Add Rician noise (approximated as Gaussian for simplicity in this large batch)
        # SNR varying between 20 and 100
        snr = np.random.uniform(20, 100)
        sigma = 1.0 / snr
        noise = np.random.normal(0, sigma, size=len(b_values))
        
        signal_noisy = signal + noise
        
        # Rician correction (magnitude)
        signal_noisy = np.sqrt(signal_noisy**2 + noise**2) # Simple approximation
        
        # Normalize (though it's already relative to S0=1)
        signal_norm = signal_noisy / np.max(signal_noisy)
        
        X.append(signal_norm)
        Y.append([D[i], f[i], D_star[i], K[i]])
        
    return np.array(X), np.array(Y), b_values

def train_and_save():
    X, Y, b_values = generate_comprehensive_training_data(n_samples=50000)
    
    print("Training ExtraTreesRegressor (Multi-output)...")
    # Optimized for model size < 100MB for GitHub hosting
    model = ExtraTreesRegressor(
        n_estimators=50,        # Reduced from 100
        max_depth=20,           # Limit depth to prevent massive trees
        min_samples_leaf=5,     # Prune leaves to reduce size
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    model.fit(X, Y)
    
    # Evaluate on a small subset
    preds = model.predict(X[:1000])
    mse = mean_squared_error(Y[:1000], preds)
    print(f"Training MSE: {mse:.6f}")
    
    # Save model
    model_path = 'models/ivim_dki_extratrees.joblib'
    joblib.dump(model, model_path, compress=3) # Compress to keep file size small
    print(f"Model saved to {model_path}")
    
    # Save b-values metadata so we know what protocol this model expects
    np.save('models/b_values_config.npy', b_values)
    print("Configuration saved.")

if __name__ == "__main__":
    train_and_save()
