import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ivim_model import calculate_ivim_params

def generate_synthetic_signal(b_values, D, f, D_star, K=0, noise_level=0.02):
    """
    Generates synthetic IVIM-Kurtosis signal.
    S(b) = S0 * [ f * exp(-b*D*) + (1-f) * exp(-b*D + 1/6 * b^2 * D^2 * K) ]
    """
    S0 = 1000
    b = np.array(b_values)
    
    # Diffusion part with Kurtosis
    # Note: Taylor expansion for Kurtosis is usually valid for b*D < 1 or similar.
    # Standard DKI model: exp(-b*D + 1/6 * b^2 * D^2 * K)
    
    term_diff = np.exp(-b * D + (1/6) * (b**2) * (D**2) * K)
    term_perf = np.exp(-b * D_star)
    
    signal = S0 * (f * term_perf + (1 - f) * term_diff)
    
    # Add Rician noise (approximated as Gaussian for high SNR)
    noise = np.random.normal(0, noise_level * S0, size=len(b))
    signal_noisy = signal + noise
    signal_noisy[signal_noisy < 0] = 0 # Magnitude signal
    
    return signal_noisy

def main():
    print("=== IVIM-DKI Estimation Demo ===")
    
    # 1. Define Ground Truth Parameters
    D_true = 0.001   # mm^2/s
    f_true = 0.15    # fraction
    D_star_true = 0.01 # mm^2/s
    K_true = 0.8     # dimensionless
    
    print(f"Ground Truth: D={D_true}, f={f_true}, D*={D_star_true}, K={K_true}")
    
    # 2. Define b-values (typical clinical protocol)
    b_values = [0, 10, 20, 30, 50, 80, 100, 200, 400, 800, 1000, 1500, 2000]
    print(f"b-values: {b_values}")
    
    # 3. Generate Synthetic Data
    signal = generate_synthetic_signal(b_values, D_true, f_true, D_star_true, K_true)
    
    # 4. Fit Model
    print("\nFitting model...")
    # We use gof=0.999 to force the segmented fit (since synthetic data is very clean)
    # We use model_type='quadratic' to enable Kurtosis estimation
    r2, D_est, f_est, D_star_est, K_est = calculate_ivim_params(b_values, signal, gof=0.999, model_type='quadratic')
    
    # 5. Show Results
    print("\n--- Results ---")
    print(f"Estimated D:  {D_est:.6f} (Error: {abs(D_est - D_true)/D_true*100:.2f}%)")
    print(f"Estimated f:  {f_est:.6f} (Error: {abs(f_est - f_true)/f_true*100:.2f}%)")
    print(f"Estimated D*: {D_star_est:.6f} (Error: {abs(D_star_est - D_star_true)/D_star_true*100:.2f}%)")
    print(f"Estimated K:  {K_est:.6f} (Error: {abs(K_est - K_true)/K_true*100:.2f}%)")
    print(f"Goodness of fit (R2): {r2:.4f}")
    
    # 6. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(b_values, signal, 'o', label='Noisy Data')
    
    # Reconstruct fitted curve
    # Note: The fitting function returns parameters, we need to reconstruct the curve to plot
    # Ideally we would have a 'predict' function in ivim_model.py
    # For now, we manually reconstruct using the same logic as generation
    
    # Reconstruct using estimated parameters
    # Note: The fitting logic uses segmented approach, so the reconstruction might be slightly different 
    # if we strictly follow the fitting steps, but for visualization, the full model is best.
    
    b_smooth = np.linspace(0, max(b_values), 100)
    term_diff_est = np.exp(-b_smooth * D_est + (1/6) * (b_smooth**2) * (D_est**2) * K_est)
    term_perf_est = np.exp(-b_smooth * D_star_est)
    S0_est = np.max(signal) # Approximation used in fitting
    signal_est = S0_est * (f_est * term_perf_est + (1 - f_est) * term_diff_est)
    
    plt.plot(b_smooth, signal_est, '-', label='Fitted Curve')
    plt.xlabel('b-value (s/mm^2)')
    plt.ylabel('Signal Intensity')
    plt.title('IVIM-DKI Fit Demo')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    output_plot = 'demo_fit.png'
    plt.savefig(output_plot)
    print(f"\nPlot saved to {output_plot}")

if __name__ == "__main__":
    main()
