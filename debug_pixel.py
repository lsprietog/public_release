import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ivim_model import calculate_ivim_params, _fit_pixel

def func_qua(x, a0, a1, a2):
    return a2 * (x ** 2) + a1 * x + a0

def main():
    print("=== Debugging Single Pixel Fit ===")
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'CFIN')
    nii_file = [f for f in os.listdir(data_dir) if f.endswith('.nii') and 'DTI' in f][0]
    bval_file = [f for f in os.listdir(data_dir) if f.endswith('.bval')][0]
    
    img = nib.load(os.path.join(data_dir, nii_file))
    data = img.get_fdata()
    bvals = np.loadtxt(os.path.join(data_dir, bval_file))
    
    # Preprocess (Average)
    unique_b = np.unique(np.round(bvals, -1))
    avg_data_list = []
    for b_val in unique_b:
        idxs = np.where(np.abs(bvals - b_val) < 10)[0]
        if len(idxs) > 0:
            avg_data_list.append(np.mean(data[:, :, :, idxs], axis=3))
    avg_data = np.stack(avg_data_list, axis=3)
    
    # Pick a pixel in the brain
    # Slice 9, x=50, y=50 (approx center)
    slice_idx = 9
    x, y = 48, 48
    
    signal = avg_data[x, y, slice_idx, :]
    print(f"Pixel ({x}, {y}, {slice_idx})")
    print(f"b-values: {unique_b}")
    print(f"Signal: {signal}")
    
    # Run IVIM calculation
    print("\n--- Running calculate_ivim_params (Quadratic) ---")
    r2, D, f, D_star, K = calculate_ivim_params(unique_b, signal, model_type='quadratic', gof=0.9)
    print(f"Result: D={D}, f={f}, D*={D_star}, K={K}, R2={r2}")
    
    # Manual check of the fit
    vec_b = unique_b
    vec_S = signal / np.max(signal)
    vec_S_log = np.log(vec_S + 1e-10)
    
    # Fit high b
    limit_dif = 180
    idx = np.where(vec_b >= limit_dif)[0][0]
    b_high = vec_b[idx:]
    S_high = vec_S_log[idx:]
    
    print(f"\nHigh-b data (b >= {limit_dif}):")
    print(f"b: {b_high}")
    print(f"log(S): {S_high}")
    
    # Try fit
    bounds = ([-np.inf, -np.inf, 0], [np.inf, 0, np.inf])
    try:
        popt, _ = curve_fit(func_qua, b_high, S_high, bounds=bounds)
        print(f"Manual Curve Fit params: a0={popt[0]}, a1={popt[1]}, a2={popt[2]}")
        print(f"Implied D = {-popt[1]}")
        print(f"Implied K = {popt[2] / popt[1]**2 * 6 if popt[1]!=0 else 0}")
    except Exception as e:
        print(f"Fit failed: {e}")

if __name__ == "__main__":
    main()
