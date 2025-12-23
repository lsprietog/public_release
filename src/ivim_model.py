import numpy as np
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import multiprocessing

def func_lin(x, a0, a1):
    """Linear function: a1*x + a0"""
    return a1 * x + a0

def func_qua(x, a0, a1, a2):
    """Quadratic function: a2*x^2 + a1*x + a0"""
    return a2 * (x ** 2) + a1 * x + a0

def _fit_step(x_data, y_data, model_type):
    """
    Performs a single fitting step (linear or quadratic).
    """
    if model_type == 'linear':
        # Constrain slope to be negative (decay)
        bounds = (-np.inf, [np.inf, 0])
        try:
            popt, _ = curve_fit(func_lin, x_data, y_data, bounds=bounds)
            residuals = np.sum((y_data - func_lin(x_data, *popt)) ** 2)
            a2 = 0
            a0, a1 = popt
        except RuntimeError:
            return 0, 0, 0, 0
    else:
        # Constrain quadratic term
        bounds = ([-np.inf, -np.inf, 0], [np.inf, 0, np.inf])
        try:
            popt, _ = curve_fit(func_qua, x_data, y_data, bounds=bounds)
            residuals = np.sum((y_data - func_qua(x_data, *popt)) ** 2)
            a0, a1, a2 = popt
        except RuntimeError:
            return 0, 0, 0, 0
    
    y_mean = np.mean(y_data)
    ss_tot = np.sum((y_data - y_mean) ** 2)
    
    r2 = 1 - residuals / ss_tot if ss_tot != 0 else 0
    
    return a0, a1, a2, r2

def _fit_pixel(x, y, model_type='linear', gof_threshold=0.9):
    """
    Iterative fitting process. Removes outliers if R2 is below threshold.
    """
    a0, a1, a2, r2 = _fit_step(x, y, model_type)
    
    if r2 < gof_threshold:
        best_r2 = 0
        best_params = (a0, a1, a2)
        best_data = (x, y)
        
        # Optimization: Only check points with high residuals if N is large
        if len(x) > 20:
            if model_type == 'linear':
                y_pred = func_lin(x, a0, a1)
            else:
                y_pred = func_qua(x, a0, a1, a2)
            
            residuals = (y - y_pred)**2
            # Check only the top 5 worst outliers for speed
            n_check = min(len(x), 5)
            indices_to_check = np.argsort(residuals)[-n_check:]
        else:
            indices_to_check = range(len(x))
        
        # Leave-one-out strategy on candidate points
        for i in indices_to_check:
            x_subset = np.delete(x, i)
            y_subset = np.delete(y, i)
            
            curr_a0, curr_a1, curr_a2, curr_r2 = _fit_step(x_subset, y_subset, model_type)
            
            if curr_r2 > best_r2:
                best_r2 = curr_r2
                best_params = (curr_a0, curr_a1, curr_a2)
                best_data = (x_subset, y_subset)
        
        return best_params, best_r2, best_data
    
    return (a0, a1, a2), r2, (x, y)

def media(r1, r2, z=1):
    """Calculates mean (geometric, harmonic, or arithmetic)."""
    if r1 <= 0.01: r1 = 1E-3
    if r2 <= 0.01: r2 = 1E-3
    
    if z == 1:    # Geometric
        m = np.sqrt(r1 * r2)
    elif z == 2:  # Harmonic
        m = 2 / (1/r1 + 1/r2)
    else:         # Arithmetic
        m = (r1 + r2) / 2
    return m

def calculate_ivim_params(b_values, signal_values, gof=0.9, limit_dif=180, model_type='linear'):
    """
    Calculates IVIM and Kurtosis parameters (D, f, D*, K) for a single voxel using a segmented fitting approach.
    
    Args:
        b_values: Array of b-values.
        signal_values: Array of signal intensities corresponding to b-values.
        gof: Goodness of fit threshold (R2) to trigger outlier removal.
        limit_dif: b-value threshold separating perfusion (low b) and diffusion (high b) regimes.
        model_type: 'linear' for Mono-exponential (IVIM), 'quadratic' for Kurtosis (IVIM-DKI).
        
    Returns:
        r2_new: Combined R2 score.
        D: Diffusion coefficient.
        f: Perfusion fraction.
        D_pse: Pseudo-diffusion coefficient (D*).
        K: Kurtosis (0 if model_type is 'linear').
    """
    vec_b = np.array(b_values)
    vec_S = np.array(signal_values)
    
    # Normalize signal
    vec_S[np.isnan(vec_S)] = 0
    S0 = np.max(vec_S)
    if S0 == 0:
        return 0, 0, 0, 0, 0
        
    vec_S = vec_S / S0
    
    # Avoid log(0) issues
    vec_S[vec_S <= 0] = 1e-10
    vec_S_log = np.log(vec_S)

    # Split data into vascular (low b) and cellular (high b) regimes
    try:
        limite_dif_idx = np.where(vec_b >= limit_dif)[0][0]
    except IndexError:
        limite_dif_idx = len(vec_b)

    vec_b_vas = vec_b[:limite_dif_idx + 1]
    vec_b_cel = vec_b[limite_dif_idx:]
    
    vec_S_vas = vec_S[:limite_dif_idx + 1]
    vec_S_cel = vec_S[limite_dif_idx:]
    vec_S_cel_log = vec_S_log[limite_dif_idx:]

    # 1. Fit Diffusion/Kurtosis (High b-values)
    # Perform initial fit on the full dataset to check overall quality
    a, r, data = _fit_pixel(vec_b, vec_S_log, model_type, gof_threshold=gof)
    
    if r < gof:
        # If global fit is poor, proceed with Segmented Fitting
        
        # Step 1: Fit High-b values (Diffusion regime)
        a, r, data = _fit_pixel(vec_b_cel, vec_S_cel_log, model_type, gof_threshold=gof) 
        
        # Calculate Diffusion (D) and Kurtosis (K) from high-b fit
        D = -a[1]
        if model_type == 'quadratic' and D != 0:
            K = (a[2] / D**2) * 6
        else:
            K = 0
            
        # Calculate Perfusion Fraction (f) from intercept
        # Intercept a[0] corresponds to ln(1-f)
        uno_menos_f = np.exp(a[0])
        f = 1 - uno_menos_f
        
        # Step 2: Extrapolate diffusion contribution to low-b values
        # S_diff_extrapolated = exp(a2*b^2 + a1*b + a0)
        if model_type == 'linear':
             diffusion_contribution = np.exp(a[1]*vec_b_vas + a[0])
        else:
             diffusion_contribution = np.exp(a[2]*(vec_b_vas**2) + a[1]*vec_b_vas + a[0])
             
        # Step 3: Subtract Diffusion from Total Signal to isolate Perfusion
        # S_perfusion = S_total - S_diffusion
        y4 = vec_S_vas - diffusion_contribution

        # Ensure residuals are positive for logarithmic fitting
        if np.min(y4) < 0:
             y4 = y4 + abs(np.min(y4))
             
        # Prepare for Perfusion fit (log of residuals)
        y5 = np.zeros(len(y4))
        valid_indices = []
        for i in range(len(y4)):
            if y4[i] > 0:
                y5[i] = np.log(y4[i])
                valid_indices.append(i)
        
        if len(valid_indices) > 2:
            y5_clean = y5[valid_indices]
            vec_b_vas_clean = vec_b_vas[valid_indices]
            
            # Step 4: Fit Perfusion (Pseudo-diffusion) using a linear model
            A_param, R_param, _ = _fit_pixel(vec_b_vas_clean, y5_clean, 'linear', gof_threshold=gof)
            
            D_pse = -A_param[1]
            r2_new = media(R_param, r, 1)
        else:
            # Fallback: Failed to isolate perfusion component (D*)
            # But we keep D, f, K from the high-b fit
            D_pse = 0
            r2_new = r
            
    else:
        # Good global fit: Assume Mono-exponential / Kurtosis model fits entire range
        D = -a[1]
        f = 0
        if model_type == 'quadratic' and D != 0:
            K = (a[2] / D**2) * 6
        else:
            K = 0
        D_pse = 0
        r2_new = r

    return r2_new, D, f, D_pse, K

def process_slice_parallel(b_values, slice_data, mask=None, gof=0.9, limit_dif=180, model_type='linear', n_jobs=-1):
    """
    Processes an entire 2D slice (Rows x Cols x b-values) in parallel to extract IVIM/DKI parameters.
    
    Args:
        b_values: Array of b-values.
        slice_data: 3D array (Rows, Cols, b-values) containing signal intensities.
        mask: Binary mask (Rows, Cols) indicating pixels to process. If None, processes all.
        gof: Goodness of fit threshold.
        limit_dif: b-value threshold for segmentation.
        model_type: 'linear' or 'quadratic'.
        n_jobs: Number of parallel jobs. -1 uses all available CPUs.
        
    Returns:
        Tuple of 2D maps: (R2, D, f, D*, K)
    """
    rows, cols, n_b = slice_data.shape
    
    if mask is None:
        mask = np.ones((rows, cols), dtype=bool)
        
    # Prepare list of tasks for parallel execution
    tasks = []
    coords = []
    
    for i in range(rows):
        for j in range(cols):
            if mask[i, j]:
                signal = slice_data[i, j, :]
                tasks.append((b_values, signal, gof, limit_dif, model_type))
                coords.append((i, j))
                
    if not tasks:
        # Return empty maps if no pixels to process
        return (np.zeros((rows, cols)) for _ in range(5))

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
        
    print(f"Processing {len(tasks)} pixels using {n_jobs} threads...")
    
    # Execute parallel processing
    # Note: 'threading' backend is preferred on Windows to avoid pickling overhead and issues with local functions
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        delayed(calculate_ivim_params)(*t) for t in tasks
    )
    
    # Reconstruct 2D parameter maps from flat results
    map_R2 = np.zeros((rows, cols))
    map_D = np.zeros((rows, cols))
    map_f = np.zeros((rows, cols))
    map_D_star = np.zeros((rows, cols))
    map_K = np.zeros((rows, cols))
    
    for (i, j), (r2, D, f, D_star, K) in zip(coords, results):
        map_R2[i, j] = r2
        map_D[i, j] = D
        map_f[i, j] = f
        map_D_star[i, j] = D_star
        map_K[i, j] = K
        
    return map_R2, map_D, map_f, map_D_star, map_K
