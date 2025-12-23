import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ivim_model import calculate_ivim_params, process_slice_parallel
from utils import filtro_2D

def load_cfin_data(data_dir):
    """
    Loads CFIN dataset files.
    """
    # Find files
    nii_file = None
    bval_file = None
    
    for f in os.listdir(data_dir):
        if f.endswith('.nii') and 'DTI' in f:
            nii_file = os.path.join(data_dir, f)
        elif f.endswith('.bval'):
            bval_file = os.path.join(data_dir, f)
            
    if not nii_file or not bval_file:
        raise FileNotFoundError("Could not find required .nii or .bval files in data directory.")
        
    print(f"Loading NIfTI: {os.path.basename(nii_file)}")
    img = nib.load(nii_file)
    data = img.get_fdata()
    
    print(f"Loading b-values: {os.path.basename(bval_file)}")
    bvals = np.loadtxt(bval_file)
    
    return data, bvals

def preprocess_dwi_data(data, bvals):
    """
    Averages signals for unique b-values (Trace-weighted image).
    This is necessary for DTI/DKI datasets with multiple directions per shell.
    """
    unique_b, inverse_indices = np.unique(np.round(bvals, -1), return_inverse=True) # Round to nearest 10 to group
    
    # Sort unique b-values
    sorted_indices = np.argsort(unique_b)
    unique_b = unique_b[sorted_indices]
    
    # Initialize averaged data
    rows, cols, slices, _ = data.shape
    avg_data = np.zeros((rows, cols, slices, len(unique_b)))
    
    print(f"Averaging {len(bvals)} volumes into {len(unique_b)} unique b-values: {unique_b}")
    
    for i, b_idx in enumerate(sorted_indices):
        # Find all original indices corresponding to this unique b-value
        # Note: inverse_indices maps original -> unique. 
        # We need to find where inverse_indices == b_idx (before sorting)
        # Actually, let's do it simpler.
        pass

    # Re-implementing loop for clarity
    avg_data_list = []
    for b_val in unique_b:
        # Find indices in original bvals that are close to this b_val
        # Using a tolerance of 10 s/mm2
        idxs = np.where(np.abs(bvals - b_val) < 10)[0]
        
        if len(idxs) > 0:
            # Average across the 4th dimension (volumes)
            vol_avg = np.mean(data[:, :, :, idxs], axis=3)
            avg_data_list.append(vol_avg)
        else:
            print(f"Warning: No volumes found for b={b_val}")
            
    # Stack along the 4th dimension
    avg_data = np.stack(avg_data_list, axis=3)
    
    return avg_data, unique_b

def main():
    print("=== CFIN Dataset IVIM Demo ===")
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'CFIN')
    
    try:
        data, bvals = load_cfin_data(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python download_data.py' first.")
        return

    print(f"Original Data shape: {data.shape}")
    print(f"Original Number of b-values: {len(bvals)}")
    
    # Preprocess: Average shells
    data, bvals = preprocess_dwi_data(data, bvals)
    print(f"Processed Data shape: {data.shape}")
    
    # Select a middle slice for demonstration
    slice_idx = data.shape[2] // 2
    print(f"Processing slice {slice_idx}...")
    
    slice_data = data[:, :, slice_idx, :]
    rows, cols, _ = slice_data.shape
    
    # Initialize parameter maps
    map_D = np.zeros((rows, cols))
    map_f = np.zeros((rows, cols))
    map_D_star = np.zeros((rows, cols))
    map_K = np.zeros((rows, cols))
    map_R2 = np.zeros((rows, cols))
    
    # Simple mask to avoid background (threshold on b0)
    b0_idx = np.argmin(bvals)
    b0_img = slice_data[:, :, b0_idx]
    mask = b0_img > np.mean(b0_img) * 0.2 # Simple threshold
    
    # DEMO OPTIMIZATION:
    # We now use parallel processing and optimized outlier search.
    # We can process the full slice much faster.
    
    print(f"\nProcessing full slice with parallel execution.")
    
    # Use parallel processing on the full mask
    map_R2, map_D, map_f, map_D_star, map_K = process_slice_parallel(
        bvals, slice_data, mask=mask, gof=0.90, model_type='quadratic', n_jobs=-1
    )

    # Apply filter to smooth results (optional, as in the paper)
    # Note: The provided filtro_2D in utils.py appears to be a binary mask filter, 
    # not a smoothing filter for continuous values. We skip it for parameter maps.
    # print("Applying spatial filter...")
    # map_D = filtro_2D(1, cols, rows, map_D)
    
    # Print statistics
    print("\n--- Parameter Statistics ---")
    print(f"D  : Mean={np.mean(map_D):.6f}, Max={np.max(map_D):.6f}, Min={np.min(map_D):.6f}")
    print(f"f  : Mean={np.mean(map_f):.6f}, Max={np.max(map_f):.6f}, Min={np.min(map_f):.6f}")
    print(f"D* : Mean={np.mean(map_D_star):.6f}, Max={np.max(map_D_star):.6f}, Min={np.min(map_D_star):.6f}")
    print(f"K  : Mean={np.mean(map_K):.6f}, Max={np.max(map_K):.6f}, Min={np.min(map_K):.6f}")
    
    # Plot results
    print("Plotting results...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Helper to rotate for better visualization (anatomical orientation)
    def show_map(ax, data, title, vmin, vmax, cmap):
        # Rotate 90 degrees to match anatomical view usually
        im = ax.imshow(np.rot90(data), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    # Adjust vmin/vmax based on typical physiological values
    show_map(axes[0], map_R2, "Goodness of Fit (R2)", 0.5, 1.0, 'gray')
    show_map(axes[1], map_D, "Diffusion (D)", 0, 0.003, 'viridis')
    show_map(axes[2], map_f, "Perfusion Fraction (f)", 0, 0.3, 'viridis') # f is usually 0-0.3
    show_map(axes[3], map_D_star, "Pseudo-Diffusion (D*)", 0, 0.1, 'viridis') # D* is usually high
    show_map(axes[4], map_K, "Kurtosis (K)", 0, 2.0, 'viridis')

    plt.tight_layout()
    plt.savefig('cfin_demo_results.png')
    print("Results saved to cfin_demo_results.png")

if __name__ == "__main__":
    main()
