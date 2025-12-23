import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_ivim
from dipy.core.gradients import gradient_table
import nibabel as nib
from tempfile import TemporaryDirectory

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ivim_model import process_slice_parallel
from utils import filtro_2D

def main():
    print("=== DIPY IVIM Dataset Demo ===")
    
    # Fetch IVIM data
    print("Fetching IVIM data from DIPY...")
    # This downloads the data to the user's home directory by default (.dipy/ivim)
    files, folder = fetch_ivim()
    
    # The fetch_ivim returns a dictionary or list of files depending on version, 
    # but usually it downloads 'ivim_data.nii.gz' and 'ivim_bvals.txt'
    # Let's find them in the folder
    print(f"Data downloaded to: {folder}")
    
    nii_path = os.path.join(folder, 'ivim.nii.gz')
    bval_path = os.path.join(folder, 'ivim.bval')
    
    if not os.path.exists(nii_path):
        # Fallback for different dipy versions
        nii_path = os.path.join(folder, 'ivim_data.nii.gz')
        
    print(f"Loading NIfTI: {nii_path}")
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    print(f"Loading b-values: {bval_path}")
    bvals = np.loadtxt(bval_path)
    
    print(f"Data shape: {data.shape}")
    print(f"b-values: {bvals}")
    
    # Select slice 15 as requested
    slice_idx = 15
    if slice_idx >= data.shape[2]:
        slice_idx = data.shape[2] // 2
        
    print(f"Processing slice {slice_idx}...")
    
    slice_data = data[:, :, slice_idx, :]
    rows, cols, _ = slice_data.shape
    
    # Create a mask (simple thresholding on b0)
    b0_idx = np.argmin(bvals)
    b0_img = slice_data[:, :, b0_idx]
    mask = b0_img > np.mean(b0_img) * 0.2
    
    print(f"Processing full slice with parallel execution.")
    
    # Use parallel processing
    # Note: The original paper uses 'quadratic' (Kurtosis) model often, but for standard IVIM data 'linear' might be safer?
    # Let's stick to 'quadratic' as it's the paper's main contribution.
    map_R2, map_D, map_f, map_D_star, map_K = process_slice_parallel(
        bvals, slice_data, mask=mask, gof=0.90, model_type='quadratic', n_jobs=-1
    )

    # Apply filter
    # Note: filtro_2D is binary, skipping for parameter maps
    # print("Applying spatial filter...")
    # map_D = filtro_2D(1, cols, rows, map_D)
    # map_f = filtro_2D(1, cols, rows, map_f)
    # map_D_star = filtro_2D(1, cols, rows, map_D_star)
    
    # Plot results matching the style of Principal_IVIM.ipynb
    print("Plotting results...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    def show_map(ax, data, title, vmin=None, vmax=None, cmap='gray'):
        # Rotate to match typical orientation if needed
        im = ax.imshow(np.rot90(data), cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    # Ranges based on typical values or auto-scaled
    show_map(axes[0], map_R2, "R2", 0.5, 1.0, 'gray')
    show_map(axes[1], map_D, "D (Diffusion)", 0, 0.002, 'viridis') # Typical D in brain ~0.0007 - 0.001 mm2/s
    show_map(axes[2], map_f, "f (Perfusion)", 0, 0.3, 'viridis')
    show_map(axes[3], map_D_star, "D* (Pseudo-Diff)", 0, 0.05, 'viridis')
    show_map(axes[4], map_K, "K (Kurtosis)", 0, 1.5, 'viridis')

    plt.tight_layout()
    plt.savefig('dipy_ivim_results.png')
    print("Results saved to dipy_ivim_results.png")

if __name__ == "__main__":
    main()
