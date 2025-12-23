import numpy as np

def filtro_2D(veces, col, row, Matriz_R):
    """
    Applies a 2D filter to the R2 matrix to smooth it.
    """
    R_old = np.copy(Matriz_R)
    R_new = np.copy(Matriz_R)
    
    for k in range(veces):
        for i in range(1, col-1):
            for j in range(1, row-1):
                w_A = (R_old[i-1, j] + R_old[i, j-1] + R_old[i+1, j] + R_old[i, j+1]) / 4
                w_B = (R_old[i-1, j-1] + R_old[i+1, j-1] + R_old[i-1, j+1] + R_old[i+1, j+1]) / (4 * np.sqrt(2))
                
                if (w_A + w_B) > 1:
                    R_new[i, j] = 1
                else:
                    R_new[i, j] = 0
        R_old = np.copy(R_new)
    
    return R_new

def filtro_3D(veces, slices, col, row, Tensor_R):
    """
    Applies a 3D filter to the R2 tensor.
    """
    R_old = np.copy(Tensor_R)
    R_new = np.copy(Tensor_R)
    
    for u in range(veces):
        for k in range(1, slices-1):
            for i in range(1, col-1):
                for j in range(1, row-1):
                    w_A = (R_old[i-1, j, k] + R_old[i, j-1, k] + R_old[i+1, j, k] + R_old[i, j+1, k] + R_old[i, j, k-1] + R_old[i, j, k+1]) / 6
                    w_B1 = (R_old[i-1, j, k-1] + R_old[i+1, j, k-1] + R_old[i-1, j, k+1] + R_old[i+1, j, k+1]) / (12 * np.sqrt(2))
                    w_B2 = (R_old[i-1, j-1, k] + R_old[i-1, j+1, k] + R_old[i+1, j-1, k] + R_old[i+1, j+1, k]) / (12 * np.sqrt(2))
                    w_B3 = (R_old[i, j-1, k-1] + R_old[i, j+1, k-1] + R_old[i, j-1, k+1] + R_old[i, j+1, k+1]) / (12 * np.sqrt(2))
                    w_C = (R_old[i-1, j-1, k+1] + R_old[i-1, j+1, k+1] + R_old[i+1, j-1, k+1] + R_old[i+1, j+1, k+1] + R_old[i-1, j-1, k-1] + R_old[i-1, j+1, k-1] + R_old[i+1, j-1 ,k-1] + R_old[i+1, j+1 ,k-1]) / (8 * np.sqrt(3))
                    w_B = w_B1 + w_B2 + w_B3
                    
                    if (w_A + w_B + w_C) >= 1:
                        R_new[i, j, k] = 1
                    else:
                        R_new[i, j, k] = 0
        R_old = np.copy(R_new)
        
    return R_new
