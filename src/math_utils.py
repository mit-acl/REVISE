import numpy as np
import scipy

def check_psd(A):
    """
    Check if matrix is positive semidefinite by attempting
    Cholesky decomposition.

    Parameters
    -----------
        A: np.ndarray
            Matrix to check for positive semidefiniteness
    Returns
    --------
        True if A is PSD, False otherwise
    """
    try:
        _ = scipy.linalg.cholesky(A, lower=True)
        return True
    except np.linalg.LinAlgError:
        return False

def ensure_psd(A, eps=1e-9):
    """
    Ensure matrix is positive semidefinite by adding I*eps.

    Parameters
    -----------
        A: np.ndarray
            Matrix which is within floating point error
            of having all positive eigenvalues
    Returns
    --------
        A_psd: np.ndarray
            A + I*1e-9
    """
    # Small tweak to address numerical issues with very small eigenvalues
    A_psd = A + np.eye(A.shape[0])*eps

    return A_psd
