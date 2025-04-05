import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse.linalg import svds
from scipy.linalg import svd, norm

def pca(A, k):
    """
    Compute PCA on matrix A (each column is a sample) and return:
      - The PCA scores: U.T @ (A - mean(A, axis=1))
      - The matrix U.T (transposed left singular vectors)
    
    If k equals the minimum dimension of A, the full SVD is used.
    Otherwise, a partial SVD is computed.
    """
    A = np.asarray(A)
    m, n = A.shape
    if k > min(m, n):
        raise ValueError("k must be <= minimum dimension of A")
    # Center each row (feature)
    A_centered = A - np.mean(A, axis=1, keepdims=True)
    if k == min(m, n):
        U, s, Vt = svd(A_centered, full_matrices=False)
    else:
        # Compute partial SVD (svds may return singular values in ascending order)
        U, s, Vt = svds(A_centered, k=k)
        # Sort in descending order:
        idx = np.argsort(s)[::-1]
        s = s[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]
    # Return (U.T @ A_centered, U.T) to mimic Julia's: (U' * A_centered, U')
    return U.T @ A_centered, U.T

def svd_pow(A, p):
    """
    Compute A^(p) in the SVD sense:
      A_p = U * diag(s^p) * Vt
    Requires A to be square and all singular values positive.
    """
    A = np.asarray(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    U, s, Vt = svd(A)
    if not np.all(s > 0):
        raise ValueError("Not all singular values are positive")
    return U @ np.diag(s**p) @ Vt

def k_lowest_ind(A, k):
    """
    Given a 1D array A, return a boolean mask that is True for the k smallest elements.
    If k is 0, return an array of False values.
    """
    A = np.asarray(A)
    if k < 0 or k > A.size:
        raise ValueError("k must be between 0 and the length of A")
    if k == 0:
        return np.zeros(A.shape, dtype=bool)
    # Sort all elements; the cutoff is the kth smallest element (using 1-indexing logic)
    sorted_A = np.sort(A.flatten())
    cutoff = sorted_A[k-1]
    return A <= cutoff

def step_vec(n, k):
    """
    Create a boolean vector of length n where the first k elements are True and the rest False.
    """
    v = np.zeros(n, dtype=bool)
    v[:k] = True
    return v

def sb_pairplot(A, clean=5000):
    """
    Create a pairplot from the transposed matrix A using Seaborn,
    adding a "poison" column that marks the samples after the first 'clean' samples as poisoned.
    """
    A = np.asarray(A)
    d, n = A.shape
    df = pd.DataFrame(A.T)
    # Mark as poison those indices not in the first 'clean' samples.
    df["poison"] = ~step_vec(n, clean)
    sns.pairplot(df, diag_kind="kde", hue="poison")
    
def flat(A):
    """
    Flatten a square matrix A into a 1D array.
    """
    A = np.asarray(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square")
    return A.flatten()

def sharp(v):
    """
    Reshape a 1D array v into a square matrix. Raises an error if the length is not a perfect square.
    """
    v = np.asarray(v)
    n = v.size
    m = int(np.sqrt(n))
    if m * m != n:
        raise ValueError("Array cannot be reshaped into a square matrix")
    return v.reshape((m, m))
