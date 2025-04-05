import numpy as np
from scipy.linalg import fractional_matrix_power, expm
from util import pca  # assumes a pca(reps, num_components) function returning (scores, U)
from kmeans_filters import k_lowest_ind  # reuse the same helper from your kmeans module

def cov_estimation_iterate(reps_pca, eps_div_n, tau, dummy, limit):
    """
    Placeholder for the cov_estimation_iterate function (from dkk17.jl).
    Replace this with your actual iterative covariance estimation algorithm.
    For now, simply selects all samples.
    """
    return np.ones(reps_pca.shape[1], dtype=bool)

def rcov_quantum_filter(reps, eps, k, alpha=4, tau=0.1, limit1=2, limit2=1.5):
    """
    Quantum-based filtering routine.
    
    Parameters:
      reps: A 2D NumPy array of shape (d, n)
      eps: An integer used for thresholding (e.g., number of poisoned samples)
      k: Number of principal components to use
      alpha, tau: Tuning parameters (defaults: 4 and 0.1)
      limit1, limit2: Multiplicative limits (defaults: 2 and 1.5)
      
    Returns:
      A boolean mask (shape (n,)) indicating samples that are not considered poisoned.
    """
    d, n = reps.shape
    reps_pca, U = pca(reps, num_components=k)
    
    if k == 1:
        reps_estimated_white = reps_pca
        sigma_prime = np.ones((1, 1))
    else:
        # Use the covariance estimation routine (placeholder here)
        selected = cov_estimation_iterate(reps_pca, eps / n, tau, None, limit=int(round(limit1 * eps)))
        # Select columns from reps_pca corresponding to the mask or indices returned by cov_estimation_iterate
        reps_pca_selected = reps_pca[:, selected]
        # Compute covariance matrix of the selected samples (observations in rows)
        sigma = np.cov(reps_pca_selected.T, bias=False)
        # Compute the inverse square root of sigma
        inv_sqrt_sigma = fractional_matrix_power(sigma, -0.5)
        # Whiten the original PCA data
        reps_estimated_white = inv_sqrt_sigma @ reps_pca
        sigma_prime = np.cov(reps_estimated_white.T, bias=False)
    
    I = np.eye(sigma_prime.shape[0])
    if k > 1:
        norm_sigma_prime = np.linalg.norm(sigma_prime, 2)
        M = expm(alpha * (sigma_prime - I) / (norm_sigma_prime - 1))
    else:
        M = np.ones((1, 1))
    M /= np.trace(M)
    
    # Compute the quadratic form for each column vector of reps_estimated_white.
    # In Julia: [x'M*x for x in eachcol(reps_estimated_white)]
    vals = np.array([col.T @ M @ col for col in reps_estimated_white.T])
    # Get a boolean mask selecting the indices with the lowest values (negated, as in Julia)
    estimated_poison_ind = k_lowest_ind(-vals, int(round(limit2 * eps)))
    # Return the logical NOT (i.e. mark as clean those not selected as poison)
    return ~estimated_poison_ind

def rcov_auto_quantum_filter(reps, eps, alpha=4, tau=0.1, limit1=2, limit2=1.5):
    """
    Automatically determine the best quantum filter by varying the number of principal components.
    
    Parameters:
      reps: A 2D NumPy array
      eps: An integer threshold parameter
      alpha, tau: Tuning parameters (defaults: 4 and 0.1)
      limit1, limit2: Multiplicative limits (defaults: 2 and 1.5)
      
    Returns:
      The boolean mask (of shape (n,)) corresponding to the best filter found.
    """
    reps_pca, U = pca(reps, num_components=100)
    best_opnorm = -np.inf
    best_selected = None
    best_k = None

    # Generate 10 k values by squaring 10 numbers linearly spaced from 1 to 10.
    k_values = [int(round(x**2)) for x in np.linspace(1, 10, 10)]
    
    for k in k_values:
        selected = rcov_quantum_filter(reps, eps, k, alpha, tau, limit1=limit1, limit2=limit2)
        # Compute sigma on the selected samples from reps_pca
        reps_pca_selected = reps_pca[:, selected]
        if reps_pca_selected.shape[1] == 0:
            continue
        sigma = np.cov(reps_pca_selected.T, bias=False)
        inv_sqrt_sigma = fractional_matrix_power(sigma, -0.5)
        reps_estimated_white = inv_sqrt_sigma @ reps_pca
        sigma_prime = np.cov(reps_estimated_white.T, bias=False)
        I = np.eye(sigma_prime.shape[0])
        norm_sigma_prime = np.linalg.norm(sigma_prime, 2)
        denom = norm_sigma_prime - 1
        if np.abs(denom) < 1e-8:
            continue
        M = expm(alpha * (sigma_prime - I) / denom)
        M /= np.trace(M)
        op = np.trace(sigma_prime @ M) / np.trace(M)
        poison_removed = np.sum(~selected[-eps:])
        print(f"k: {k}, op: {op}, poison_removed: {poison_removed}")
        if op > best_opnorm:
            best_opnorm = op
            best_selected = selected
            best_k = k
    print(f"best_k: {best_k}, best_opnorm: {best_opnorm}")
    return best_selected
