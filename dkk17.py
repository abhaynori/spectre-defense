import numpy as np
import math
from util_jl_to import pca, sharp, k_lowest_ind
from numpy.linalg import eig, eigh
from scipy.linalg import svd, fractional_matrix_power, expm
from util import sharp, k_lowest_ind  # sharp reshapes a length-(d^2) vector into (d,d)
# Note: You may need to add additional helpers from util.py (e.g. flat) as needed.
# If you have a progress bar utility, you can import tqdm; otherwise, it is optional.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# -----------------------
# Helper Functions
# -----------------------

def symmetric_invsqrt(S):
    """
    Given a symmetric positive-definite matrix S, compute its inverse square root.
    """
    w, v = eigh(S)
    inv_sqrt_w = np.diag(1.0 / np.sqrt(w))
    return v @ inv_sqrt_w @ v.T

# -----------------------
# dkk17.jl Conversion
# -----------------------

def cov_tail(T, d, eps, tau):
    """
    In Julia:
      cov_Tail(T, d, ε, τ) = if T <= 10*log(1/ε) then 1 else 3ε/(T*log(T))^2 end
    """
    if T <= 10 * math.log(1/eps):
        return 1
    return 3 * eps / ((T * math.log(T)) ** 2)


def Q(G, P):
    """
    In Julia:
      Q(G, P) = 2norm(P)^2
    Here we interpret this as 2 * (spectral norm of P)^2.
    (Parameter G is unused.)
    """
    return 2 * (np.linalg.norm(P, 2) ** 2)


def cov_estimation_filter(S_prime, eps, tau=0.1, limit=None, method='krylov'):
    """
    In Julia, this function uses an early filter based on the quadratic forms of the whitened data.
    
    Parameters:
      S_prime : 2D np.array (d x n)
      eps     : threshold parameter (ε)
      tau     : default 0.1
      limit   : optional integer (or None)
      method  : either 'arpack' or 'krylov' (only 'arpack' is implemented here)
    
    Returns:
      - If an early filter condition is met, returns a boolean mask (of length n)
      - Otherwise, for the 'arpack' branch, performs an eigen–decomposition on a matrix TS.
    
    Note: The 'krylov' branch is not implemented and will raise an error.
    """
    d, n = S_prime.shape
    C = 10
    C_prime = 0  # C′ in Julia code
    # Compute Σ′ = S′ * S′' / n
    Sigma_prime = (S_prime @ S_prime.T) / n
    # In Julia: G′ = MvNormal(Σ′). Here we simply set G_prime = Σ′.
    G_prime = Sigma_prime
    # Compute symmetric inverse square root of Σ′
    invsqrt_Sigma = symmetric_invsqrt(Sigma_prime)
    # Compute Y = invsqrt_Sigma * S_prime
    Y = invsqrt_Sigma @ S_prime
    # Compute xinvSx = for each column y in Y, y'y
    xinvSx = np.sum(Y * Y, axis=0)
    threshold = C * d * math.log(n / tau)
    mask = xinvSx >= threshold
    if np.any(mask):
        print("early filter")
        if limit is None:
            return np.logical_not(mask)
        else:
            return np.logical_or(np.logical_not(mask),
                                   k_lowest_ind(xinvSx, max(0, n - limit)))
    
    if method == 'arpack':
        # For each column y in Y, compute kron(y, y)
        Z = np.column_stack([np.kron(Y[:, i], Y[:, i]) for i in range(n)])
        # Id_flat: flatten identity matrix of size d x d
        Id_flat = np.eye(d).flatten()
        # Compute TS = - (outer product of Id_flat with itself) + (Z @ Z.T) / n
        TS = -np.outer(Id_flat, Id_flat) + (Z @ Z.T) / n
        # Compute the dominant eigenvalue and eigenvector of TS.
        # Here we use np.linalg.eig and select the eigenpair with largest absolute eigenvalue.
        vals, vecs = eig(TS)
        idx = np.argmax(np.abs(vals))
        lambda_val = np.real(vals[idx])
        v = np.real(vecs[:, idx])
    else:
        raise NotImplementedError("Method 'krylov' is not implemented in this conversion.")
    
    # Use the sharp function to reshape v into (d, d)
    v_sharp = sharp(v)
    # Compute the threshold comparison:
    # In Julia: if λ <= (1 + C*ε*log(1/ε)^2)*Q(collect(invsqrtΣ′)*G′, sharp(v)) / 2 then return G′
    threshold_val = (1 + C * eps * (math.log(1/eps) ** 2)) * Q(invsqrt_Sigma @ G_prime, v_sharp) / 2
    if lambda_val <= threshold_val:
        return G_prime  # early termination signal
    
    # Otherwise, compute V = (sharp(v) + sharp(v)')/2 (symmetrize)
    V = (v_sharp + v_sharp.T) / 2
    # For each column y in Y, compute: 1/sqrt(2) * (y' V y - trace(V))
    ps = np.array([ (1/np.sqrt(2)) * (y.T @ V @ y - np.trace(V)) for y in Y.T ])
    mu = np.median(ps)
    diffs = np.abs(ps - mu)
    # Sort diffs and iterate over them (using sorted order)
    sorted_diffs = np.sort(diffs)
    for i, diff in enumerate(sorted_diffs):
        shift = 3
        if diff < shift:
            continue
        T_val = diff - shift
        if T_val <= C_prime:
            continue
        if (i + 1) / n >= cov_tail(T_val, d, eps, tau):
            if limit is None:
                return diffs <= T_val
            else:
                return np.logical_or(diffs <= T_val,
                                     k_lowest_ind(diffs, max(0, n - limit)))
    # If no condition is met, function returns None implicitly.
    return None


def cov_estimation_iterate(S_prime, eps, tau=0.1, k=None, iters=None, limit=None):
    """
    Iteratively refines the sample selection using cov_estimation_filter.
    
    Parameters:
      S_prime : 2D array (d x n)
      eps     : ε parameter
      tau     : τ (default 0.1)
      k       : number of PCA components to use (or None)
      iters   : maximum iterations (or None)
      limit   : optional limit on removals (or None)
    
    Returns:
      A boolean mask (of length n) indicating which columns of the original S_prime are selected.
    """
    d, n = S_prime.shape
    idxs = np.arange(n)
    i = 0
    if limit is not None:
        orig_limit = limit
        pbar = tqdm(total=limit, desc="Filtering")
    while True:
        if iters is not None and i >= iters:
            break
        if k is None:
            S_prime_k = S_prime
        else:
            S_prime_k, _ = pca(S_prime, k)
        select = cov_estimation_filter(S_prime_k, eps, tau, limit=limit)
        # Here we assume that if select is not a boolean array then it signals termination.
        if not isinstance(select, np.ndarray):
            print(f"Terminating early {i} success...")
            break
        if select is None:
            print(f"Terminating early {i} fail...")
            break
        if limit is not None:
            limit = limit - (len(select) - np.sum(select))
            assert limit >= 0
            pbar.update(orig_limit - limit - pbar.n)
        S_prime = S_prime[:, select]
        idxs = idxs[select]
        i += 1
        if limit == 0:
            break
    select_mask = np.zeros(n, dtype=bool)
    select_mask[idxs] = True
    return select_mask


def rcov(S_prime, eps, tau=0.1, k=None, iters=None, limit=None):
    """
    Computes an estimate of the covariance matrix.
    
    Returns:
      (S_selected * S_selected') where S_selected are the columns of S_prime selected by cov_estimation_iterate.
    """
    select = cov_estimation_iterate(S_prime, eps, tau, k, iters=iters, limit=limit)
    selected = S_prime[:, select]
    return selected @ selected.T


def rpca(S_prime, eps, tau=0.1, k=100, iters=None, limit=None):
    """
    Robust PCA estimation.
    
    Computes a paired difference matrix from a random permutation of columns,
    applies cov_estimation_iterate, then computes PCA on the selected samples.
    
    Returns a tuple: (U * S_prime, U, covariance_estimate/2)
    """
    d, n = S_prime.shape
    perm = np.random.permutation(n)
    S_perm = S_prime[:, perm]
    half = n // 2
    # If n is odd, discard the extra column
    if n % 2 != 0:
        S_perm = S_perm[:, :n - (n % 2)]
    S_paired = S_perm[:, :half] - S_perm[:, half:2*half]
    if limit is not None:
        limit = int(round(limit - (limit**2) / (2 * n)))
    selected = cov_estimation_iterate(S_paired, eps, tau, iters=iters, limit=limit)
    S_selected = S_paired[:, selected]
    # Compute PCA on the paired difference matrix for selected columns.
    _, U = pca(S_paired[:, selected], k)
    S_selected_transformed = U @ S_paired[:, selected]
    # Compute covariance of (U @ S_paired[:, selected]) with no bias correction, divided by 2.
    cov_est = np.cov((U @ S_paired[:, selected]).T, bias=True) / 2
    return U @ S_prime, U, cov_est


def mean_tail(T, d, eps, delta, tau, nu=1):
    """
    Computes the tail bound for the mean estimation.
    
    In Julia: mean_Tail(T, d, ε, δ, τ, ν=1) = 8exp(-T^2/(2ν)) + 8ε/(T^2*log(d*log(d/(ε*τ))))
    """
    return 8 * math.exp(-T**2 / (2 * nu)) + 8 * eps / (T**2 * math.log(d * math.log(d / (eps * tau))))


def mean_estimation_filter(S_prime, eps, tau=0.1, nu=1, limit=None):
    """
    Filters samples based on a mean estimation criterion.
    
    Parameters:
      S_prime : 2D array (d x n)
      eps, tau, nu : parameters (with defaults as in the Julia code)
      limit : optional removal limit
      
    Returns:
      A boolean mask (of length n) based on the criterion, or None if the filter condition is not met.
    """
    d, n = S_prime.shape
    mu = np.mean(S_prime, axis=1, keepdims=True)
    Sigma = np.cov(S_prime.T, bias=True)
    vals, vecs = eig(Sigma)
    idx = np.argmax(np.abs(vals))
    lambda_val = np.real(vals[idx])
    v = np.real(vecs[:, idx])
    if lambda_val - 1 <= eps * math.log(1/eps):
        return None
    delta = 3 * math.sqrt(eps * (lambda_val - 1))
    # Compute λmags: for each column of (S_prime - mu), compute the absolute inner product with v.
    lambda_mags = np.abs((S_prime - mu).T @ v)
    sorted_indices = np.argsort(lambda_mags)
    sorted_lambda_mags = lambda_mags[sorted_indices]
    for i, mag in enumerate(sorted_lambda_mags):
        if mag < delta:
            continue
        T_val = mag - delta
        if (n - i) / n > mean_tail(T_val, d, eps, delta, tau, nu):
            if limit is None:
                return lambda_mags <= mag
            else:
                return np.logical_or(lambda_mags <= mag,
                                     k_lowest_ind(lambda_mags, max(0, n - limit)))
    return None


def mean_estimation_iterate(A, eps, tau=0.1, nu=1, iters=None, limit=None):
    """
    Iteratively refines the sample selection using mean_estimation_filter.
    
    Returns:
      A boolean mask (of length n) for the original columns of A.
    """
    d, n = A.shape
    idxs = np.arange(n)
    i = 0
    if limit is not None:
        orig_limit = limit
        pbar = tqdm(total=limit, desc="Mean filtering")
    while True:
        if iters is not None and i >= iters:
            break
        select = mean_estimation_filter(A, eps, tau, nu, limit=limit)
        if select is None:
            print(f"Terminating early at iteration {i}...")
            break
        if limit is not None:
            limit = limit - (len(select) - np.sum(select))
            assert limit >= 0
            pbar.update(orig_limit - limit - pbar.n)
        A = A[:, select]
        idxs = idxs[select]
        i += 1
    select_mask = np.zeros(n, dtype=bool)
    select_mask[idxs] = True
    return select_mask


def rmean(A, eps, tau=0.1, nu=1, iters=None, limit=None):
    """
    Returns the mean (along features) of the columns of A selected by mean_estimation_iterate.
    """
    select = mean_estimation_iterate(A, eps, tau, nu, iters=iters, limit=limit)
    return np.mean(A[:, select], axis=1, keepdims=True)
