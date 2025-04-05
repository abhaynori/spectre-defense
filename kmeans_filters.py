import numpy as np
from sklearn.cluster import KMeans
from util_jl_to import pca, sharp, k_lowest_ind

def ind_to_indexes(c):
    """
    Returns the indices of array c where the value is nonzero (or True).
    Equivalent to the Julia function ind_to_indexes.
    """
    return np.flatnonzero(c)

def sample_ind_wo_replacement(c, k):
    """
    Given a boolean array c, randomly select k indices (without replacement)
    from those where c is True, and return a boolean array of the same shape as c
    with only the selected indices set to True.
    """
    a = np.zeros(c.shape, dtype=bool)
    idxs = np.flatnonzero(c)
    if len(idxs) < k:
        k = len(idxs)
    perm = np.random.permutation(len(idxs))
    selected = idxs[perm[:k]]
    a[selected] = True
    return a

def kmeans_filter1(reps, eps):
    """
    Applies k-means clustering (with 2 clusters) on the representations (reps)
    and computes the ratio of “poisoned” vs. “clean” samples in the last eps samples.
    Returns a randomly sampled index from the cluster deemed as “good.”
    
    Parameters:
      reps: A 2D NumPy array of shape (features, samples).
      eps: An integer indicating the number of samples at the end to consider as poisoned.
    """
    n = reps.shape[1]
    # Transpose reps so that samples are rows (required by scikit-learn)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(reps.T)
    # In Python, cluster labels are 0 and 1
    kmass = kmeans.labels_
    c1 = (kmass == 0)
    c2 = (kmass == 1)
    # Compute counts in the first n-eps samples (assumed clean) and last eps samples (assumed poisoned)
    c1c = np.sum(c1[:n-eps])
    c1p = np.sum(c1[n-eps:])
    c2c = np.sum(c2[:n-eps])
    c2p = np.sum(c2[n-eps:])
    # Choose the cluster with the lower ratio of "poison" samples as the "good" cluster.
    if (c1p / (c1p + c1c)) > (c2p / (c2p + c2c)):
        good = c1
    else:
        good = c2
    # Randomly select one index from the good cluster
    good_indexes = np.flatnonzero(good)
    if len(good_indexes) == 0:
        return None
    return np.random.choice(good_indexes)

def kmeans_filter2(reps, eps, k=100, limit=1.5):
    """
    Repeatedly applies kmeans_filter1 (on PCA-reduced representations)
    until a target number of indices have been marked for removal.
    Then returns a boolean mask over all samples, with indices in the removal set set to False.
    
    Parameters:
      reps: A 2D NumPy array of shape (features, samples).
      eps: An integer used to determine how many samples at the end are considered poisoned.
      k: (Optional) Not used in this function; kept for compatibility.
      limit: A multiplier applied to eps to determine the removal target.
    """
    # Reduce dimensionality to 100 using PCA (taking only the PCA scores)
    reps_pca, _ = pca(reps, num_components=100)
    to_remove = set()
    n = 0
    target = int(round(limit * eps))
    while len(to_remove) < target:
        i = kmeans_filter1(reps_pca, eps)
        if i is None:
            break
        to_remove.add(i)
        n += 1
        if n > 10000:
            break
    # Create a boolean mask of length equal to the number of samples (all True by default)
    ind = np.ones(reps.shape[1], dtype=bool)
    # Mark indices for removal (set to False)
    for i in to_remove:
        if 0 <= i < len(ind):
            ind[i] = False
    return ind
