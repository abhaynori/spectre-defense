#!/usr/bin/env python3
import sys
import os
import re
import numpy as np
from util_jl_to import pca, sharp, k_lowest_ind
from kmeans_filters import kmeans_filter2, k_lowest_ind  # k_lowest_ind defined in kmeans_filters.py
from quantum_filters import rcov_auto_quantum_filter

def main():
    if len(sys.argv) < 2:
        print("Usage: run_filters.py <name1> <name2> ...")
        sys.exit(1)

    log_file_path = "run_filters.log"
    with open(log_file_path, "a") as log_file:
        for name in sys.argv[1:]:
            # Extract target_label: assuming the third part (separated by "-") contains a number.
            parts = name.split("-")
            if len(parts) < 3:
                print(f"Invalid name format: {name}")
                continue
            try:
                # This mimics Julia's split(name, "-")[3][end:end] by taking the last character of the third part.
                target_label = int(parts[2][-1])
            except ValueError:
                print(f"Could not parse target_label from {name}")
                continue

            # Load the representations array from file and transpose it.
            reps_path = os.path.join("output", name, f"label_{target_label}_reps.npy")
            try:
                reps = np.load(reps_path).T  # transpose to match Julia's '
            except Exception as e:
                print(f"Error loading {reps_path}: {e}")
                continue

            n = reps.shape[1]
            # Extract eps from the digits at the end of the name.
            m = re.search(r"[0-9]+$", name)
            if not m:
                print(f"Could not find eps in {name}")
                continue
            eps = int(m.group(0))
            removed = int(round(1.5 * eps))

            # --- PCA Filter ---
            print(f"{name}: Running PCA filter")
            reps_pca, U = pca(reps, num_components=1)
            # Compute a vector based on the absolute difference from its mean (negated)
            pc_row = reps_pca[0, :]
            mean_pc = np.mean(pc_row)
            values = -np.abs(mean_pc - pc_row)
            k_val = int(round(1.5 * eps))
            pca_poison_mask = k_lowest_ind(values, k_val)
            poison_removed = np.sum(pca_poison_mask[-eps:])
            clean_removed = removed - poison_removed
            print(f"PCA filter - poison_removed: {poison_removed}, clean_removed: {clean_removed}")
            log_file.write(f"{name}-pca: {poison_removed}, {clean_removed}\n")
            np.save(os.path.join("output", name, "mask-pca-target.npy"), pca_poison_mask)

            # --- k-means Filter ---
            print(f"{name}: Running kmeans filter")
            # kmeans_filter2 returns a boolean mask (True for clean samples)
            kmeans_clean_mask = kmeans_filter2(reps, eps)
            kmeans_poison_mask = np.logical_not(kmeans_clean_mask)
            poison_removed = np.sum(kmeans_poison_mask[-eps:])
            clean_removed = removed - poison_removed
            print(f"kmeans filter - poison_removed: {poison_removed}, clean_removed: {clean_removed}")
            log_file.write(f"{name}-kmeans: {poison_removed}, {clean_removed}\n")
            np.save(os.path.join("output", name, "mask-kmeans-target.npy"), kmeans_poison_mask)

            # --- Quantum Filter ---
            print(f"{name}: Running quantum filter")
            quantum_clean_mask = rcov_auto_quantum_filter(reps, eps)
            quantum_poison_mask = np.logical_not(quantum_clean_mask)
            poison_removed = np.sum(quantum_poison_mask[-eps:])
            clean_removed = removed - poison_removed
            print(f"quantum filter - poison_removed: {poison_removed}, clean_removed: {clean_removed}")
            log_file.write(f"{name}-quantum: {poison_removed}, {clean_removed}\n")
            np.save(os.path.join("output", name, "mask-rcov-target.npy"), quantum_poison_mask)

if __name__ == "__main__":
    main()
