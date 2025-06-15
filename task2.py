import numpy as np
import os
from scipy.stats import ttest_rel

results = np.load("wyniki_klasyfikacji.npy")

folder = "datasets"
file_names = os.listdir(folder)
max_samples = 0
max_idx = 0

for i, fname in enumerate(file_names):
	data = np.loadtxt(os.path.join(folder, fname), delimiter=',')
	if data.shape[0] > max_samples:
	    max_samples = data.shape[0]
	    max_idx = i

print(f"Najwięcej obiektów ma plik: {file_names[max_idx]} ({max_samples} obiektów)")


subset_results = results[max_idx]
n_classifiers = subset_results.shape[1]

t_stat = np.full((n_classifiers, n_classifiers), np.nan)
p_val = np.full((n_classifiers, n_classifiers), np.nan)
better = np.zeros((n_classifiers, n_classifiers), dtype=bool)
stat_significant = np.zeros((n_classifiers, n_classifiers), dtype=bool)
stat_better = np.zeros((n_classifiers, n_classifiers), dtype=bool)


for i in range(n_classifiers):
    for j in range(n_classifiers):
        if i != j:
            t, p = ttest_rel(subset_results[:, i], subset_results[:, j])
            t_stat[i, j] = t
            p_val[i, j] = p
            better[i, j] = np.mean(subset_results[:, i]) > np.mean(subset_results[:, j])
            stat_significant[i, j] = p < 0.05
            stat_better[i, j] = better[i, j] and stat_significant[i, j]


np.set_printoptions(precision=4, suppress=True)

print("\nt-statistic:\n", t_stat)
print("\np-value:\n", p_val)
print("\nBetter matrix:\n", better)
print("\nSignificant matrix:\n", stat_significant)
print("\nSignificantly better matrix:\n", stat_better)


mean_scores = np.mean(subset_results, axis=0)
clf_names = ["GaussianNB", "KNN", "MLP"]

for i in range(n_classifiers):
    for j in range(n_classifiers):
        if i != j and stat_better[i, j]:
            print(f"{clf_names[i]} with {mean_scores[i]:.3f} better than {clf_names[j]} with {mean_scores[j]:.3f}")