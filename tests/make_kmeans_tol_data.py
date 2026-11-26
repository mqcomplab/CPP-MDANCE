"""A small dataset on which MDANCE's KmeansNANI (scikit-learn's KMeans, tol
1e-4) stops before the labels settle: overlapping blobs, values written to 4
decimals and read back, so MDANCE and the C++ test read the same numbers.
Writes <dir>/kmeans_tol.csv and kmeans_tol_py_labels.csv (MDANCE's labels,
comp_sim, k as printed, N_atoms 1, percentage 10).
Usage: python make_tol_data.py <dir>"""
import sys, os, warnings
import numpy as np
from sklearn.cluster import KMeans
from mdance.cluster.nani import KmeansNANI
warnings.filterwarnings('ignore')
out = sys.argv[1]
for seed in range(60):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1.0, size=(8, 4))
    X = np.vstack([c + rng.normal(0, 1.0, size=(200, 4)) for c in centres])
    f = os.path.join(out, 'kmeans_tol.csv')
    np.savetxt(f, X, delimiter=',', fmt='%.3f')
    X = np.loadtxt(f, delimiter=',')
    for k in (6, 9, 12):
        m = KmeansNANI(data=X, n_clusters=k, metric='MSD', N_atoms=1, init_type='comp_sim', percentage=10)
        try:
            lab, _, it = m.execute_kmeans_all()
            seeds = m.initiate_kmeans()
        except ValueError:
            continue
        full = KMeans(k, init=seeds, n_init=1, tol=0.0, max_iter=10000).fit(X)
        d = int((lab != full.labels_).sum())
        if d >= 3:
            np.savetxt(os.path.join(out, 'kmeans_tol_py_labels.csv'), lab, fmt='%d')
            print(f'seed {seed} k={k}: MDANCE {it} iterations, to convergence {full.n_iter_}; {d} frames differ')
            sys.exit(0)
print('none found')
