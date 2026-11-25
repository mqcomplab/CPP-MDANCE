"""MDANCE (Python) timed on the same calls as bench/mdance_bench.c.

    python mdance_py_bench.py data.csv natoms out.txt [only]

Each clustering is followed by what the C API computes for every result:
each cluster's medoid (its representative) and MSD, and the Calinski-
Harabasz and Davies-Bouldin scores. Labels go to out.txt for comparison.
Timings to standard output, in the C program's format.
"""
import os
import sys
import tempfile
import time
import warnings

import numpy as np

from mdance.cluster.nani import KmeansNANI
from mdance.cluster.helm import HELM
from mdance.cluster.divine import Divine
from mdance.cluster.equal import ExtendedQuality
from mdance.tools import bts
from mdance.prime.sim_calc import FrameSimilarity
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings("ignore")

path, natoms, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
only = sys.argv[4] if len(sys.argv) > 4 else ""
t0 = time.perf_counter()
X = np.loadtxt(path, delimiter=",")
print(f"{X.shape[0]} frames, {X.shape[1]} columns, {natoms} atoms (read in {1000*(time.perf_counter()-t0):.0f} ms)")
out = open(outp, "w")


def want(tag):
    return not only or tag in only


def report(name, ms):
    print(f"{name:34s} {ms:10.1f} ms", flush=True)


def common(labels):
    """What every C-API result carries."""
    labels = np.asarray(labels)
    ks = sorted(set(labels.tolist()) - {-1})
    reps, msds = [], []
    for k in ks:
        m = np.where(labels == k)[0]
        reps.append(int(m[bts.calculate_medoid(X[m], "MSD", N_atoms=natoms)]))
        msds.append(bts.mean_sq_dev(X[m], N_atoms=natoms) if len(m) > 1 else 0.0)
    keep = labels >= 0
    if len(ks) > 1:
        ch = calinski_harabasz_score(X[keep], labels[keep])
        db = davies_bouldin_score(X[keep], labels[keep])
    else:
        ch = db = 0.0
    return reps, msds, ch, db


def timed(name, fn):
    t = time.perf_counter()
    labels = fn()
    common(labels)
    ms = 1000 * (time.perf_counter() - t)
    report(name, ms)
    out.write(name + " labels=" + ",".join(str(int(v)) for v in labels) + "\n")
    out.flush()
    return labels


def nani(k, init, pct):
    def run():
        km = KmeansNANI(data=X, n_clusters=k, metric="MSD", N_atoms=natoms, init_type=init, percentage=pct)
        labels, centers, n_iter = km.execute_kmeans_all()
        return labels
    return run


if want("kmeans"):
    for init, cname in [("strat_all", "StratAll"), ("strat_reduced", "StratReduced"),
                        ("comp_sim", "CompSim"), ("div_select", "DivSelect")]:
        timed(f"kmeans k=10 {cname}", nani(10, init, 10))
    t = time.perf_counter()
    for k in range(2, 21):
        nani(k, "strat_all", 10)()
        # (the scores, as the C API's scan records them)
    report("kmeans scan k=2..20 (total)", 1000 * (time.perf_counter() - t))

if want("helm"):
    km = KmeansNANI(data=X, n_clusters=40, metric="MSD", N_atoms=natoms, init_type="strat_all", percentage=50)
    init, _, _ = km.execute_kmeans_all()

    def helm(scheme):
        def run():
            starts = []
            for c in sorted(set(init.tolist())):
                m = X[init == c]
                starts.append([[c], (np.sum(m, axis=0), np.sum(m ** 2, axis=0)), len(m)])
            h = HELM({len(starts): starts}, metric="MSD", N_atoms=natoms, merge_scheme=scheme, n_clusters=5)
            res = h.run()
            final = res[sorted(res)[0]]
            lab = np.full(len(X), -1)
            for i, cl in enumerate(final):
                for c in cl[0]:
                    lab[init == c] = i
            return lab
        return run
    timed("helm k=5 from 40", helm("inter"))

if want("divine"):
    for anchors, cname, refine in [("nani", "NANI", False), ("outlier_pair", "OutlierPair", False),
                                   ("splinter_split", "SplinterPair", False), ("outlier_pair", "OutlierPair refine", True)]:
        def run(anchors=anchors, refine=refine):
            d = Divine(X, split="weighted_MSD", anchors=anchors, init_type="strat_all", end="k", k=10,
                       refine=refine, N_atoms=natoms, threshold=0, percentage=10)
            clusters, labels, scores, msds = d.run()
            return labels
        timed(f"divine k=10 {cname}", run)

if want("equal"):
    th = float(os.environ.get("BENCH_EQUAL_THRESHOLD", "2"))
    for seed in ["medoid", "comp_sim"]:
        if seed == "comp_sim" and os.environ.get("BENCH_SKIP_COMPSIM"):
            continue
        def run(seed=seed):
            eq = ExtendedQuality(X, th, "MSD", natoms, seed_method=seed, n_seeds=1, check_sim=False,
                                 reject_lowd=False, percentage=10)
            res = eq.run()
            # (MDANCE returns each cluster's rows of coordinates, not frame
            # numbers: each row is found again by its bytes)
            row_of = {}
            for f in range(len(X)):
                row_of.setdefault(X[f].tobytes(), []).append(f)
            lab = np.full(len(X), -1)
            for i, members in res.items():
                for m in members:
                    frames = row_of[np.asarray(m, dtype=X.dtype).tobytes()]
                    lab[frames.pop(0) if len(frames) > 1 else frames[0]] = i
            return lab
        timed(f"equal {seed} threshold={th:g}", run)

if want("analysis") or want("prime"):
    km = KmeansNANI(data=X, n_clusters=10, metric="MSD", N_atoms=natoms, init_type="strat_all", percentage=10)
    lab, _, _ = km.execute_kmeans_all()
    if want("analysis"):
        t = time.perf_counter()
        bts.extended_comparison(X, data_type="full", metric="MSD", N_atoms=natoms)
        for k in sorted(set(lab.tolist())):
            m = X[lab == k]
            bts.extended_comparison(m, data_type="full", metric="MSD", N_atoms=natoms)
            bts.calculate_outlier(m, "MSD", N_atoms=natoms)
        report("analysis MSD", 1000 * (time.perf_counter() - t))
    if want("prime"):
        # MDANCE's PRIME reads normalized cluster files: written first (not
        # timed), largest cluster as c0, as its workflow does
        tmp = tempfile.mkdtemp()
        Xn = (X - X.min()) / (X.max() - X.min())
        order = sorted(set(lab.tolist()), key=lambda k: -np.sum(lab == k))
        for i, k in enumerate(order):
            np.save(os.path.join(tmp, f"normed_clusttraj.c{i}.npy"), Xn[lab == k])
        for nary in ["RR", "SM"]:
            t = time.perf_counter()
            fs = FrameSimilarity(cluster_folder=tmp, trim_frac=0.1, n_clusters=len(order),
                                 weighted_by_frames=False, n_ary=nary, weight="nw")
            fs.calculate_pairwise()
            fs.calculate_union()
            fs.calculate_medoid()
            fs.calculate_outlier()
            report(f"prime {nary}", 1000 * (time.perf_counter() - t))

if want("select"):
    N = len(X)
    for meth in ["diversity", "outliers", "repsample", "medoid", "outlier"]:
        t = time.perf_counter()
        if meth == "diversity":
            bts.diversity_selection(X, 10, "MSD", N_atoms=natoms, method="comp_sim", start="medoid")
        elif meth == "outliers":
            cs = bts.calculate_comp_sim(X, "MSD", N_atoms=natoms)
            np.argsort(cs)[:10]
        elif meth == "repsample":
            bts.rep_sample(X, metric="MSD", N_atoms=natoms, n_bins=10, n_samples=10)
        elif meth == "medoid":
            bts.calculate_medoid(X, "MSD", N_atoms=natoms)
        else:
            bts.calculate_outlier(X, "MSD", N_atoms=natoms)
        report(f"select {meth} MSD", 1000 * (time.perf_counter() - t))
out.close()
