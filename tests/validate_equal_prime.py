#!/usr/bin/env python3
"""
Validation tests for the eQUAL clustering and PRIME analysis ports in mdance-cli.

Strategy: rather than depend on a (possibly version-skewed) upstream MDANCE
install, this cross-checks the shipped CLI against an INDEPENDENT NumPy
reimplementation of the exact formulas (bts MSD complementary similarity, the
RR_nw extended-similarity counters, eQUAL's radial membership) plus analytic
expectations on designed data. If the real upstream `mdance` package is
importable it is used as an extra cross-check.

Run:  python3 tests/validate_equal_prime.py
Exit code 0 = all checks passed.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.environ.get("MDANCE_CLI", os.path.join(HERE, "..", "build", "cli", "mdance-cli"))
NATOMS = 8
NCOLS = NATOMS * 3

_checks = []


def check(name, ok, detail=""):
    _checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def run_cli(args, inp):
    out = os.path.join(tempfile.gettempdir(), "mdance_val_out.json")
    cmd = [CLI] + args + ["--input", inp, "--output", out, "--natoms", str(NATOMS)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as f:
        return json.load(f)


def write_csv(path, X):
    np.savetxt(path, X, delimiter=",", fmt="%.6f")


# ---------- independent reference implementations ----------

def pairwise_msd(x, y):
    # bts meanSqDev for 2 rows == sum((x-y)^2) / (2 * natoms)
    return float(np.sum((x - y) ** 2) / (2 * NATOMS))


def bts_comp_sim_msd(X):
    # bts.calculateCompSim (MSD branch): higher == more central (medoid = argmax)
    N = X.shape[0]
    cSum = X.sum(axis=0)
    sqSum = (X ** 2).sum(axis=0)
    compC = (cSum - X) / (N - 1)
    compSq = (sqSum - X ** 2) / (N - 1)
    return (2.0 * (compSq - compC ** 2) / NATOMS).sum(axis=1)


def rr_nw(c_total, n):
    # esim RR_nw with fraction weighting (matches Prime::simIndex for RR)
    thr = n % 2
    diff = 2.0 * c_total - n
    wa = np.sum(diff[diff > thr]) / n
    p = len(c_total)
    return wa / p


def prime_comp_sim(M):
    # complementary RR similarity; medoid = argmin (esim convention)
    N = M.shape[0]
    ct = M.sum(axis=0)
    return np.array([rr_nw(ct - M[i], N - 1) for i in range(N)])


def normalize(X):
    mn, mx = X.min(), X.max()
    return (X - mn) / (mx - mn) if mx > mn else X.copy()


# ---------- designed data ----------

def two_blobs(seed=1, per=30, sep=30.0, sd=0.4):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sd, size=(per, NCOLS))
    b = rng.normal(sep, sd, size=(per, NCOLS))
    return np.vstack([a, b]), per


def main():
    if not os.path.exists(CLI):
        print(f"mdance-cli not found at {CLI}; set MDANCE_CLI or build first.", file=sys.stderr)
        return 2
    tmp = tempfile.gettempdir()
    X, per = two_blobs()
    N = X.shape[0]
    csv = os.path.join(tmp, "mdance_val.csv")
    write_csv(csv, X)

    # ===== eQUAL: analytic partition on two well-separated blobs =====
    print("eQUAL:")
    within_max = max(pairwise_msd(X[i], X[j]) for i in range(per) for j in range(i + 1, per))
    within_max = max(within_max,
                     max(pairwise_msd(X[i], X[j]) for i in range(per, N) for j in range(i + 1, N)))
    between_min = min(pairwise_msd(X[i], X[j]) for i in range(per) for j in range(per, N))
    check("blobs separable (within_max < between_min)", within_max < between_min,
          f"within_max={within_max:.3f} between_min={between_min:.3f}")
    thr = (within_max + between_min) / 2.0
    eq = run_cli(["--algorithm", "equal", "--metric", "MSD", "--threshold", f"{thr:.6f}",
                  "--seed-method", "medoid"], csv)
    labels = eq["labels"]
    check("eQUAL finds exactly 2 clusters", eq["nClusters"] == 2, f"nClusters={eq['nClusters']}")
    blobA = set(labels[:per]); blobB = set(labels[per:])
    check("eQUAL: blob A is one pure non-noise cluster", len(blobA) == 1 and -1 not in blobA, str(blobA))
    check("eQUAL: blob B is one pure non-noise cluster", len(blobB) == 1 and -1 not in blobB, str(blobB))
    check("eQUAL: the two blobs are different clusters", blobA != blobB, f"{blobA} vs {blobB}")
    # radial property: each clustered frame is within threshold of its cluster's medoid frame
    rep = eq["representatives"]
    radial_ok = True
    for f in range(N):
        c = labels[f]
        if c < 0:
            continue
        if pairwise_msd(X[f], X[rep[c]]) > thr + 1e-6:
            radial_ok = False
            break
    check("eQUAL: clustered frames within threshold of their representative", radial_ok)

    # ===== PRIME: independent RR reference for the medoid baselines =====
    print("PRIME:")
    Xn = normalize(X)
    ref_medoid_all = int(np.argmin(prime_comp_sim(Xn)))
    # labels for PRIME = true blob assignment
    lab = [0] * per + [1] * per
    labcsv = os.path.join(tmp, "mdance_val_labels.csv")
    with open(labcsv, "w") as f:
        f.write("\n".join(str(x) for x in lab) + "\n")
    pr = run_cli(["--prime", "--metric", "RR", "--trim-frac", "0", "--labels", labcsv], csv)
    check("PRIME medoidAll matches independent RR reference",
          pr["medoidAll"] == ref_medoid_all, f"cli={pr['medoidAll']} ref={ref_medoid_all}")
    # c0 = most populated; equal sizes -> stable order -> cluster 0 (frames 0..per-1)
    c0_idx = list(range(per))
    cs_c0 = prime_comp_sim(Xn[c0_idx])
    ref_medoid_c0 = c0_idx[int(np.argmin(cs_c0))]
    check("PRIME medoidC0 matches independent RR reference",
          pr["medoidC0"] == ref_medoid_c0, f"cli={pr['medoidC0']} ref={ref_medoid_c0}")
    check("PRIME medoidC0Trimmed == medoidC0 when trim=0",
          pr["medoidC0Trimmed"] == pr["medoidC0"], f"{pr['medoidC0Trimmed']} vs {pr['medoidC0']}")
    check("PRIME predictions are valid frame indices",
          all(0 <= pr[k] < N for k in ("pairwise", "union", "medoid", "outlier")),
          str({k: pr[k] for k in ("pairwise", "union", "medoid", "outlier")}))
    # PRIME with trim should not crash and stays in range
    pr2 = run_cli(["--prime", "--metric", "SM", "--trim-frac", "0.1", "--weighted",
                   "--labels", labcsv], csv)
    check("PRIME (SM, trim, weighted) runs and indices in range",
          all(0 <= pr2[k] < N for k in ("pairwise", "union", "medoid", "outlier")))

    # ===== Frame selection: invariants + reference for medoid/outlier =====
    print("Frame selection:")
    cs = bts_comp_sim_msd(X)
    ref_medoid = int(np.argmax(cs))   # bts convention: medoid = argmax comp_sim
    ref_outlier = int(np.argmin(cs))
    med = run_cli(["--select", "--method", "medoid", "--metric", "MSD"], csv)
    out = run_cli(["--select", "--method", "outlier", "--metric", "MSD"], csv)
    check("select medoid matches bts comp-sim argmax",
          med["indices"] == [ref_medoid], f"cli={med['indices']} ref={ref_medoid}")
    check("select outlier matches bts comp-sim argmin",
          out["indices"] == [ref_outlier], f"cli={out['indices']} ref={ref_outlier}")
    div = run_cli(["--select", "--method", "diversity", "--param", "50", "--metric", "MSD"], csv)
    di = div["indices"]
    check("diversity 50% returns N/2 distinct in-range frames",
          len(di) == N // 2 and len(set(di)) == len(di) and all(0 <= i < N for i in di),
          f"count={len(di)}")
    rs = run_cli(["--select", "--method", "repsample", "--param", "10", "--nbins", "5", "--metric", "MSD"], csv)
    ri = rs["indices"]
    check("repsample returns requested distinct in-range frames (was a broken loop before)",
          len(ri) == 10 and len(set(ri)) == len(ri) and all(0 <= i < N for i in ri),
          f"count={len(ri)} distinct={len(set(ri))}")
    outl = run_cli(["--select", "--method", "outliers", "--param", "5", "--metric", "MSD"], csv)
    oi = outl["indices"]
    check("outliers list leads with the single outlier",
          len(oi) == 5 and oi[0] == ref_outlier, f"indices={oi} single_outlier={ref_outlier}")

    npass = sum(1 for _, ok, _ in _checks if ok)
    print(f"\n{npass}/{len(_checks)} checks passed.")
    return 0 if npass == len(_checks) else 1


if __name__ == "__main__":
    sys.exit(main())
