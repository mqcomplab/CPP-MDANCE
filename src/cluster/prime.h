#pragma once

#include "../tools/bts.h"
#include "../tools/esim.h"
#include "../tools/types.h"

/*
 * PRIME -- Protein Retrieval via Integrative Molecular Ensembles.
 *
 * Given an already-clustered ensemble (per-frame cluster labels), PRIME predicts
 * the representative / "native"-like frame using extended (n-ary) similarity. It
 * does NOT need topology, masses, charges, RMSD or MSD -- only the coordinate
 * matrix and the cluster labels (populations are derived from the labels).
 *
 * Port of mqcomplab/MDANCE PRIME (src/mdance/prime: sim_calc.py, rep_frames.py,
 * normalize.py).  Faithful points that matter:
 *   - similarity uses the RR_nw / SM_nw weighted (fraction) indices computed
 *     directly from MD::Counters (NOT extendedComparison, which returns a
 *     DISSIMILARITY in this codebase). HIGHER = more similar.
 *   - medoid = argMIN of complementary similarity, outlier = argMAX (esim.py
 *     convention -- the MIRROR of bts.cpp's calculateMedoid/Outlier).
 *   - global scalar min-max normalization over the whole matrix (v3).
 *   - c0 = the MOST POPULATED cluster; population weighting drops c0 (weights[1:]).
 *   - calculate_max_key returns the candidate frame whose entry is the global max
 *     across all per-cluster scores (not the per-frame average).
 *
 * All returned indices are rows of the ORIGINAL (pre-normalization) coordinate
 * matrix, i.e. frame indices. A field is -1 when undefined.
 */
struct PrimeResult {
    int pairwise = -1;
    int uni = -1;
    int medoid = -1;
    int outlier = -1;
    int medoid_all = -1;
    int medoid_c0 = -1;
    int medoid_c0_trimmed = -1;
    int nclusters = 0;
};

class Prime {
    Mat X;                 // globally min-max normalized coordinates
    std::vector<int> labels;
    MD::Metric mt;         // RR or SM only
    double trimFrac;
    bool weighted;
    PrimeResult result;

    double simIndex(const ArrayXd& cTotal, int nObjects) const;
    ArrayXd compSim(const ArrayXXd& M) const;       // raw similarity; higher = more similar
    int medoidIdx(const ArrayXXd& M) const;         // argMIN compSim
    int outlierIdx(const ArrayXXd& M) const;         // argMAX compSim
    ArrayXXd gather(const std::vector<int>& rows) const;
    void run();

public:
    Prime(Mat data, std::vector<int> labels, MD::Metric mt,
          double trimFrac = 0.0, bool weighted = true);
    PrimeResult getResult() const { return result; }
};
