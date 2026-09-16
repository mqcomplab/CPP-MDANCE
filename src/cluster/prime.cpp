#include "prime.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

Prime::Prime(Mat data, std::vector<int> labels, MD::Metric mt,
             double trimFrac, bool weighted)
    : labels(std::move(labels)), mt(mt), trimFrac(trimFrac), weighted(weighted)
{
    // v3 normalization: single scalar min/max over the WHOLE matrix.
    double mn = data.minCoeff();
    double mx = data.maxCoeff();
    double range = mx - mn;
    if (range > 0.0) {
        X = (data - mn) / range;
    } else {
        X = data;  // constant matrix: leave as-is
    }
    run();
}

// RR_nw = w_a / p ; SM_nw = total_w_sim / p, computed via the fraction-weighted
// counters (wFactor = 0). HIGHER = more similar.
double Prime::simIndex(const ArrayXd& cTotal, int nObjects) const {
    if (nObjects <= 0) return 0.0;
    MD::Threshold th(MD::ThresholdType::None, nObjects);  // value = nObjects % 2
    MD::Counters cnt = calculateCounters(cTotal, nObjects, th, 0);
    if (cnt.p == 0) return 0.0;
    if (mt == MD::Metric::SM) return cnt.totalWsim / cnt.p;
    return cnt.wa / cnt.p;  // default RR
}

// Complementary similarity of each row: similarity of the set with that row removed.
ArrayXd Prime::compSim(const ArrayXXd& M) const {
    int N = (int)M.rows();
    ArrayXd out(N);
    if (N <= 1) { out.setZero(); return out; }
    ArrayXd cTotal = M.colwise().sum();
    for (int i = 0; i < N; ++i) {
        ArrayXd comp = cTotal - M.row(i).transpose();
        out(i) = simIndex(comp, N - 1);
    }
    return out;
}

int Prime::medoidIdx(const ArrayXXd& M) const {
    if (M.rows() <= 1) return 0;
    ArrayXd cs = compSim(M);
    Index idx;
    cs.minCoeff(&idx);   // medoid = argmin (esim.py convention)
    return (int)idx;
}

int Prime::outlierIdx(const ArrayXXd& M) const {
    if (M.rows() <= 1) return 0;
    ArrayXd cs = compSim(M);
    Index idx;
    cs.maxCoeff(&idx);   // outlier = argmax
    return (int)idx;
}

ArrayXXd Prime::gather(const std::vector<int>& rows) const {
    ArrayXXd m((Index)rows.size(), X.cols());
    for (int i = 0; i < (int)rows.size(); ++i) m.row(i) = X.row(rows[i]);
    return m;
}

void Prime::run() {
    int nframes = (int)X.rows();

    int nClusters = 0;
    for (int l : labels) if (l + 1 > nClusters) nClusters = l + 1;

    // members per cluster
    std::vector<std::vector<int>> members(nClusters);
    for (int j = 0; j < (int)labels.size() && j < nframes; ++j) {
        if (labels[j] >= 0) members[labels[j]].push_back(j);
    }
    // non-empty clusters ordered by population DESCENDING -> order[0] == c0
    std::vector<int> order;
    for (int c = 0; c < nClusters; ++c) if (!members[c].empty()) order.push_back(c);
    std::stable_sort(order.begin(), order.end(),
        [&](int a, int b) { return members[a].size() > members[b].size(); });
    result.nclusters = (int)order.size();

    // --- baseline: medoid over ALL frames (row index == frame index) ---
    {
        ArrayXd cs = compSim(X);
        Index idx;
        cs.minCoeff(&idx);
        result.medoid_all = (int)idx;
    }

    if (order.empty()) return;

    // --- c0 = most populated cluster ---
    std::vector<int> c0rows = members[order[0]];
    ArrayXXd c0mat = gather(c0rows);
    ArrayXd csC0 = compSim(c0mat);
    {
        int medLocal = medoidIdx(c0mat);
        result.medoid_c0 = c0rows[medLocal];
    }

    // --- trim c0: drop the floor(n*trimFrac) rows with HIGHEST comp_sim ---
    std::vector<int> c0t = c0rows;
    int n_c0 = (int)c0rows.size();
    int cutoff = (trimFrac > 0.0) ? (int)std::floor(n_c0 * trimFrac) : 0;
    if (cutoff > 0 && cutoff < n_c0) {
        std::vector<int> idx(n_c0);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return csC0(a) > csC0(b); });   // highest first
        c0t.clear();
        for (int i = cutoff; i < n_c0; ++i) c0t.push_back(c0rows[idx[i]]);  // keep the rest
    }
    ArrayXXd c0tmat = gather(c0t);
    {
        int medLocal = medoidIdx(c0tmat);
        result.medoid_c0_trimmed = c0t[medLocal];
    }

    // --- the 4 PRIME scorers need at least one non-c0 cluster ---
    if (order.size() < 2) return;

    // precompute each non-c0 cluster's submatrix, column sum, medoid/outlier rows, population
    struct CK {
        ArrayXXd mat;
        ArrayXd colsum;
        ArrayXd medoidRow;
        ArrayXd outlierRow;
        double pop;
    };
    std::vector<CK> cks;
    cks.reserve(order.size() - 1);
    for (size_t oi = 1; oi < order.size(); ++oi) {
        const std::vector<int>& mem = members[order[oi]];
        CK ck;
        ck.mat = gather(mem);
        ck.colsum = ck.mat.colwise().sum();
        ck.medoidRow = ck.mat.row(medoidIdx(ck.mat)).transpose();
        ck.outlierRow = ck.mat.row(outlierIdx(ck.mat)).transpose();
        ck.pop = (double)mem.size();
        cks.push_back(std::move(ck));
    }

    const double NEG_INF = -std::numeric_limits<double>::infinity();
    double bestPair = NEG_INF, bestUni = NEG_INF, bestMed = NEG_INF, bestOut = NEG_INF;
    int idxPair = -1, idxUni = -1, idxMed = -1, idxOut = -1;

    for (int fi = 0; fi < (int)c0t.size(); ++fi) {
        ArrayXd x = c0tmat.row(fi).transpose();
        int frame = c0t[fi];
        for (const CK& ck : cks) {
            double w = weighted ? ck.pop : 1.0;

            // pairwise: avg over y in ck of sim(x + y, 2)
            double tot = 0.0;
            int m = (int)ck.mat.rows();
            for (int y = 0; y < m; ++y) {
                tot += simIndex(x + ck.mat.row(y).transpose(), 2);
            }
            double sp = (m > 0 ? tot / m : 0.0) * w;
            if (sp > bestPair) { bestPair = sp; idxPair = frame; }

            // union: sim(sum(ck) + x, len(ck)+1)
            double su = simIndex(ck.colsum + x, (int)ck.pop + 1) * w;
            if (su > bestUni) { bestUni = su; idxUni = frame; }

            // medoid: sim(ck_medoid + x, 2)
            double sm = simIndex(ck.medoidRow + x, 2) * w;
            if (sm > bestMed) { bestMed = sm; idxMed = frame; }

            // outlier: sim(ck_outlier + x, 2)
            double so = simIndex(ck.outlierRow + x, 2) * w;
            if (so > bestOut) { bestOut = so; idxOut = frame; }
        }
    }
    result.pairwise = idxPair;
    result.uni = idxUni;
    result.medoid = idxMed;
    result.outlier = idxOut;
}
