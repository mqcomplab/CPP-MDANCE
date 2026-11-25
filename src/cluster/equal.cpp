#include "equal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

#include "../tools/scores.h"

Equal::Equal(Mat data, MD::Metric mt, double threshold, int nAtoms,
             MD::EqualSeed seedMethod, double nSeeds, bool checkSim,
             double simThreshold, bool rejectLowd, double minSamples,
             int percentage, MD::AlignMethod alignMethod)
    : data(data), bufC(data.cols()), bufS(data.cols()), mt(mt), seedMethod(seedMethod), threshold(threshold),
      nAtoms(nAtoms), checkSim(checkSim), simThreshold(simThreshold),
      rejectLowd(rejectLowd), percentage(percentage), alignMethod(alignMethod)
{
    nTotal = (int)this->data.rows();
    if (nTotal == 0) {
        createLabels();
        return;
    }
    if (nAtoms <= 0) {
        throw std::invalid_argument("eQUAL: nAtoms must be positive.");
    }
    if (!std::isfinite(threshold) || threshold < 0) {
        throw std::invalid_argument("eQUAL: the threshold must be a number >= 0.");
    }
    if (!std::isfinite(nSeeds) || !std::isfinite(minSamples)) {
        throw std::invalid_argument("eQUAL: n_seeds and min_samples must be numbers.");
    }
    // (clamped before the casts: a double past INT_MAX cast to int is undefined)
    nSeeds = std::min(std::max(nSeeds, 0.0), (double)nTotal);
    minSamples = std::min(std::max(minSamples, 0.0), (double)nTotal + 1);

    // n_seeds / min_samples dual typing: value in (0,1) is a fraction of nTotal,
    // a value >= 1 is a literal count.
    nSeedsInt = (nSeeds > 0 && nSeeds < 1) ? (int)(nTotal * nSeeds) : (int)nSeeds;
    if (nSeedsInt < 1) nSeedsInt = 1;
    if (nSeedsInt >= nTotal) nSeedsInt = nTotal - 1 > 0 ? nTotal - 1 : 1;

    minSamplesInt = (minSamples > 0 && minSamples < 1) ? (int)(nTotal * minSamples) : (int)minSamples;
    if (minSamplesInt < 1) minSamplesInt = 1;

    growClusters();
    createLabels();
}

// The radial comparison of two frames, extendedComparison of the two rows
// as it computed it (their column sums, then the MSD or the index of those),
// in two buffers made once: it allocated a 2-row matrix and two vectors for
// every pair, the bulk of eQual's time.
double Equal::pairDistance(Index a, Index b) {
    const Index D = data.cols();
    Eigen::Map<const ArrayXd> x(data.row(a).data(), D);
    Eigen::Map<const ArrayXd> y(data.row(b).data(), D);
    bufC = x + y;
    if (mt == MD::Metric::MSD) {
        bufS = x.square() + y.square();
        return msdCondensed(bufC, bufS, 2, nAtoms);
    }
    MD::Threshold th(MD::ThresholdType::None, 1);
    th.value = 0;   // 2 % 2
    MD::Indices idx = genSimIdx(bufC, 2, th, 0);
    return 1 - idx.getIndex(mt);
}

double Equal::intraSim(const vector<Index>& members) {
    if (members.empty()) return 0.0;
    ArrayXXd m((Index)members.size(), data.cols());
    for (int i = 0; i < (int)members.size(); ++i) m.row(i) = data.row(members[i]);
    return extendedComparison(m, (Index)members.size(), nAtoms, false, mt);
}

vector<Index> Equal::chooseSeeds(const ArrayXXd& work, const vector<Index>& activeVec) {
    int na = (int)work.rows();
    vector<Index> seeds;

    // complementary similarity over the current active set; in this codebase
    // (bts.cpp) HIGHER comp_sim == more central, so the medoid is the argmax.
    // Only the first rows of the order are used: topCompSimOrder gives them
    // without sorting every row (the first rows a full sort gives).
    if (seedMethod == MD::EqualSeed::Medoid) {
        int take = std::min(nSeedsInt, na);
        vector<int> order = topCompSimOrder(work, take, nAtoms, mt);
        for (int i = 0; i < take; ++i) seeds.push_back(activeVec[order[i]]);
        return seeds;
    }

    // comp_sim: take the high-density region (size n_max, based on the FIXED
    // original n_objects per upstream), then diversity-select the whole subset;
    // every selected point becomes a seed.
    int nMax = (int)(percentage * (double)nTotal / 100.0);
    if (nMax > na) nMax = na;
    if (nMax < 1) nMax = 1;
    vector<int> order = topCompSimOrder(work, nMax, nAtoms, mt);

    ArrayXXd subset(nMax, work.cols());
    vector<Index> subToActive(nMax);
    for (int i = 0; i < nMax; ++i) {
        subset.row(i) = work.row(order[i]);
        subToActive[i] = activeVec[order[i]];
    }
    if (nMax < 2) {
        seeds.push_back(subToActive[0]);
        return seeds;
    }
    vector<Index> div = diversitySelection(subset, 100, mt, nAtoms, true, MD::StartSeed::Medoid);
    for (Index li : div) {
        if (li >= 0 && li < nMax) seeds.push_back(subToActive[(int)li]);
    }
    if (seeds.empty()) {
        for (int i = 0; i < nMax; ++i) seeds.push_back(subToActive[i]);
    }
    return seeds;
}

void Equal::growClusters() {
    int cols = (int)data.cols();
    std::set<Index> active;
    for (int i = 0; i < nTotal; ++i) active.insert((Index)i);
    clusters.clear();
    // each frame's |x|^2, for the radial step's algebra
    ArrayXd xsq(nTotal);
    for (int i = 0; i < nTotal; ++i) xsq[i] = data.row(i).square().sum();

    while ((int)active.size() > 2 && (int)active.size() > nSeedsInt) {
        MD::progress(1.0 - (double)active.size() / nTotal, "eQUAL");
        vector<Index> activeVec(active.begin(), active.end());

        ArrayXXd work((Index)activeVec.size(), cols);
        for (int i = 0; i < (int)activeVec.size(); ++i) work.row(i) = data.row(activeVec[i]);

        vector<Index> seeds = chooseSeeds(work, activeVec);
        if (seeds.empty()) break;

        // Propose one cluster per seed: every active point within the radial
        // threshold (extendedComparison(seed, candidate) <= threshold).
        //
        // With the MSD the pairs are sorted out by algebra first: the sum
        // pairDistance computes is |x - y|^2 = |x|^2 + |y|^2 - 2 x.y, the dot
        // products of a block of seeds with every active frame one matrix
        // product. A pair whose value is clear of the threshold by more than
        // the rounding error of both computations is decided so; the others
        // (near the threshold, or not finite) are computed by pairDistance as
        // before. The clusters are the same, in the same order.
        vector<vector<Index>> cand(seeds.size());
        size_t maxLen = 0;
        const Index na = (Index)activeVec.size();
        if (mt == MD::Metric::MSD && na > 16) {
            const double thr = threshold * 2.0 * nAtoms;     // pairDistance = sum / (2 nAtoms)
            const double gam = 8.0 * (double)(cols + 16) * std::numeric_limits<double>::epsilon();
            const double lo = thr * (1 - 8 * std::numeric_limits<double>::epsilon());
            const double hi = thr * (1 + 8 * std::numeric_limits<double>::epsilon());
            const Index B = 64;
            for (size_t b0 = 0; b0 < seeds.size(); b0 += B) {
                Index nb = (Index)std::min<size_t>(B, seeds.size() - b0);
                ArrayXXd sb(nb, cols);
                for (Index q = 0; q < nb; ++q) sb.row(q) = data.row(seeds[b0 + q]);
                Eigen::MatrixXd dots = work.matrix() * sb.matrix().transpose();   // na x nb
                for (Index q = 0; q < nb; ++q) {
                    Index s = seeds[b0 + q];
                    vector<Index>& cl = cand[b0 + q];
                    for (Index r = 0; r < na; ++r) {
                        Index c = activeVec[r];
                        double m = xsq[s] + xsq[c];
                        double a = m - 2 * dots(r, q);
                        double e = gam * m;
                        bool in;
                        if (std::isfinite(a) && std::isfinite(e) && a + e < lo) in = true;
                        else if (std::isfinite(a) && std::isfinite(e) && a - e > hi) in = false;
                        else in = pairDistance(s, c) <= threshold;
                        if (in) cl.push_back(c);
                    }
                    if (cl.size() > maxLen) maxLen = cl.size();
                }
            }
        } else {
            for (size_t q = 0; q < seeds.size(); ++q) {
                Index s = seeds[q];
                vector<Index>& cl = cand[q];
                for (Index c : activeVec) {
                    double dval = pairDistance(s, c);
                    if (dval <= threshold) cl.push_back(c);
                }
                if (cl.size() > maxLen) maxLen = cl.size();
            }
        }
        if (maxLen == 0) break;

        // Winner = densest proposal; ties broken by lowest intra-cluster sim.
        vector<int> tied;
        for (int i = 0; i < (int)cand.size(); ++i) {
            if (cand[i].size() == maxLen) tied.push_back(i);
        }
        int winIdx;
        if (tied.size() == 1) {
            winIdx = tied[0];
        } else {
            double bestSim = 9999.0;
            int bestT = -1;
            for (int t = 0; t < (int)tied.size(); ++t) {
                double s = intraSim(cand[tied[t]]);
                if (s < bestSim) { bestSim = s; bestT = t; }
            }
            winIdx = tied[bestT >= 0 ? bestT : 0];
        }
        vector<Index> winner = cand[winIdx];

        // Remove winner members from the active set (by original index).
        for (Index idx : winner) active.erase(idx);

        // check_sim gate: reject + TERMINATE on first too-loose winner.
        if (checkSim) {
            if (intraSim(winner) > simThreshold) break;
        }
        // low-density gate: drop a too-small winner (members become noise).
        if (rejectLowd && (int)winner.size() < minSamplesInt) {
            // dropped
        } else {
            clusters.push_back(std::move(winner));
        }
        // (alignment step would go here; alignTraj only implements None.)
    }
}

void Equal::createLabels() {
    labels.assign(nTotal, -1);
    std::stable_sort(clusters.begin(), clusters.end(),
        [](const vector<Index>& a, const vector<Index>& b) { return a.size() > b.size(); });
    clusterSizes.clear();
    for (int c = 0; c < (int)clusters.size(); ++c) {
        clusterSizes.push_back((int)clusters[c].size());
        for (Index idx : clusters[c]) labels[idx] = c;
    }
}

vector<vector<Index>> Equal::getClusters() { return clusters; }
vector<int> Equal::getLabels() { return labels; }
vector<int> Equal::getClusterSizes() { return clusterSizes; }

pair<double, double> Equal::computeScores(Veci labelsIn, Mat dataIn) {
    return {calinskiHarabaszScore(dataIn, labelsIn), daviesBouldinScore(dataIn, labelsIn)};
}
