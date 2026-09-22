#include "equal.h"

#include <algorithm>
#include <set>

#include "../tools/scores.h"

Equal::Equal(Mat data, MD::Metric mt, double threshold, int nAtoms,
             MD::EqualSeed seedMethod, double nSeeds, bool checkSim,
             double simThreshold, bool rejectLowd, double minSamples,
             int percentage, MD::AlignMethod alignMethod)
    : data(data), mt(mt), seedMethod(seedMethod), threshold(threshold),
      nAtoms(nAtoms), checkSim(checkSim), simThreshold(simThreshold),
      rejectLowd(rejectLowd), percentage(percentage), alignMethod(alignMethod)
{
    nTotal = (int)data.rows();

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
    ArrayXd cs = calculateCompSim(work, nAtoms, mt);

    vector<int> order(na);
    for (int i = 0; i < na; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return cs[a] > cs[b]; });

    if (seedMethod == MD::EqualSeed::Medoid) {
        int take = std::min(nSeedsInt, na);
        for (int i = 0; i < take; ++i) seeds.push_back(activeVec[order[i]]);
        return seeds;
    }

    // comp_sim: take the high-density region (size n_max, based on the FIXED
    // original n_objects per upstream), then diversity-select the whole subset;
    // every selected point becomes a seed.
    int nMax = (int)(percentage * (double)nTotal / 100.0);
    if (nMax > na) nMax = na;
    if (nMax < 1) nMax = 1;

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

    while ((int)active.size() > 2 && (int)active.size() > nSeedsInt) {
        vector<Index> activeVec(active.begin(), active.end());

        ArrayXXd work((Index)activeVec.size(), cols);
        for (int i = 0; i < (int)activeVec.size(); ++i) work.row(i) = data.row(activeVec[i]);

        vector<Index> seeds = chooseSeeds(work, activeVec);
        if (seeds.empty()) break;

        // Propose one cluster per seed: every active point within the radial
        // threshold (extendedComparison(seed, candidate) <= threshold).
        vector<vector<Index>> cand;
        cand.reserve(seeds.size());
        size_t maxLen = 0;
        ArrayXXd pair(2, cols);
        for (Index s : seeds) {
            pair.row(0) = data.row(s);
            vector<Index> cl;
            for (Index c : activeVec) {
                pair.row(1) = data.row(c);
                double dval = extendedComparison(pair, 2, nAtoms, false, mt);
                if (dval <= threshold) cl.push_back(c);
            }
            if (cl.size() > maxLen) maxLen = cl.size();
            cand.push_back(std::move(cl));
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
