#pragma once

#include "../tools/bts.h"
#include "../tools/types.h"

/*
 * eQUAL (Extended QUALity) clustering -- a radial / quality-threshold clustering
 * that grows clusters greedily from seeds using MSD / extended-comparison, and
 * derives the cluster count automatically from a radial threshold (no preset k).
 *
 * Port of mqcomplab/MDANCE src/mdance/cluster/equal.py (class ExtendedQuality).
 * Faithful to the grow_clusters loop. Intentional, documented divergences:
 *   - Only the deterministic, pure-coordinate seed methods are supported:
 *     comp_sim and medoid. sklearn-based greedy/vanilla/mini_batch_kmeans are
 *     rejected at the parse layer.
 *   - Member removal is by ORIGINAL frame index (not value equality), which is
 *     behaviour-equivalent for unique rows and strictly safer for duplicates.
 *   - alignment (uni/kron) is a no-op (alignTraj only implements None).
 *   - reject_lowd defaults OFF (matches the plugin's flag convention); upstream
 *     constructor defaults it ON.
 *
 * Frames left unclustered (trailing <= 2 / <= n_seeds points, members of a
 * rejected low-density cluster, or everything after a check_sim termination)
 * receive label -1.
 */
class Equal {
    Mat data;
    vector<int> labels;

    MD::Metric mt;
    MD::EqualSeed seedMethod;
    double threshold;
    int nAtoms;
    bool checkSim;
    double simThreshold;
    bool rejectLowd;
    int nSeedsInt;
    int minSamplesInt;
    int percentage;
    MD::AlignMethod alignMethod;
    int nTotal;

    vector<vector<Index>> clusters;   // each: original frame indices
    vector<int> clusterSizes;

    void growClusters();
    vector<Index> chooseSeeds(const ArrayXXd& work, const vector<Index>& activeVec);
    double intraSim(const vector<Index>& members);
    void createLabels();

public:
    Equal(Mat data, MD::Metric mt, double threshold, int nAtoms,
          MD::EqualSeed seedMethod = MD::EqualSeed::Medoid,
          double nSeeds = 1, bool checkSim = false, double simThreshold = 0,
          bool rejectLowd = false, double minSamples = 10, int percentage = 10,
          MD::AlignMethod alignMethod = MD::AlignMethod::None);

    vector<vector<Index>> getClusters();
    vector<int> getLabels();
    vector<int> getClusterSizes();
    pair<double, double> computeScores(Veci labels, Mat data);
};
