#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "../src/cluster/prime.h"

namespace {

constexpr int NATOMS = 4;
constexpr int NCOLS = NATOMS * 3;

// Three groups of unequal size, so the most-populated cluster (PRIME's c0) is
// unambiguous: 12 / 6 / 3 frames.
ArrayXXd threeGroups() {
    const int sizes[3] = {12, 6, 3};
    ArrayXXd data(21, NCOLS);
    int row = 0;
    for (int g = 0; g < 3; ++g) {
        for (int i = 0; i < sizes[g]; ++i, ++row) {
            for (int j = 0; j < NCOLS; ++j) {
                data(row, j) = 10.0 + g * 30.0 + 0.1 * ((i * 5 + j) % 7);
            }
        }
    }
    return data;
}

std::vector<int> threeGroupLabels() {
    std::vector<int> labels;
    for (int i = 0; i < 12; ++i) labels.push_back(0);
    for (int i = 0; i < 6; ++i)  labels.push_back(1);
    for (int i = 0; i < 3; ++i)  labels.push_back(2);
    return labels;
}

}  // namespace

TEST(Prime, PredictionsAreValidFrameIndices) {
    ArrayXXd data = threeGroups();
    Prime prime(data, threeGroupLabels(), MD::Metric::RR);
    PrimeResult r = prime.getResult();

    const int n = (int)data.rows();
    for (int idx : {r.pairwise, r.uni, r.medoid, r.outlier,
                    r.medoid_all, r.medoid_c0, r.medoid_c0_trimmed}) {
        EXPECT_TRUE(idx == -1 || (idx >= 0 && idx < n))
            << "index " << idx << " is not a frame of the input";
    }
}

TEST(Prime, CountsClustersFromTheLabels) {
    ArrayXXd data = threeGroups();
    Prime prime(data, threeGroupLabels(), MD::Metric::RR);
    EXPECT_EQ(prime.getResult().nclusters, 3);
}

// c0 is the most populated cluster, so its medoid has to come from it.
TEST(Prime, MedoidC0ComesFromTheLargestCluster) {
    ArrayXXd data = threeGroups();
    std::vector<int> labels = threeGroupLabels();
    Prime prime(data, labels, MD::Metric::RR);
    PrimeResult r = prime.getResult();

    ASSERT_NE(r.medoid_c0, -1);
    EXPECT_EQ(labels[r.medoid_c0], 0) << "medoidC0 came from a smaller cluster";
}

TEST(Prime, TrimmingNothingMatchesTheUntrimmedMedoid) {
    ArrayXXd data = threeGroups();
    Prime prime(data, threeGroupLabels(), MD::Metric::RR, 0.0);
    PrimeResult r = prime.getResult();
    EXPECT_EQ(r.medoid_c0_trimmed, r.medoid_c0);
}

TEST(Prime, SMMetricRunsAndStaysInRange) {
    ArrayXXd data = threeGroups();
    Prime prime(data, threeGroupLabels(), MD::Metric::SM, 0.1, true);
    PrimeResult r = prime.getResult();

    const int n = (int)data.rows();
    EXPECT_EQ(r.nclusters, 3);
    for (int idx : {r.pairwise, r.uni, r.medoid, r.outlier, r.medoid_all}) {
        EXPECT_TRUE(idx == -1 || (idx >= 0 && idx < n));
    }
}

// PRIME applies global min-max normalization internally, and that is
// idempotent -- pre-normalized input must give the same answer.
TEST(Prime, IsInvariantUnderGlobalRescaling) {
    ArrayXXd data = threeGroups();
    std::vector<int> labels = threeGroupLabels();

    PrimeResult plain = Prime(data, labels, MD::Metric::RR).getResult();
    ArrayXXd scaled = (data - data.minCoeff()) / (data.maxCoeff() - data.minCoeff());
    PrimeResult rescaled = Prime(scaled, labels, MD::Metric::RR).getResult();

    EXPECT_EQ(plain.medoid_all, rescaled.medoid_all);
    EXPECT_EQ(plain.medoid_c0, rescaled.medoid_c0);
    EXPECT_EQ(plain.pairwise, rescaled.pairwise);
    EXPECT_EQ(plain.uni, rescaled.uni);
}
