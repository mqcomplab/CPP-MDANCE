#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "../src/cluster/prime.h"

namespace {

constexpr int NATOMS = 4;
constexpr int NCOLS = NATOMS * 3;

// Three groups of unequal size, so the most-populated cluster (PRIME's c0) is
// unambiguous: 12 / 6 / 3 frames. No two frames are the same (the pattern this
// replaces repeated every 7th: answers chosen among ties prove little).
ArrayXXd threeGroups() {
    const int sizes[3] = {12, 6, 3};
    ArrayXXd data(21, NCOLS);
    int row = 0;
    for (int g = 0; g < 3; ++g) {
        for (int i = 0; i < sizes[g]; ++i, ++row) {
            for (int j = 0; j < NCOLS; ++j) {
                data(row, j) = 10.0 + g * 30.0 + 0.1 * ((i * 5 + j * 3) % 13) + 0.013 * i * (j + 1);
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
        // (-1 is "none": with three clusters every prediction exists)
        EXPECT_TRUE(idx >= 0 && idx < n)
            << "index " << idx << " is not a frame of the input";
    }
}

// The four predictions score c0's frames against the other clusters: each
// is a frame of c0, the trimmed medoid too.
TEST(Prime, PredictionsAreFramesOfTheLargestCluster) {
    ArrayXXd data = threeGroups();
    std::vector<int> labels = threeGroupLabels();
    for (double trim : {0.0, 0.25}) {
        for (MD::Metric mt : {MD::Metric::RR, MD::Metric::SM}) {
            PrimeResult r = Prime(data, labels, mt, trim, true).getResult();
            for (int idx : {r.pairwise, r.uni, r.medoid, r.outlier, r.medoid_c0_trimmed}) {
                ASSERT_GE(idx, 0);
                EXPECT_EQ(labels[idx], 0) << "frame " << idx << " is not in c0";
            }
        }
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
        EXPECT_TRUE(idx >= 0 && idx < n);
    }
}

// PRIME applies global min-max normalization internally: coordinates scaled
// by a power of two (exactly, in floating point) normalize to the same values
// and give the same answers. (The test this replaces compared normalized data
// with itself normalized again: the same numbers on both sides.)
TEST(Prime, IsInvariantUnderGlobalRescaling) {
    ArrayXXd data = threeGroups();
    std::vector<int> labels = threeGroupLabels();

    PrimeResult plain = Prime(data, labels, MD::Metric::RR).getResult();
    ArrayXXd scaled = data * 4.0;
    PrimeResult rescaled = Prime(scaled, labels, MD::Metric::RR).getResult();

    EXPECT_EQ(plain.medoid_all, rescaled.medoid_all);
    EXPECT_EQ(plain.medoid_c0, rescaled.medoid_c0);
    EXPECT_EQ(plain.pairwise, rescaled.pairwise);
    EXPECT_EQ(plain.uni, rescaled.uni);
    EXPECT_EQ(plain.medoid, rescaled.medoid);
    EXPECT_EQ(plain.outlier, rescaled.outlier);
}

TEST(Prime, RefusesLabelsThatAreNoCluster) {
    ArrayXXd data = threeGroups();
    std::vector<int> labels = threeGroupLabels();
    labels[3] = 2147483647;    // overflowed l + 1 into a write out of bounds
    EXPECT_THROW(Prime(data, labels, MD::Metric::RR), std::invalid_argument);
    labels[3] = -2;
    EXPECT_THROW(Prime(data, labels, MD::Metric::RR), std::invalid_argument);
}
