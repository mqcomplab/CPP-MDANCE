#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

#include "../src/cluster/equal.h"
#include "../src/tools/bts.h"

namespace {

constexpr int NATOMS = 4;
constexpr int NCOLS = NATOMS * 3;

// Two compact, well-separated groups: rows [0, per) and [per, 2*per).
ArrayXXd twoBlobs(int per = 12, double sep = 40.0) {
    ArrayXXd data(2 * per, NCOLS);
    for (int b = 0; b < 2; ++b) {
        for (int i = 0; i < per; ++i) {
            for (int j = 0; j < NCOLS; ++j) {
                // Deterministic jitter well below `sep`.
                data(b * per + i, j) = b * sep + 0.05 * ((i * 7 + j * 3) % 5);
            }
        }
    }
    return data;
}

// eQUAL grows a cluster by taking every point whose PAIRWISE extended
// comparison with the seed is <= threshold, so a usable threshold sits between
// the widest within-blob distance and the narrowest between-blob one.
double pairDistance(const ArrayXXd& data, int a, int b) {
    ArrayXXd pair(2, data.cols());
    pair.row(0) = data.row(a);
    pair.row(1) = data.row(b);
    return extendedComparison(pair, 2, NATOMS, false, MD::Metric::MSD);
}

}  // namespace

TEST(Equal, SeparatesWellSeparatedBlobs) {
    const int per = 12;
    ArrayXXd data = twoBlobs(per);

    double withinMax = 0.0, betweenMin = 1e30;
    for (int i = 0; i < 2 * per; ++i) {
        for (int j = i + 1; j < 2 * per; ++j) {
            double d = pairDistance(data, i, j);
            if ((i < per) == (j < per)) withinMax = std::max(withinMax, d);
            else                        betweenMin = std::min(betweenMin, d);
        }
    }
    ASSERT_LT(withinMax, betweenMin) << "test data is not separable";
    const double threshold = 0.5 * (withinMax + betweenMin);

    Equal eq(data, MD::Metric::MSD, threshold, NATOMS);
    std::vector<int> labels = eq.getLabels();

    ASSERT_EQ((int)labels.size(), 2 * per);
    std::set<int> blobA(labels.begin(), labels.begin() + per);
    std::set<int> blobB(labels.begin() + per, labels.end());

    EXPECT_EQ(blobA.size(), 1u) << "first blob was split";
    EXPECT_EQ(blobB.size(), 1u) << "second blob was split";
    EXPECT_EQ(blobA.count(-1), 0u) << "first blob left unclustered";
    EXPECT_EQ(blobB.count(-1), 0u) << "second blob left unclustered";
    EXPECT_NE(*blobA.begin(), *blobB.begin()) << "the two blobs were merged";
    EXPECT_EQ((int)eq.getClusters().size(), 2);
}

// Every cluster is grown from one seed, so each must contain a member that all
// of its other members are within `threshold` of.
TEST(Equal, EveryClusterHasAMemberWithinThresholdOfAllOthers) {
    ArrayXXd data = twoBlobs();
    const double threshold = 4.0;

    Equal eq(data, MD::Metric::MSD, threshold, NATOMS);

    for (const auto& cluster : eq.getClusters()) {
        bool foundSeed = false;
        for (Index seed : cluster) {
            bool allWithin = true;
            for (Index member : cluster) {
                if (pairDistance(data, (int)seed, (int)member) > threshold) {
                    allWithin = false;
                    break;
                }
            }
            if (allWithin) { foundSeed = true; break; }
        }
        EXPECT_TRUE(foundSeed) << "cluster of " << cluster.size()
                               << " has no member covering the rest";
    }
}

// The cluster count is a consequence of the threshold: a wider radius can only
// absorb more points per cluster, never fewer.
TEST(Equal, RaisingTheThresholdDoesNotIncreaseClusterCount) {
    ArrayXXd data = twoBlobs();

    int previous = -1;
    for (double t : {1.0, 5.0, 20.0, 100.0, 1000.0}) {
        Equal eq(data, MD::Metric::MSD, t, NATOMS);
        int count = (int)eq.getClusters().size();
        if (previous >= 0) {
            EXPECT_LE(count, previous) << "threshold " << t << " produced more clusters";
        }
        previous = count;
    }
}

TEST(Equal, LabelsAndSizesAreConsistent) {
    ArrayXXd data = twoBlobs();
    Equal eq(data, MD::Metric::MSD, 4.0, NATOMS);

    std::vector<int> labels = eq.getLabels();
    std::vector<int> sizes = eq.getClusterSizes();

    ASSERT_EQ((int)labels.size(), (int)data.rows());
    ASSERT_EQ(sizes.size(), eq.getClusters().size());

    // Sizes are reported largest-first, and every label indexes a real cluster.
    for (size_t c = 1; c < sizes.size(); ++c) {
        EXPECT_GE(sizes[c - 1], sizes[c]) << "cluster sizes are not descending";
    }
    std::vector<int> counted(sizes.size(), 0);
    for (int l : labels) {
        EXPECT_GE(l, -1);
        EXPECT_LT(l, (int)sizes.size());
        if (l >= 0) counted[l]++;
    }
    EXPECT_EQ(counted, sizes);
}
