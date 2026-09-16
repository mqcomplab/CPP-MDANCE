#include <gtest/gtest.h>

#include "../src/tools/result_utils.h"

namespace {

// Three frames per cluster, three well-separated clusters.
ArrayXXd makeData() {
    ArrayXXd data(9, 2);
    data << 0, 0,  0, 1,  1, 0,
            50, 50, 50, 51, 51, 50,
            100, 100, 100, 101, 101, 100;
    return data;
}

Eigen::ArrayXi makeLabels(int a, int b, int c) {
    Eigen::ArrayXi labels(9);
    labels << a, a, a, b, b, b, c, c, c;
    return labels;
}

}  // namespace

// A HELM run that merges nothing must hand every frame back to the cluster it
// started in, whatever the initial label VALUES are. buildClusterTree used to
// tag each tree with its dense position while the caller matched on the label
// value, so any partition that was not exactly {0..K-1} silently came back as
// all -1 -- reported as a successful run.
TEST(ResultUtils, MapsBackNonDenseInitialLabels) {
    ArrayXXd data = makeData();

    std::vector<int> dense =
        labelsFromHelmClusters(buildClusterTree(data, makeLabels(0, 1, 2)),
                               makeLabels(0, 1, 2), 9);
    EXPECT_EQ(dense, (std::vector<int>{0, 0, 0, 1, 1, 1, 2, 2, 2}));

    // Same partition, labels shifted out of 0..K-1.
    std::vector<int> shifted =
        labelsFromHelmClusters(buildClusterTree(data, makeLabels(5, 15, 25)),
                               makeLabels(5, 15, 25), 9);
    EXPECT_EQ(shifted, dense);

    // Same partition with a -1 "noise" group, as produced by eQUAL. The sorted
    // label set starts at -1, so every dense index was off by one.
    std::vector<int> withNoise =
        labelsFromHelmClusters(buildClusterTree(data, makeLabels(-1, 0, 1)),
                               makeLabels(-1, 0, 1), 9);
    EXPECT_EQ(withNoise, dense);
}

// More labels than frames used to write past the end of the label vector, which
// aborts the process (and would take VMD down through the C API).
TEST(ResultUtils, RejectsLabelCountMismatch) {
    ArrayXXd data = makeData();
    Eigen::ArrayXi labels = makeLabels(0, 1, 2);
    std::vector<HCTree> trees = buildClusterTree(data, labels);

    // buildClusterTree has to reject it too: it indexes data.row(j) for every
    // label, so a long label array reads past the end of the coordinates.
    Eigen::ArrayXi tooMany(20);
    tooMany.setZero();
    EXPECT_THROW(buildClusterTree(data, tooMany), std::runtime_error);

    EXPECT_THROW(labelsFromHelmClusters(trees, labels, 4), std::runtime_error);
    EXPECT_THROW(labelsFromHelmClusters(trees, labels, 20), std::runtime_error);
    EXPECT_NO_THROW(labelsFromHelmClusters(trees, labels, 9));
}
