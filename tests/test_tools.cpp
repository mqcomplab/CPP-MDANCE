#include <gtest/gtest.h>

#include <cmath>

#include "../src/tools/cluster.h"
#include "../src/tools/scores.h"

namespace {

ArrayXXd spreadOutPoints() {
    ArrayXXd data(9, 4);
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 4; ++j) data(i, j) = i * 3.0 + j;
    }
    return data;
}

}  // namespace

// A single cluster leaves no between-cluster dispersion, so the
// Calinski-Harabasz ratio is 0/0. The resulting nan propagates into anything
// that serializes the score -- and is not representable in JSON at all.
TEST(Scores, FiniteForASingleCluster) {
    ArrayXXd data = spreadOutPoints();
    VectorXi labels = VectorXi::Zero(9);

    double ch = calinskiHarabaszScore(data, labels);
    double db = daviesBouldinScore(data, labels);
    EXPECT_TRUE(std::isfinite(ch)) << "calinskiHarabasz = " << ch;
    EXPECT_TRUE(std::isfinite(db)) << "daviesBouldin = " << db;
    EXPECT_EQ(ch, 0.0);
    EXPECT_EQ(db, 0.0);
}

TEST(Scores, StillDiscriminatesBetweenPartitions) {
    ArrayXXd data = spreadOutPoints();
    VectorXi tight(9), scattered(9);
    tight    << 0, 0, 0, 1, 1, 1, 2, 2, 2;   // matches the ordering of the data
    scattered << 0, 1, 2, 0, 1, 2, 0, 1, 2;  // interleaved

    EXPECT_GT(calinskiHarabaszScore(data, tight),
              calinskiHarabaszScore(data, scattered));
}

// The values, by hand (scikit-learn gives the same): the rows lie on a line,
// 6 apart. Tight: centroids 18 apart, within-cluster sum 216, between 1944:
// CH = (1944/2)/(216/6) = 27; mean distance to the centroid 4: DB = 8/18.
// Interleaved: centroids 6 apart, within 1944, between 216: CH = 1/3;
// mean distance 12: DB = 24/6 = 4.
TEST(Scores, MatchHandComputedValues) {
    ArrayXXd data = spreadOutPoints();
    VectorXi tight(9), scattered(9);
    tight    << 0, 0, 0, 1, 1, 1, 2, 2, 2;
    scattered << 0, 1, 2, 0, 1, 2, 0, 1, 2;
    EXPECT_NEAR(calinskiHarabaszScore(data, tight), 27.0, 1e-9);
    EXPECT_NEAR(calinskiHarabaszScore(data, scattered), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(daviesBouldinScore(data, tight), 4.0 / 9.0, 1e-12);
    EXPECT_NEAR(daviesBouldinScore(data, scattered), 4.0, 1e-9);
}

// The default constructor's body used to be statements with no effect, leaving
// n holding whatever was on the stack.
TEST(Cluster, DefaultConstructorInitializesCount) {
    ::Cluster c;
    EXPECT_EQ(c.getN(), 0);
}
