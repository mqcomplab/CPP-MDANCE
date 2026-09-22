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
}

TEST(Scores, StillDiscriminatesBetweenPartitions) {
    ArrayXXd data = spreadOutPoints();
    VectorXi tight(9), scattered(9);
    tight    << 0, 0, 0, 1, 1, 1, 2, 2, 2;   // matches the ordering of the data
    scattered << 0, 1, 2, 0, 1, 2, 0, 1, 2;  // interleaved

    EXPECT_GT(calinskiHarabaszScore(data, tight),
              calinskiHarabaszScore(data, scattered));
}

// The default constructor's body used to be statements with no effect, leaving
// n holding whatever was on the stack.
TEST(Cluster, DefaultConstructorInitializesCount) {
    ::Cluster c;
    EXPECT_EQ(c.getN(), 0);
}
