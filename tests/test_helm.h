#include <gtest/gtest.h>

#include "../src/tools/types.h"
#include "../../../src/tools/cluster.h"
#include "../../tools/hc_utils.h"

// test fixture for helm
class TestHelm : public ::testing::Test {
protected:
    TestHelm();
    void inputCluster();
    ArrayXXd makeDataByRow(ArrayXd a, ArrayXd b);
    ArrayXXd data;
    ArrayXi labels; // labels of frames
    set<int> uniqueLabels;
    std::vector<HCTree> clusterTree;
};