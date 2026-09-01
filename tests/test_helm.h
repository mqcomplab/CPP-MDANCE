#include <gtest/gtest.h>

#include "../src/tools/types.h"
#include "../src/tools/hc_utils.h"

// test fixture for helm
class TestHelm : public ::testing::Test {
protected:
    TestHelm();
    void inputCluster();
    ArrayXXd makeDataByRow(ArrayXd a, ArrayXd b);
    Mat getSelectedData(vector<HCTree> clusters);
    ArrayXXd data;
    ArrayXi labels; // labels of frames
    set<int> uniqueLabels;
    std::vector<HCTree> clusterTree;
};