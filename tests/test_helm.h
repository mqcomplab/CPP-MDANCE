#include <gtest/gtest.h>

#include "../src/tools/types.h"
#include "../../../src/tools/cluster.h"

// test fixture for helm
class TestHelm : public ::testing::Test {
protected:
    TestHelm();
    void inputCluster();
    ArrayXXd data;
    ArrayXi labels; // labels of frames
    std::map<int, std::vector<Cluster>> clusters_map;
};