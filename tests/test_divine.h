#include <gtest/gtest.h>

#include "../src/tools/types.h"

class DivineTest : public ::testing::Test{
protected:
    Mat data;
    vector<int> labels;
    vector<vector<Index>> clusters;
    vector<pair<double, double>> scores;

    DivineTest();
};