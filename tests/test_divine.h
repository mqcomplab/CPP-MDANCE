#include <gtest/gtest.h>

#include "../src/tools/types.h"

class DivineTest : public ::testing::Test{
protected:
    Mat data;
    Veci labels;
    vector<vector<Index>> clusters;

    DivineTest();
};