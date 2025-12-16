#include  <filesystem>

#include "test_divine.h"
#include "test_utils.h"

#include "../../src/cluster/divine.cpp"

DivineTest::DivineTest(){
    //read in data (make_blobs.csv)
    std::filesystem::path path = std::filesystem::current_path();
    path = path.parent_path().parent_path();
    path = path/"tests"/"data/make_blobs.csv";
    data = readCSVtoEigen(path.string());
}

TEST_F(DivineTest, TestThreshold){
    Divine model=Divine(data, MD::DivineSplit::WeightedMSD, MD::DivineAnchors::NANI, 
                        MD::KinitType::CompSim, 0, 3, true, 1, 0.2);
    labels = model.getLabels();
    clusters = model.getClusters();
}