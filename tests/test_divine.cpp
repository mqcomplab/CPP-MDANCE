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
    //std::cout<<data.row(1)<<std::endl;
    Divine model=Divine(data, MD::DivineSplit::WeightedMSD, MD::DivineAnchors::NANI, 
                        MD::KinitType::CompSim, 0, 3, true, 1, 0.2);
    labels = model.getLabels();
    clusters = model.getClusters();

    for(auto c:clusters){
        ASSERT_GE(c.size(), 30);
    }
}

TEST_F(DivineTest, TestCombinations){
    vector<MD::DivineSplit> split={
        MD::DivineSplit::MSD,
        MD::DivineSplit::Radius,
        MD::DivineSplit::WeightedMSD
    };
    vector<MD::DivineAnchors> anchor={
        MD::DivineAnchors::NANI,
        MD::DivineAnchors::OutlierPair,
        MD::DivineAnchors::SplinterPair
    };
    vector<bool> refine={
        true,
        false
    };

    int nAtoms = 1;
    int end = 0;        //end = 0 means 'k'
    for(auto s:split){
        for(auto a:anchor){
            for(auto r:refine){
                Divine model=Divine(
                    data,
                    s,
                    a,
                    MD::KinitType::CompSim,
                    end,
                    3,
                    r,
                    nAtoms
                );
                labels = model.getLabels();
                clusters = model.getClusters();
                scores = model.getScores();

                set<int> uniqueLabels;
                for(auto i:labels){
                    uniqueLabels.insert(i);
                }
                
                //assert #1
                //asserst #2
                EXPECT_EQ(clusters.size(), 3);
                //assert isinstance
                EXPECT_EQ(labels.size(), data.rows());
                EXPECT_EQ(uniqueLabels.size(), 3);
            }
        }
    }
}