#include  <filesystem>

#include "test_divine.h"
#include "test_utils.h"

#include "../../src/cluster/divine.cpp"

DivineTest::DivineTest(){
    //read in data (make_blobs.csv)
    std::filesystem::path path = std::filesystem::current_path();
    path = path.parent_path().parent_path();
    path = path/"tests"/"data/make_blobs_divine.csv";
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


TEST_F(DivineTest, TestPoint){
    Mat smallData = data(Eigen::seq(0,4), Eigen::placeholders::all);
    int end=1;
    int k=0;
    bool refine=false;
    Divine model = Divine(
        smallData,
        MD::DivineSplit::MSD,
        MD::DivineAnchors::SplinterPair,
        MD::KinitType::StratAll,
        end,
        k,
        false
    );
    labels = model.getLabels();
    clusters = model.getClusters();

    EXPECT_EQ(clusters.size(), smallData.rows());
    set<int> uniqueLabels;
    for(int i:labels){
        uniqueLabels.insert(i);
    }
    EXPECT_EQ(labels.size(), uniqueLabels.size());
}


TEST_F(DivineTest, TestEmpty){
    Mat data = Mat::Zero(0, 2);
    Divine model=Divine(
        data,
        MD::DivineSplit::MSD,
        MD::DivineAnchors::NANI,
        MD::KinitType::StratAll,
        0,
        3
    );

    clusters = model.getClusters();
    labels = model.getLabels();

    EXPECT_EQ(clusters.size(), 0);
    EXPECT_EQ(labels.size(), 0);
}