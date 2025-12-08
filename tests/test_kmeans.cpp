#include "test_utils.h"
#include "../src/cluster/KMeansRex/KMeans.cpp"

TEST(KMeansNani, simcsv){
    ArrayXXd data = readCSVtoEigen("sim.csv");
    // Note: the settings for kmeans need to match those used in the python version
    int nAtoms = 50;
    int nClusters = 10;
    KmeansNANI test(data, nClusters, MD::Metric::MSD, MD::KinitType::CompSim, nAtoms, 10);

    ArrayXi labels = test.getLabels();
    for (int i = 0; i < labels.size(); ++i){
        if (labels[i] < 0 || labels[i] >= nClusters){
            std::cerr << "Error: Label out of bounds: " << labels[i] << std::endl;
        }
    }
    
    ArrayXXd pyLabels = readCSVtoEigen("sim_py_labels.csv");
    ArrayXi pyLabelsInt = pyLabels.col(0).cast<int>();

    ASSERT_EQ(labels.size(), pyLabelsInt.size());
    for (int i = 0; i < labels.size(); ++i){
        EXPECT_EQ(labels[i], pyLabelsInt[i]) << "Labels differ at index " << i;
    }   
}