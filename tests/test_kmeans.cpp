#include <filesystem>

#include <gtest/gtest.h>

#include "test_utils.h"
#include "../src/cluster/KMeansRex/KMeans.cpp"

TEST(KMeansNani, simcsv){
    // Construct path to data file
    // Note: assumes test executable is run from build/tests/ directory, which is the default behavior of CMake. The data is in tests/data/
    std::filesystem::path buildTestPath = std::filesystem::current_path();
    std::filesystem::path projectRoot = buildTestPath.parent_path().parent_path();
    std::filesystem::path dataPath = projectRoot / "tests" / "data";
    std::filesystem::path inputDataFile = dataPath / "sim.csv";
    ArrayXXd data = readCSVtoEigen(inputDataFile.string());
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
    std::filesystem::path pyLabelsFile = dataPath / "sim_py_labels_kmeans.csv";
    ArrayXXd pyLabels = readCSVtoEigen(pyLabelsFile.string());
    ArrayXi pyLabelsInt = pyLabels.col(0).cast<int>();

    ASSERT_EQ(labels.size(), pyLabelsInt.size());
    for (int i = 0; i < labels.size(); ++i){
        EXPECT_EQ(labels[i], pyLabelsInt[i]) << "Labels differ at index " << i;
    }   
}