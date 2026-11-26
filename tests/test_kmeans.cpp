#include <filesystem>

#include <gtest/gtest.h>

#include "test_utils.h"
#include "../src/cluster/KMeansRex/KMeans.h"

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
        // (this only printed before: a label out of bounds failed nothing)
        EXPECT_TRUE(labels[i] >= 0 && labels[i] < nClusters) << "label out of bounds: " << labels[i] << " at " << i;
    }
    std::filesystem::path pyLabelsFile = dataPath / "sim_py_labels_kmeans.csv";
    ArrayXXd pyLabels = readCSVtoEigen(pyLabelsFile.string());
    ArrayXi pyLabelsInt = pyLabels.col(0).cast<int>();

    ASSERT_EQ(labels.size(), pyLabelsInt.size());
    for (int i = 0; i < labels.size(); ++i){
        EXPECT_EQ(labels[i], pyLabelsInt[i]) << "Labels differ at index " << i;
    }   
}

// tol: scikit-learn's stop, as MDANCE (Python) runs KMeans. On kmeans_tol.csv
// (tests/make_kmeans_tol_data.py: 1600 points of overlapping blobs) MDANCE's
// KmeansNANI, CompSim and k 6, stops after 17 iterations at its tol 1e-4,
// where the labels settle after 22: five frames end in other clusters.
static ArrayXXd tolData(ArrayXi* pyLabels = nullptr) {
    std::filesystem::path dataPath = std::filesystem::current_path().parent_path().parent_path() / "tests" / "data";
    if (pyLabels) {
        ArrayXXd l = readCSVtoEigen((dataPath / "kmeans_tol_py_labels.csv").string());
        *pyLabels = l.col(0).cast<int>();
    }
    return readCSVtoEigen((dataPath / "kmeans_tol.csv").string());
}

TEST(KMeansNaniTol, MatchesMdanceStop) {
    ArrayXi py;
    ArrayXXd X = tolData(&py);
    KmeansNANI km(X, 6, MD::Metric::MSD, MD::KinitType::CompSim, 1, 10, 16, -1, 1e-4);
    ArrayXi labels = km.getLabels();
    ASSERT_EQ(labels.size(), py.size());
    for (int i = 0; i < labels.size(); ++i) EXPECT_EQ(labels[i], py[i]) << "frame " << i;
}

TEST(KMeansNaniTol, ZeroRunsToConvergence) {
    ArrayXi py;
    ArrayXXd X = tolData(&py);
    // (the default, as before: the labels settle, unlike MDANCE's here)
    KmeansNANI byDefault(X, 6, MD::Metric::MSD, MD::KinitType::CompSim, 1, 10);
    KmeansNANI zero(X, 6, MD::Metric::MSD, MD::KinitType::CompSim, 1, 10, 16, -1, 0.0);
    ArrayXi a = byDefault.getLabels(), b = zero.getLabels();
    EXPECT_TRUE((a == b).all());
    EXPECT_GT((a != py).count(), 0);
}

TEST(KMeansNaniTol, LabelsAreTheLastCentres) {
    // stopped by tol, the labels are taken again from the last centres
    // (scikit-learn's last E-step): each frame is nearest its centre
    ArrayXXd X = tolData();
    KmeansNANI km(X, 6, MD::Metric::MSD, MD::KinitType::CompSim, 1, 10, 16, -1, 1e-2);
    ArrayXi labels = km.getLabels();
    Mat C = km.getCenters();
    for (Index i = 0; i < X.rows(); ++i) {
        Index best;
        (C.rowwise() - X.row(i)).square().rowwise().sum().minCoeff(&best);
        EXPECT_EQ(labels[i], (int)best) << "frame " << i;
    }
}

TEST(KMeansNaniTol, RejectsNegativeOrNaN) {
    ArrayXXd X = tolData();
    EXPECT_THROW(KmeansNANI(X, 6, MD::Metric::MSD, MD::KinitType::CompSim, 1, 10, 16, -1, -1e-4), std::invalid_argument);
    EXPECT_THROW(KmeansNANI(X, 6, MD::Metric::MSD, MD::KinitType::CompSim, 1, 10, 16, -1, std::nan("")), std::invalid_argument);
}
