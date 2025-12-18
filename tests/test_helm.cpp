#include <filesystem>

#include "test_helm.h"
#include "test_utils.h"

#include "../../src/cluster/helm.cpp"

TestHelm::TestHelm() {
    // read in data:
    std::filesystem::path buildTestPath = std::filesystem::current_path();
    std::filesystem::path projectRoot = buildTestPath.parent_path().parent_path();
    std::filesystem::path dataPath = projectRoot / "tests" / "data";
    std::filesystem::path inputDataFile = dataPath / "sim.csv";
    data = readCSVtoEigen(inputDataFile.string());

    // read label file
    std::filesystem::path labelFile = dataPath / "sim_labels_helm.csv";
    ArrayXXd labelData = readCSVtoEigen(labelFile.string());
    labels = labelData.col(1).cast<int>();

    // initialize clusters_map
    inputCluster();
}

 vector<Cluster> TestHelm::inputCluster(){
    for (int i = 0; i < labels.size(); i++) {
        uniqueLabels.insert(labels(i));
    }
    std::vector<Cluster> clusters;
    int N0 = uniqueLabels.size();
    for ( int i = 0; i < N0; i++) {
        std::vector<int> indices = {i};
        Vec cSumi = Vec::Zero(data.cols()); 
        Vec sqSumi = Vec::Zero(data.cols());
        int ni = 0;
        for (int j = 0; j < labels.size(); j++) {
            if (labels(j) == i) {
                ni++;
                cSumi += data.row(j);
                sqSumi += data.row(j).square();
            }
        }
        Eigen::Map<Veci> idx(indices.data(), indices.size());
        clusters.push_back(Cluster(idx, cSumi, sqSumi, ni));
    }
    if (N0 != clusters.size()) {
        throw std::runtime_error("Error in inputCluster: number of unique labels does not match number of clusters.");
    }
    clusters_map[N0] = clusters;

    return clusters;
}

TEST_F(TestHelm, TestPops){
    int nAtoms = 50;
    int nClusters = 10;
    Helm helm = Helm(clusters_map, nAtoms, MD::Metric::MSD, MD::MergeScheme::Inter, nClusters); 
    map<int, vector<Cluster>> res = helm.run();
    std::vector<double> pops;
    std::vector<Cluster> clusters = res[10];
    std::vector<std::vector<int>> merged_clusts;
    for (int i = 0; i < clusters.size(); i++) {
        double pop = (double) clusters[i].getN()/6001.0;
        pops.push_back(pop);
        Veci idx = clusters[i].getIndices();
        std::vector<int> temp_idx(idx.data(), idx.data() + idx.size());
        merged_clusts.push_back(temp_idx);
    }
    std::sort(pops.begin(), pops.end());
    std::vector<double> expected_pops = { 0.32511248, 0.21213131, 0.11898017, 0.11714714, 0.09948342, 0.07648725, 0.02216297,  0.01366439, 0.00799867, 0.00683219};
    std::vector<std::vector<int>> expected_merged_clusters = {
        {33}, 
        {49}, 
        {53}, 
        {0, 11}, 
        {47, 2, 23, 36, 40, 38, 58}, 
        {1, 19, 25, 8, 15, 31, 9, 41, 18, 5, 55}, 
        {50, 51, 56}, 
        {35, 52, 57, 22, 45, 32, 37}, 
        {12, 44, 28, 54, 30, 48, 59, 39, 20, 42}, 
        {13, 29, 7, 10, 3, 6, 16, 4, 17, 43, 24, 14, 27, 46, 21, 26, 34}
    };

    std::sort(expected_pops.begin(), expected_pops.end());
    for (int i = 0; i < pops.size(); i++) {
        EXPECT_NEAR(pops[i], expected_pops[i], 0.00001);
    }

    for (int i = 0; i < merged_clusts.size(); i++) {
        ASSERT_EQ(merged_clusts[i].size(), expected_merged_clusters[i].size());
        for (int j = 0; j < merged_clusts[i].size(); j++) {
            EXPECT_EQ(merged_clusts[i][j], expected_merged_clusters[i][j]);
        }
    }
}

TEST_F(TestHelm, TestClus){
    int N0 = uniqueLabels.size();
    vector<Cluster> clusters = inputCluster();
    int nAtoms = 50; 
    int nClusters = 37;
    Helm helm = Helm(clusters_map, nAtoms, MD::Metric::MSD, MD::MergeScheme::Inter, nClusters);
    map<int, vector<Cluster>> res = helm.run();

    
}
