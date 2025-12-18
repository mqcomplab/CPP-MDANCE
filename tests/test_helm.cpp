#include <filesystem>

#include "test_helm.h"
#include "test_utils.h"

#include "../../src/cluster/helm.cpp"
#include "../../tools/scores.cpp"

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

 void TestHelm::inputCluster(){
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
    inputCluster();
    int nAtoms = 50; 
    int nClusters = 37;
    Helm helm = Helm(clusters_map, nAtoms, MD::Metric::MSD, MD::MergeScheme::Inter, nClusters);
    map<int, vector<Cluster>> res = helm.run();

    vector<pair<double, double>> scores;
    for(auto it=res.begin(); it!=res.end(); it++){
        vector<int> idx;
        for(auto c:it->second){
            idx.insert(idx.end(), c.getIndices().begin(), c.getIndices().end());
        }

        vector<int> temp;
        for(int i:idx){
            for(int j=0; j<labels.size(); j++){
                if(labels(j)==i){
                    temp.emplace_back(j);
                }
            }
        }
        Mat arr = data(temp, Eigen::placeholders::all);
        scores.emplace_back(helm.computeScores(it->second, arr));
    }

    vector<pair<double, double>> expectedScores = {
        {291.2198060306322, 1.7370614645545726},
        {295.7352641684398, 1.7122884537735075}, 
        {58296.11490509768595, 1.7245038612367665}, 
        {297.8213492701506, 1.7246552370601154}, 
        {299.0810730592307, 1.738637643465005}, 
        {300.34386300863565, 1.750692719498292}, 
        {302.7012989347063, 1.755325543510106}, 
        {304.51797241739484, 1.7672459242576242}, 
        {306.6006824661377, 1.7695122974000195}, 
        {308.9021982976074, 1.7607190308607732}, 
        {307.1901532394585, 1.7721711289472055}, 
        {297.43176362926556, 1.7888202076551794}, 
        {299.34757173879535, 1.7833018738518591}, 
        {301.52432336513584, 1.7936056164868994}, 
        {306.08279103249237, 1.8034172355518991}, 
        {310.3208501789975, 1.809472686067903}, 
        {310.0361137593372, 1.825336725435764}, 
        {313.8978629372179, 1.8344594141834631}, 
        {315.0527200319327, 1.828991467839653}, 
        {314.5428710018461, 1.818951502602474}, 
        {314.94309021259465, 1.8091147842592137}, 
        {318.77039278363225, 1.8100164274989112}, 
        {323.7748207939142, 1.817475357075402}, 
        {329.3068227654963, 1.8437066236912167}
    };

    for(int i=0; i<scores.size(); i++){
        EXPECT_NEAR(scores[i].first, expectedScores[i].first, 1e-5);
        EXPECT_NEAR(scores[i].second, expectedScores[i].second, 1e-5);
    }
}
