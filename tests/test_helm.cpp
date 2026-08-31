#include <filesystem>
#include <list> 

#include "test_helm.h"
#include "test_utils.h"

#include "../../src/cluster/helm.h"
#include "../../tools/bts.h"
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

    std::vector<HCTree> clusters;
    int N0 = uniqueLabels.size();
    for ( int i = 0; i < N0; i++) {
        std::list<int> indices = {i};
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
        // todo: zind is set to 1 --> Not sure if that is correct
        HCTree tree = HCTree();
        tree.insertRoot(indices, cSumi, sqSumi, ni, i);
        clusters.push_back(tree);
    }
    if (N0 != clusters.size()) {
        throw std::runtime_error("Error in inputCluster: number of unique labels does not match number of clusters.");
    }
    clusterTree = clusters;
}

ArrayXXd TestHelm::makeDataByRow(ArrayXd a, ArrayXd b){
        //a and b need to be same length
        //a will be csum, and b will be sqsum
        //this function necessary for extendedComparison()
        Mat data(2, a.size());
        data.row(0) = a;
        data.row(1) = b;
        return data;
}

Mat TestHelm::getSelectedData(vector<HCTree> clusters){
    // extracts data for frames that are in the resulting clusters. This is necessary for computing CH and DB scores.

    vector<int> selectedClusterLabels; // contains all the cluster labels that are in the resulting clusters. 
    for(auto c:clusters){
        for(int i:c.getRootClusterIndices()){
            selectedClusterLabels.emplace_back(i);
        }
    }
    vector<int> selectedFrameIndices; // list of  frames within the selected clusters.
    for(int i:selectedClusterLabels){
        for(int j=0; j<labels.size(); j++){
            if(labels(j)==i){
                selectedFrameIndices.emplace_back(j);
            }
        }
    }
    Mat selectedData = data(selectedFrameIndices, Eigen::all);
    return selectedData;
}

TEST_F(TestHelm, TestPops){
    int nAtoms = 50;
    int nClusters = 10;
    Helm helm = Helm(clusterTree, nAtoms, MD::Metric::MSD, MD::MergeScheme::Inter, nClusters); 
    vector<HCTree> clusters = helm.run();

    ASSERT_EQ(clusters.size(), nClusters);

    std::vector<double> pops;
    std::vector<std::vector<int>> merged_clusts;
    for (int i = 0; i < clusters.size(); i++) {
        double pop = (double) clusters[i].getRootNObjects()/6001.0;
        pops.push_back(pop);
        std::list<int> idx = clusters[i].getRootClusterIndices();
        // convert idx to vector:
        std::vector<int> idx_vec(idx.begin(), idx.end());
        merged_clusts.push_back(idx_vec);
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

        // we need to sort cluster indices since these are stored in a list. 
        std::sort(expected_merged_clusters[i].begin(), expected_merged_clusters[i].end());
        
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
    Helm helm = Helm(clusterTree, nAtoms, MD::Metric::MSD, MD::MergeScheme::Inter, nClusters);
    vector<HCTree> clusters = helm.run();

    ASSERT_EQ(clusters.size(), nClusters);

    //Compute CH and DB scores
    Mat selectedData = getSelectedData(clusters);
    pair<double, double> scoreRes = helm.computeScores(clusters, selectedData);

    // compare scores to expected values
    EXPECT_NEAR(scoreRes.first, 329.3068227654963, 1e-5);
    EXPECT_NEAR(scoreRes.second, 1.8437066236912167, 1e-5);
}

TEST_F(TestHelm, TrimK){
    int nAtoms = 50;
    int nClusters = 8; 
    bool trimStart = true;
    int trimK = 1;
    double minSamples = 0.025; 
    // trim_start=True, trim_k=1, trim_val=None, min_samples=0.025)()
    Helm helm = Helm(clusterTree,
        nAtoms, 
        MD::Metric::MSD, 
        MD::MergeScheme::Inter, 
        nClusters,
        -1, // default eps value
        trimStart,
        minSamples,
        0, // default trimVal value
        trimK
    ); 
    vector<HCTree> clusters = helm.run();
    int expectedNClusters = 8;
    // check if 8 in the map
    ASSERT_EQ(clusters.size(), expectedNClusters);

    // to check trimming, we need to have 8 clusters --> traverse the cluster tree to get the 8 clusters.


    std::vector<double> msds;
    std::vector<int> Niks;
    for (int i = 0; i < clusters.size(); i++) {
        ArrayXXd data = makeDataByRow(clusters[i].getRootCSum(), clusters[i].getRootSQSum());
        int Ni = clusters[i].getRootNObjects();
        Niks.push_back(Ni);
        double msd = extendedComparison(data, Ni, nAtoms, true, MD::Metric::MSD);
        msds.push_back(msd);
        double pop = (double) clusters[i].getRootNObjects()/6001.0;
        EXPECT_GT(pop, 0.025);
    }

    // check if cluster sizes are correct
    std::vector<int> expected_Niks = {205, 161, 568, 153, 211, 160, 180, 158};
    for (int i = 0; i < Niks.size(); i++) {
        EXPECT_EQ(Niks[i], expected_Niks[i]);
    }

    // check msds
    std::vector<double> expected_msds = {
        0.7159290983066504, 2.339201422302611, 3.362941769846286, 
        3.53136323974767, 5.2539237168022215, 5.458759986365163, 
        5.9705582725822515, 6.09399997860286
    };
    for (int i = 0; i < msds.size(); i++) {
        EXPECT_NEAR(msds[i], expected_msds[i], 1e-5);
    }

    // check scores
    Mat selectedData = getSelectedData(clusters);
    pair<double, double> scoreRes = helm.computeScores(clusters, selectedData);
    EXPECT_NEAR(scoreRes.first, 1027.0159808301096, 1e-5);
    EXPECT_NEAR(scoreRes.second, 1.3503102468493706, 1e-5);
}

TEST_F(TestHelm, TrimK2){
    int nAtoms = 50;
    int nClusters = 10; 
    bool trimStart = true;
    int trimK = 50;
    double minSamples = 0.0;
    Helm helm = Helm(clusterTree,
        nAtoms, 
        MD::Metric::MSD, 
        MD::MergeScheme::Inter, 
        nClusters,
        -1, // default eps value
        trimStart,
        minSamples,
        0, // default trimVal value
        trimK
    );
    vector<HCTree> clusters = helm.run();
    int expectedNClusters = 10;
    ASSERT_EQ(expectedNClusters, clusters.size());

    std::vector<double> msds;
    for (int i = 0; i < clusters.size(); i++) {
        ArrayXXd data = makeDataByRow(clusters[i].getRootCSum(), clusters[i].getRootSQSum());
        int Ni = clusters[i].getRootNObjects();
        double msd = extendedComparison(data, Ni, nAtoms, true, MD::Metric::MSD);
        msds.push_back(msd);
    }
    // check msds
    std::vector<double> expected_msds = {
        0.533111401482992, 0.7159290983066504, 
        0.84978424877854, 1.5715154323945075, 
        2.2918233940183708, 2.339201422302611, 
        2.394802731761288, 2.516757040247173, 
        2.5290370776959334, 2.788671261009775
    };
    for (int i = 0; i < msds.size(); i++) {
        EXPECT_NEAR(msds[i], expected_msds[i], 1e-5);
    }
}

TEST_F(TestHelm, TestEps){
    int N0 = uniqueLabels.size();
    inputCluster();
    int nAtoms = 50;
    float eps = 15;

    Helm helm = Helm(clusterTree,
        nAtoms, 
        MD::Metric::MSD, 
        MD::MergeScheme::Inter, 
        0,
        eps // default eps value
    ); 
    vector<HCTree> clusters = helm.run();
    std::cout << "Number of clusters: " << clusters.size() << std::endl;

    //Compute CH and DB scores
    Mat selectedData = getSelectedData(clusters);
    pair<double, double> scoreRes = helm.computeScores(clusters, selectedData);
    EXPECT_NEAR(scoreRes.first, 302.7012989347063, 1e-5);
    EXPECT_NEAR(scoreRes.second, 1.755325543510106, 1e-5);
}

TEST_F(TestHelm, TestTrimVal){
    int N0 = uniqueLabels.size();
    inputCluster();
    int nAtoms = 50;
    int nClusters = 4;
    bool trimStart = true;
    int trimVal = 5;
    float minSamples = 0.025;

    Helm helm = Helm(clusterTree,
        nAtoms, 
        MD::Metric::MSD, 
        MD::MergeScheme::Inter, 
        nClusters,
        -1, // default eps value
        trimStart,
        minSamples,
        trimVal
    );

   vector<HCTree> clusters = helm.run();
    int expectedNClusters = 4;
    EXPECT_EQ(clusters.size(), expectedNClusters);

    vector<double> msds;
    for(auto c:clusters){
        Vec cSum = c.getRootCSum();
        Vec sqSum = c.getRootSQSum();
        int Nik = c.getRootNObjects();
        ArrayXXd data = makeDataByRow(cSum, sqSum);
        double msd = extendedComparison(data, Nik, nAtoms, true);

        msds.emplace_back(msd);
    }

    vector<double> expectedMsds = {
        0.7159290983066504, 
        2.339201422302611, 
        3.362941769846286, 
        3.53136323974767
    };

    for(int i=0; i<expectedMsds.size(); i++){
        EXPECT_NEAR(msds[i], expectedMsds[i], 1e-5);
        EXPECT_LT(msds[i], 5);
    }
}

TEST_F(TestHelm, ZMatrix){
    int nAtoms = 50;
    int nClus = 37;
    Helm helm = Helm(clusterTree,
        nAtoms, 
        MD::Metric::MSD, 
        MD::MergeScheme::Inter, 
        nClus
    );
    vector<HCTree> clusters = helm.run();
    Mat Z = helm.getZMatrix();
    Mat expectedZ(23, 4);
    expectedZ <<2, 23, 1.0, 2, 
        47, 60, 2.0, 3, 
        15, 31, 3.0, 2, 
        22, 45, 4.0, 2, 
        5, 55, 5.0, 2, 
        26, 34, 6.0, 2, 
        9, 41, 7.0, 2, 
        14, 27, 8.0, 2, 
        4, 17, 9.0, 2, 
        8, 62, 10.0, 3, 
        0, 11, 11.0, 2, 
        3, 6, 12.0, 2, 
        19, 25, 13.0, 2, 
        57, 63, 14.0, 3, 
        35, 52, 15.0, 2, 
        18, 64, 16.0, 3, 
        20, 42, 17.0, 2, 
        16, 68, 18.0, 3, 
        66, 75, 19.0, 5, 
        1, 72, 20.0, 3, 
        21, 65, 21.0, 3, 
        36, 40, 22.0, 2, 
        51, 56, 23.0, 2;
    for(int i=0; i<expectedZ.rows(); i++){
        for(int j=0; j<4; j++){
            EXPECT_EQ(Z(i,j), expectedZ(i,j));
        }
    }
        
}