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

ArrayXXd TestHelm::makeDataByRow(ArrayXd a, ArrayXd b){
        //a and b need to be same length
        //a will be csum, and b will be sqsum
        //this function necessary for extendedComparison()
        Mat data(2, a.size());
        data.row(0) = a;
        data.row(1) = b;

        return data;
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
    for(auto it=res.rbegin(); it!=res.rend(); it++){
        vector<int> idx;
        for(auto c:it->second){
            for(int i:c.getIndices()){
                idx.emplace_back(i);
            }
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
        {296.11490509768595, 1.7245038612367665}, 
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

TEST_F(TestHelm, TrimK){
    int nAtoms = 50;
    int nClusters = 1; 
    bool trimStart = true;
    int trimK = 1;
    double minSamples = 0.025; 
    // trim_start=True, trim_k=1, trim_val=None, min_samples=0.025)()
    Helm helm = Helm(clusters_map,
        nAtoms, 
        MD::Metric::MSD, 
        MD::MergeScheme::Inter, 
        nClusters,
        -1, // default eps value
        trimStart,
        MD::AlignMethod::None, // default alignMeth value
        minSamples,
        MD::Link::None, // default link value
        0, // default trimVal value
        trimK
    ); 
    map<int, vector<Cluster>> res = helm.run();
    int expectedNClusters = 8;
    // check if 8 in the map
    ASSERT_EQ(res.find(expectedNClusters) != res.end(), true);

    // check number of clusters
    std::vector<Cluster> clusters = res[expectedNClusters];
    ASSERT_EQ(clusters.size(), expectedNClusters);

    std::vector<double> msds;
    std::vector<int> Niks;
    for (int i = 0; i < clusters.size(); i++) {
        ArrayXXd data = makeDataByRow(clusters[i].getCsum(), clusters[i].getSQsum());
        int Ni = clusters[i].getN();
        Niks.push_back(Ni);
        double msd = extendedComparison(data, Ni, nAtoms, true, MD::Metric::MSD);
        msds.push_back(msd);
        double pop = (double) clusters[i].getN()/6001.0;
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

    int N0 = clusters.size();

    vector<pair<double, double>> scores;

    for(auto it=res.rbegin(); it!=res.rend(); it++){
        vector<int> idx;
        for(auto c:it->second){
            for(int i:c.getIndices()){
                idx.emplace_back(i);
            }
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

    //check scores
    std::vector<pair<double, double>> expectedScores = {
        {1027.0159808301096, 1.3503102468493706}, 
        {1104.807519769014, 1.205066197610796}, 
        {1172.6483112177802, 0.8824355830201499}, 
        {1227.2333015488437, 0.9712858749211579}, 
        {1332.727994435348, 0.9596798547189016}, 
        {1381.3259570829998, 1.1368566266419298}, 
        {1482.6296232781137, 0.9381045938050468}
    };
    for(int i=0; i<expectedScores.size(); i++){
        EXPECT_NEAR(scores[i].first, expectedScores[i].first, 1e-5);
        EXPECT_NEAR(scores[i].second, expectedScores[i].second, 1e-5);
    }
    EXPECT_NEAR(scores.back().first, -1.0, 1e-5);
    EXPECT_NEAR(scores.back().second, -1.0, 1e-5);
}

TEST_F(TestHelm, TrimK2){
    int nAtoms = 50;
    int nClusters = 1; 
    bool trimStart = true;
    int trimK = 50;
    double minSamples = 0.0;
    Helm helm = Helm(clusters_map,
        nAtoms, 
        MD::Metric::MSD, 
        MD::MergeScheme::Inter, 
        nClusters,
        -1, // default eps value
        trimStart,
        MD::AlignMethod::None, // default alignMeth value
        minSamples,
        MD::Link::None, // default link value
        0, // default trimVal value
        trimK
    );
    map<int, vector<Cluster>> res = helm.run();
    int expectedNClusters = 10;
    ASSERT_EQ(res.find(expectedNClusters) != res.end(), true);

    // check number of clusters
    std::vector<Cluster> clusters = res[expectedNClusters];
    ASSERT_EQ(clusters.size(), expectedNClusters);

    std::vector<double> msds;
    for (int i = 0; i < clusters.size(); i++) {
        ArrayXXd data = makeDataByRow(clusters[i].getCsum(), clusters[i].getSQsum());
        int Ni = clusters[i].getN();
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