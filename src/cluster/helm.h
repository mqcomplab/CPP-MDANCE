#include "../tools/types.h"
#include "../tools/cluster.h"

class Helm{
    public:
        Helm(map<int, vector<Cluster>> clusterMap, int nAtoms, MD::Metric mt = MD::Metric::MSD, 
                MD::MergeScheme mergeScheme = MD::MergeScheme::Inter, int nClusters = 0, float eps = -1, 
                bool trimStart = false, MD::AlignMethod alignMeth = MD::AlignMethod::None, 
                float minSamples = 0.01, MD::Link link = MD::Link::None,
                float trimVal=0, float trimK=0,
                bool savePairwiseSum = false,
                string inputTop ="", string inputTraj ="");
        map<int, vector<Cluster>> run();
        pair<double, double> computeScores(vector<Cluster> clusters, Mat data);
        Mat calculateZMatrix(map<int, vector<Cluster>> clusterMap);
    private:
        map<int, vector<Cluster>> clusterMap;
        int nAtoms;
        int nClusters;
        float eps; // -1 means None 
        bool trimStart;
        float trimVal;
        float trimK;
        float minSamples;
        int trimIncoming;
        MD::Metric mt;
        MD::MergeScheme mergeScheme;
        MD::AlignMethod alignMeth;
        MD::Link link;
        Mat clusterDists;
        Mat linkMatrix;
        int totalIncoming;
        bool savePairwiseSum;
        string inputTop;
        string inputTraj;
        int totalSum;

        Mat makeDataByRow(Vec a, Vec b);
        map<int, vector<Cluster>> trimClusters();
        vector<Cluster> genNewClusters(vector<Cluster>& previousClusters);
        float calc(vector<Cluster>& previousClusters, int i, int j);
        void genClusterDists(vector<Cluster>& previousClusters);
        Mat initialPairwiseMatrix(vector<Cluster>& previousClusters);
        map<int, vector<Cluster>> linkMatrixToClusterMap();
};