#include "../tools/types.h"
#include "../tools/cluster.h"
#include "../tools/hc_utils.h"

class Helm{
    public:
        Helm(vector<HCTree> clusterTree, int nAtoms, MD::Metric mt = MD::Metric::MSD, 
                MD::MergeScheme mergeScheme = MD::MergeScheme::Inter, int nClusters = 0, float eps = -1, 
                bool trimStart = false, MD::AlignMethod alignMeth = MD::AlignMethod::None, 
                float minSamples = 0.01, MD::Link link = MD::Link::None,
                float trimVal=0, float trimK=0,
                bool savePairwiseSum = false,
                string inputTop ="", string inputTraj ="");
        vector<HCTree> run();
        pair<double, double> computeScores(vector<Cluster> clusters, Mat data);
        Mat calculateZMatrix();
    private:
        vector<HCTree> clusterTree;
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
        Mat zMatrix;

        Mat makeDataByRow(Vec a, Vec b);
        vector<HCTree> trimClusters();
        vector<HCTree> genNewClusters(int ZIdx);
        float calcHelmSim(HCTree& firstTree, HCTree& secondTree);
        void genClusterDists(vector<HCTree>& previousClusters);
        void updateZMatrix(int idxA, int idxB, int mergedClusts);
        Mat initialPairwiseMatrix(vector<Cluster>& previousClusters);
        map<int, vector<Cluster>> linkMatrixToClusterMap();
};