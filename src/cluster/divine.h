#pragma once

#include "../tools/bts.h"
#include "../tools/types.h"
#include "KMeansRex/KMeans.h"


class Divine{
    Mat data;
    vector<int> labels;

    MD::Metric mt;
    MD::DivineSplit splitType;
    MD::DivineAnchors anchorType;
    MD::KinitType kinit;

    int end;    //0 for 'k', 1 for 'points'
    int kClusters;
    bool refine;
    int nAtoms;
    double threshold;
    int percentage;

    vector<vector<Index>> clusters;
    vector<pair<double, double>> scores;
    vector<int> clusterSizes;

    void divisiveAlgorithm();
    Index selectClusterToSplit(vector<bool>& failedSplits);
    bool splitCluster(Index clusterToSplit, int minFrames);

public:
    Divine(Mat data, MD::DivineSplit splitType = MD::DivineSplit::WeightedMSD, 
        MD::DivineAnchors anchorType = MD::DivineAnchors::NANI, MD::KinitType kinit = MD::KinitType::StratAll, 
        int end = 0, int k = 0, bool refine = true, int nAtoms = 1, double threshold = 0, int percenntage = 10);
    vector<vector<Index>> getClusters();
    vector<int> getLabels();
    vector<pair<double, double>> getScores();
    pair<double, double> computeScores(Veci labels, Mat data);
    void createLabels(int nTotal);
};