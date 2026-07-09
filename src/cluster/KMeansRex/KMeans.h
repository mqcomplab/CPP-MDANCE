#pragma once

#include <iostream>
#include <stdexcept>
#include <limits>
#include <iomanip>
#include <cmath>
#include <omp.h>


#include "../../tools/bts.h"
#include "../../tools/types.h"
#include "../../tools/scores.h"

constexpr int CHUNK_SIZE = 256;

struct lloyd{
    Mat data;
    Vec sampleWeight;
    Mat oldCenters;
    Mat& newCenters;
    Vec& weightClusters;
    Veci& labels;
    Vec centerShift;
};

struct chunk{
    Mat data;                   //sample x feat
    Vec sampleWeight;           //sample x 1
    Mat oldCenters;             //cluster x feat
    Vec centerSqNorm;
    Veci& labels;
    Mat& newCenters;            //cluster x feat
    Vec& weightClusters;        //cluster x 1
    Mat& pairwiseDistances;     //sample x cluster
};

class KmeansNANI{
    Mat data;
    Mat centers;
    Mat dist;
    Veci labels;
    MD::KinitType kinit;
    int seed;
    int kClusters;
    MD::Metric mt;
    int nAtoms;
    int percentage;
    int vectorizationThreshold;

    void set_seed();
    void set_vectorization_threshold(int threshold);
    int randint(int low, int high);
    int discrete_rand(Vec &p);
    void select_without_replacement(int N, int K, Vec &chosenIDs);
    void sampleRowsRandom();
    void sampleRowsPlusPlus();
    void reduced_init_Mu(bool isComp);
    void init_Mu();
    void pairwise_distance(Mat &X, Mat &Mu, Mat &Dist);
    double assignClosest(int idx);
    double calcMu();
    void run_lloyd(int Niter, float tol=0);

    /*
    void lloyd_iter(lloyd& ll);
    void updateChunk(chunk& ch);
    void relocateEmptyClusters(Mat data, Vec sampleWeight, Mat oldCenters, Mat& newCenters, Vec& weightClusters, Veci labels);
    void averageCenters(Mat& centers, Vec weightClusters);
    void centerShift(Mat oldCenters, Mat newCenters, Vec& centerShift);
    float euclidean(Vec a, Vec b, int nFeat, bool squared);
    */

public: 
    KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, MD::KinitType kinit = MD::KinitType::StratAll, int nAtoms = 1, int percentage = 10, int vectThreshold=16, int seed=0);
    KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, Mat centers, int nAtoms = 1, int percentage = 10, int vectThreshold=16, int seed=0);
    map<int,vector<Index>> createClusterDict();
    pair<double, double> computeScores();
    Veci getLabels();
    Mat getCenters();
};

