#pragma once

#include <iostream>
#include <stdexcept>
#include <limits>


#include "../../tools/bts.h"
#include "../../tools/types.h"
#include "../../tools/scores.h"



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
    double assignClosest();
    void calcMu();
    void run_lloyd(int Niter);

public: 
    KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, MD::KinitType kinit = MD::KinitType::StratAll, int nAtoms = 1, int percentage = 10, int vectThreshold=16);
    KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, Mat centers, int nAtoms = 1, int percentage = 10, int vectThreshold=16);
    map<int,vector<Index>> createClusterDict();
    pair<double, double> computeScores();
    Veci getLabels();
    Mat getCenters();
};

