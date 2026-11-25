#pragma once

#include <iostream>
#include <stdexcept>
#include <limits>
#include <numeric>
#include <chrono>
#include <cstdint>
#include <random>
#include <string>


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
    std::mt19937 rng;
    Vec rowSq;      // each frame's sum of squares (rowSqNorms)

    double genrand_double();
    void rowSqNorms();

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
    // seed -1: a seed from the clock (scikit-learn's random_state=None); the
    // seed used is getSeed(). Only the Random and k-means++ seedings draw.
    KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, MD::KinitType kinit = MD::KinitType::StratAll, int nAtoms = 1, int percentage = 10, int vectThreshold=16, int seed = -1);
    KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, Mat centers, int nAtoms = 1, int percentage = 10, int vectThreshold=16, int seed = -1);
    int getSeed() const { return seed; }
    map<int,vector<Index>> createClusterDict();
    pair<double, double> computeScores();
    Veci getLabels();
    Mat getCenters();
};

