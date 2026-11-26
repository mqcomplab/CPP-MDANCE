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
    // (own before data: data refers to own, or to the caller's data)
    Mat own;
    const Mat& data;
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
    double tol;     // 0: Lloyd until the labels stop changing; > 0: scikit-learn's stop (run_lloyd)
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
    void pairwise_distance(const Mat &X, const Mat &Mu, Mat &Dist);
    void run_seeded(int vectThreshold);
    double assignClosest();
    void calcMu();
    void run_lloyd(int Niter);
    void run_lloyd_tol(int Niter);

public: 
    // seed -1: a seed from the clock (scikit-learn's random_state=None); the
    // seed used is getSeed(). Only the Random and k-means++ seedings draw.
    // tol 0 (the default): Lloyd's iterations until the labels stop changing.
    // tol > 0: scikit-learn's KMeans stop, as MDANCE (Python) runs it (its
    // default tol is 1e-4): also once the centres' squared shift, summed, is
    // at most tol times the mean variance of the features, the labels then
    // taken again from the last centres (run_lloyd_tol).
    KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, MD::KinitType kinit = MD::KinitType::StratAll, int nAtoms = 1, int percentage = 10, int vectThreshold=16, int seed = -1, double tol = 0.0);
    // On data the caller keeps unchanged while this runs (no copy is made).
    KmeansNANI(const ArrayXXd* borrowed, int kClusters, MD::Metric mt, MD::KinitType kinit = MD::KinitType::StratAll, int nAtoms = 1, int percentage = 10, int vectThreshold=16, int seed = -1, double tol = 0.0);
    KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, Mat centers, int nAtoms = 1, int percentage = 10, int vectThreshold=16, int seed = -1);
    int getSeed() const { return seed; }
    map<int,vector<Index>> createClusterDict();
    pair<double, double> computeScores();
    Veci getLabels();
    Mat getCenters();
    KmeansNANI(const KmeansNANI&) = delete;
    KmeansNANI& operator=(const KmeansNANI&) = delete;
};

