/* An adapted version of KMeansRexCore.cpp
A fast, easy-to-read implementation of the K-Means clustering algorithm.
allowing customized initialization (random samples or plus plus)
and vectorized execution via the Eigen matrix template library.

Intended to be compiled as a shared library which can then be utilized
from high-level interactive environments, such as Matlab or Python.

Contains
--------

Utility Fcns 
* discrete_rand : sampling discrete random variable
* select_without_replacement : sample without replacement

Cluster Location Mu Initialization:
* sampleRowsRandom : sample rows of X at random (w/out replacement)
* sampleRowsPlusPlus : sample rows of X via kmeans++ procedure of Arthur et al.
    see http://en.wikipedia.org/wiki/K-means%2B%2B

K-Means Algorithm (aka Lloyd's Algorithm)
* run_lloyd : executes lloyd for specfied number of iterations

Dependencies:
  mersenneTwister2002.c : random number generator

Author: Mike Hughes (www.michaelchughes.com)
Date:   2 April 2013
*/


#include "KMeans.h"
#include <chrono>
#include <cstdint>

// ====================================================== Utility Functions
void KmeansNANI::set_seed() {
    // Note: if seed is -1, we will generate a random seed. 
    // This is in line with scikit-learns random_state=None, which we use in the python version 
    if (seed == -1){
        // generate random seed
        uint64_t t = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        seed = (int)((t ^ (t >> 32)) & 0x7fffffff);   // (not negative: a seed a caller can pass back)
    }
    // (one generator per instance: the global one of mersenneTwister2002.c
    // made concurrent calls race. std::mt19937 is the same MT19937, seeded
    // the same way as init_genrand, so a seed gives the numbers it gave.)
    rng.seed(static_cast<uint32_t>(seed));
}

double KmeansNANI::genrand_double() {
    return rng() * (1.0/4294967295.0);
}

void KmeansNANI::set_vectorization_threshold(int threshold){
    // set vectorization threshold for distance matrix computation
    if (threshold < 1){
        throw std::invalid_argument("Threshold must be at least 1");
    }
    vectorizationThreshold = threshold;
}

/*
* Return random integers from `low` (inclusive) to `high` (exclusive).
*/
int KmeansNANI::randint(int low, int high) {
    double r = ((high - low)) * genrand_double();
    int rint = (int) r; // [0,1) -> 0, [1,2) -> 1, etc
    return rint + low;
}

int KmeansNANI::discrete_rand( Vec &p ) {
    double total = p.sum();
    int K = (int) p.size();
    
    double r = total*genrand_double();
    double cursum = p(0);
    int newk = 0;
    while ( r >= cursum && newk < K-1) {
        newk++;
        cursum += p[newk];
    }
    if ( newk < 0 || newk >= K ) {
        throw std::runtime_error("Badness. Chose illegal discrete value.");
    }
    return newk;
}

void KmeansNANI::select_without_replacement( int N, int K, Vec &chosenIDs) {
    Vec p = Vec::Ones(N);
    for (int kk =0; kk<K; kk++) {
        int choice;
        int doKeep = false;
        while ( doKeep==false) {
            doKeep=true;
            choice = discrete_rand( p );
    
            for (int previd=0; previd<kk; previd++) {
                if (chosenIDs[previd] == choice ) {
                doKeep = false;
                break;
                }
            }      
        }      
        chosenIDs[kk] = choice;     
    }
}

// ======================================================= Init Cluster Locs Mu

void KmeansNANI::sampleRowsRandom() {
    int N = data.rows();
    int K = centers.rows();
    Vec ChosenIDs = Vec::Zero(K);
    select_without_replacement(N, K, ChosenIDs);
    for (int kk=0; kk<K; kk++) {
        centers.row( kk ) = data.row( ChosenIDs[kk] );
    }
}

void KmeansNANI::sampleRowsPlusPlus() {
    int N = data.rows();
    int K = centers.rows();
    if (K > N) {
        // User requested more clusters than we have available.
        // So, we'll fill only first N rows of Mu
        // and leave all remaining rows of Mu uninitialized.
        K = N;
    }
    int choice = randint(0, N); 
    centers.row(0) = data.row( choice );
    Vec minDist(N);
    Vec curDist(N);
    for (int kk=1; kk<K; kk++) {
        curDist = (data.rowwise() - centers.row(kk-1)).square().rowwise().sum();
        if (kk==1) {
            minDist = curDist;
        } else {
            minDist = curDist.min( minDist );
        }      
        choice = discrete_rand( minDist );
        centers.row(kk) = data.row( choice );
    }       
}

void KmeansNANI::reduced_init_Mu(bool isComp) {
    int nTotal = data.rows();
    int nMax = (int)((long long)nTotal * percentage / 100);
    // The seeds are drawn from the nMax frames of highest complementary
    // similarity: with fewer than k of them the diversity selection ran on
    // an empty matrix (a crash at 5 frames and 10%).
    if (nMax < kClusters) {
        throw std::runtime_error("Initialization would draw " + std::to_string(kClusters) +
            " centers from " + std::to_string(nMax) + " frames (" + std::to_string(percentage) +
            "% of " + std::to_string(nTotal) + "); increase percentage or reduce kClusters.");
    }
    Vec compSims = calculateCompSim(data, nAtoms, mt);
    vector<pair<double,int>> compSimArray;
    compSimArray.reserve(compSims.size());
    for (int i=0; i<compSims.size(); ++i){
        compSimArray.emplace_back(compSims[i],i);
    }
    std::sort(compSimArray.begin(), compSimArray.end());
    vector<Index> topIndices;
    topIndices.reserve(nMax);
    for (int i=nTotal - nMax; i<nTotal; ++i){
        topIndices.push_back(compSimArray[i].second);
    }
    Mat topCCdata = data(topIndices, Eigen::all);
    vector<Index> idx = diversitySelection(topCCdata, 100, mt, nAtoms, isComp);
    centers = topCCdata(idx,Eigen::all);
}

void KmeansNANI::init_Mu() {
    // (over 100, the reduced seedings started their loop at a negative index)
    if (percentage < 0 || percentage > 100) {
        throw std::invalid_argument("percentage must be between 0 and 100.");
    }
    vector<Index> idx;
    switch (kinit)
    {
    case MD::KinitType::Random:
        sampleRowsRandom();
        break;
    
    // The greedy k-means++ variant is not implemented yet; fall back to the
    // vanilla k-means++ sampling rather than silently leaving the centers at
    // their zero-initialized values.
    case MD::KinitType::KmeansPP:
    case MD::KinitType::VanillaKmeansPP:
        sampleRowsPlusPlus();
        break;

    case MD::KinitType::CompSim:
        reduced_init_Mu(true);
        break;

    case MD::KinitType::StratReduced:
        reduced_init_Mu(false);
        break;

    case MD::KinitType::StratAll:
        idx = diversitySelection(data, percentage, mt, nAtoms);
        centers = data(idx,Eigen::all);
        break;

    case MD::KinitType::DivSelect:
        idx = diversitySelection(data, percentage, mt, nAtoms, true);
        centers = data(idx,Eigen::all);
        break;
    }
    // only take first kClusters centers
    if (centers.rows() > kClusters){
        centers = centers(Eigen::seq(0, kClusters-1), Eigen::all).eval();
    }
    // The percentage-based initializations can select fewer than kClusters
    // candidates; catching it here beats silently clustering with fewer
    // centers than requested (labels would never reach kClusters-1).
    if (centers.rows() < kClusters){
        throw std::runtime_error("Initialization produced fewer centers than kClusters; increase percentage or reduce kClusters.");
    }
}

// ======================================================= Update Assignments Z
void KmeansNANI::pairwise_distance( Mat &X, Mat &Mu, Mat &Dist ) {
    int D = X.cols();
    int K = Mu.rows();

    // For small dims D, for loop is noticeably faster than fully vectorized.
    // Odd but true.  So we do fastest thing
    if ( D <= vectorizationThreshold ) {
        for (int kk=0; kk<K; kk++) {
            Dist.col(kk) = (X.rowwise() - Mu.row(kk)).square().rowwise().sum();
        }
    } else {
        Dist = -2*(X.matrix() * Mu.transpose().matrix());
        Dist.rowwise() += Mu.square().rowwise().sum().transpose().row(0);
    }
}

// todo: this function got fairly long, may be useful to extract the reallocation of empty clusters. 
double KmeansNANI::assignClosest() {
    double totalDist = 0;
    int minColID;

    pairwise_distance( data, centers, dist );

    // check if there are empty clusters
    Vec best_dists; // storing this so it can be sorted from worst fitting point to best
    best_dists.resize(data.rows());
    Veci populations = Eigen::ArrayXi::Zero(kClusters);

    for (int nn=0; nn<data.rows(); nn++) {
        double d = dist.row(nn).minCoeff( &minColID );
        totalDist += d; 
        labels(nn,0) = minColID;
        best_dists[nn] = d + rowSq[nn]; // add x^2 into distance so it is comparable across points (computed once: rowSqNorms)
        // todo: check if adding x^2 to the pairwise_distance would be faster
        populations[minColID] += 1; // increase population of given cluster by 1
    }

    if ((populations == 0).any()){
        // there are empty clusters
        // call function here to reassign clusters?
        std::vector<int> order(best_dists.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), 
                        [&](int a, int b) {return best_dists[a] > best_dists[b];});

        vector<int> emptyClusters;
        for (int k =0; k < populations.size(); ++k){
            if (populations[k] ==0){
                emptyClusters.push_back(k);
            }
        }
        size_t e = 0; // cluster to fill
        for (int p : order){
            if (e == emptyClusters.size()){ // filled all the clusters
                break;
            }
            if (best_dists[p] <= 0.0){
                // all remaining points fit perfectly: some clusters stay empty
                break;
            }
            int old_k = labels(p,0);
            if (populations[old_k] <=1){
                continue;
            }

            // found suitable point
            int new_k = emptyClusters[e];
            labels(p, 0) = new_k;
            // update populations
            populations[old_k] = populations[old_k] - 1;
            populations[new_k] = populations[new_k] + 1;
            ++e;
            // update total distance: point gets assigned to new cluster
            // distance to centroid is 0 in that case.
            totalDist -= dist(p, old_k);
        }
    }

    return totalDist;
}

// ======================================================= Update Locations Mu
void KmeansNANI::calcMu() {
    //Mu = Mat::Zero(Mu.rows(), Mu.cols());
    centers.fill(0);
    Vec NperCluster = Vec::Zero(centers.rows());
    // Column by column: data.row(nn) is strided by the number of frames, a
    // cache miss per value at every iteration. Each center's value is still
    // the sum of its frames' values in frame order: the same numbers.
    const Index N = data.rows();
    for (Index j=0; j<data.cols(); j++) {
        const double* col = data.col(j).data();
        for (Index nn=0; nn<N; nn++) {
            centers((int) labels(nn,0), j) += col[nn];
        }
    }
    for (Index nn=0; nn<N; nn++) {
        NperCluster[(int) labels(nn,0)] += 1;
    }
    NperCluster += MD::EPSILON_DIV; // avoid division-by-zero
    for (int k=0; k < centers.rows(); k++) {
    centers.row(k) /= NperCluster(k);
    }
}

// ======================================================= Overall Lloyd Alg.
void KmeansNANI::run_lloyd(int Niter )  {
    // prevDist must start at a value assignClosest() can never return so the
    // convergence check cannot fire on the first iteration.
    double prevDist = std::numeric_limits<double>::infinity();
    double totalDist = 0;

    // TODO: store the labels at each frame
    for (int iter=0; iter<Niter; iter++) {
        MD::checkCancel();
        totalDist = assignClosest();
        calcMu();
        if (prevDist == totalDist) {
            break;
        }
        prevDist = totalDist;
    }
}

// Note: if seed is -1, we will generate a random seed. 
// This is in line with scikit-learns random_state=None, which we use in the python version 
// set seed to any other number if we need deterministic random numbers for debugging purposes. 

void KmeansNANI::rowSqNorms() {
    // Each row's sum of squares, in the order Eigen sums a row of a
    // column-major array (one after the other), so best_dists is unchanged.
    const Index N = data.rows();
    rowSq.resize(N);
    for (Index nn=0; nn<N; nn++) {
        double r = data(nn, 0) * data(nn, 0);
        for (Index j=1; j<data.cols(); j++) r += data(nn, j) * data(nn, j);
        rowSq[nn] = r;
    }
}

KmeansNANI::KmeansNANI(ArrayXXd data_, int kClusters, MD::Metric mt, MD::KinitType kinit, int nAtoms, int percentage, int vectThreshold, int seed_) : data(std::move(data_)), kinit(kinit), seed(seed_), kClusters(kClusters), mt(mt), nAtoms(nAtoms), percentage(percentage) {
    if (kClusters < 1 || kClusters > this->data.rows()) {
        throw std::invalid_argument("kClusters must be between 1 and the number of data points.");
    }
    if (this->data.cols() == 0) {
        throw std::invalid_argument("the data has no columns.");
    }
    centers = Mat::Zero(kClusters, data.cols());
    dist = Mat::Zero(this->data.rows(), kClusters);
    labels = Veci::Zero(this->data.rows());
    set_vectorization_threshold(vectThreshold);
    set_seed();
    rowSqNorms();
    init_Mu();
    run_lloyd(300);
}

// Note: if seed is -1, we will generate a random seed. 
// This is in line with scikit-learns random_state=None, which we use in the python version 
// set seed to any other number if we need deterministic random numbers for debugging purposes. 

KmeansNANI::KmeansNANI(ArrayXXd data_, int kClusters, MD::Metric mt, Mat centers_, int nAtoms, int percentage, int vectThreshold, int seed_) : data(std::move(data_)), centers(std::move(centers_)), kinit(MD::KinitType::StratAll), seed(seed_), kClusters(kClusters), mt(mt), nAtoms(nAtoms), percentage(percentage) {
    if (kClusters < 1 || kClusters > this->data.rows()) {
        throw std::invalid_argument("kClusters must be between 1 and the number of data points.");
    }
    if (this->centers.rows() != kClusters || this->centers.cols() != this->data.cols()) {
        throw std::invalid_argument("centers must have shape (kClusters, nFeatures).");
    }
    dist = Mat::Zero(this->data.rows(), kClusters);
    labels = Veci::Zero(this->data.rows());
    set_vectorization_threshold(vectThreshold);
    set_seed();
    rowSqNorms();
    run_lloyd(300);
}
map<int,vector<Index>> KmeansNANI::createClusterDict() {
    map<int,vector<Index>> clusterDict;
    for (int i=0; i<kClusters; ++i){
        clusterDict[i] = vector<Index>();
    }
    for (int i=0; i<labels.size(); ++i){
        clusterDict[labels[i]].push_back(i);
    }
    return clusterDict;
}
pair<double, double> KmeansNANI::computeScores() {
    double ch = calinskiHarabaszScore(data, labels);
    double db = daviesBouldinScore(data, labels);
    return std::make_pair(ch, db);
}
Veci KmeansNANI::getLabels() {
    return labels;
}
Mat KmeansNANI::getCenters() {
    return centers;
}
