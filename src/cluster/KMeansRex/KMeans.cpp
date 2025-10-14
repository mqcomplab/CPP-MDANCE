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

#include <iostream>
#include "mersenneTwister2002.c"

#include "../../tools/types.h"
#include "../../tools/bts.h"
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


    // ====================================================== Utility Functions
    void set_seed() {
    init_genrand( seed );
    }

    /*
    * Return random integers from `low` (inclusive) to `high` (exclusive).
    */
    int randint(int low, int high) {
        double r = ((high - low)) * genrand_double();
        int rint = (int) r; // [0,1) -> 0, [1,2) -> 1, etc
        return rint + low;
    }

    int discrete_rand( Vec &p ) {
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
            std::cerr << "Badness. Chose illegal discrete value." << std::endl;
            return -1;
        }
        return newk;
    }

    void select_without_replacement( int N, int K, Vec &chosenIDs) {
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

    void sampleRowsRandom() {
        int N = data.rows();
        int K = centers.rows();
        Vec ChosenIDs = Vec::Zero(K);
        select_without_replacement(N, K, ChosenIDs);
        for (int kk=0; kk<K; kk++) {
            centers.row( kk ) = data.row( ChosenIDs[kk] );
        }
    }

    void sampleRowsPlusPlus() {
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

    void reduced_init_Mu(bool isComp) {
        int nTotal = data.rows();
        int nMax = nTotal * percentage / 100;
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

    void init_Mu() {
        vector<Index> idx;
        switch (kinit)
        {
        case MD::KinitType::Random:
            sampleRowsRandom();
            break;
        
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
    }

    // ======================================================= Update Assignments Z
    void pairwise_distance( Mat &X, Mat &Mu, Mat &Dist ) {
        int N = data.rows();
        int D = data.cols();
        int K = centers.rows();

        // For small dims D, for loop is noticeably faster than fully vectorized.
        // Odd but true.  So we do fastest thing 
        if ( D <= 16 ) {
            for (int kk=0; kk<K; kk++) {
                Dist.col(kk) = (data.rowwise() - centers.row(kk)).square().rowwise().sum();
            }    
        } else {
            Dist = -2*(data.matrix() * centers.transpose().matrix());
            Dist.rowwise() += centers.square().rowwise().sum().transpose().row(0);
        }
    }

    double assignClosest() {
        double totalDist = 0;
        int minRowID;

        pairwise_distance( data, centers, dist );

        for (int nn=0; nn<data.rows(); nn++) {
            totalDist += dist.row(nn).minCoeff( &minRowID );
            labels(nn,0) = minRowID;
        }
        return totalDist;
    }

    // ======================================================= Update Locations Mu
    void calc_Mu() {
        //Mu = Mat::Zero(Mu.rows(), Mu.cols());
        centers.fill(0);
        Vec NperCluster = Vec::Zero(centers.rows());
        for (int nn=0; nn<data.rows(); nn++) {
            centers.row((int) labels(nn,0)) += data.row(nn);
            NperCluster[(int) labels(nn,0)] += 1;
        }  
        NperCluster += 1e-100; // avoid division-by-zero
        for (int k=0; k < centers.rows(); k++) {
        centers.row(k) /= NperCluster(k);
        }
    }

    // ======================================================= Overall Lloyd Alg.
    void run_lloyd(int Niter )  {
        double prevDist,totalDist = 0;

        // TODO: store the labels at each frame
        for (int iter=0; iter<Niter; iter++) {
            totalDist = assignClosest();
            calc_Mu();
            if (prevDist == totalDist) {
                break;
            }
            prevDist = totalDist;
        }
    }


public: 
    KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, MD::KinitType kinit = MD::KinitType::StratAll, int nAtoms = 1, int percentage = 10) : data(data), kClusters(kClusters), mt(mt), nAtoms(nAtoms), kinit(kinit), seed(seed), percentage(percentage) {
        centers = Mat::Zero(kClusters, data.cols());
        dist = Mat::Zero(data.rows(), kClusters);
        labels = Veci::Zero(data.rows());
        set_seed();
        init_Mu();
        run_lloyd(300);
    }
    KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, Mat centers, int nAtoms = 1, int percentage = 10) : data(data), kClusters(kClusters), mt(mt), nAtoms(nAtoms), kinit(kinit), seed(seed), percentage(percentage), centers(centers) {
        dist = Mat::Zero(data.rows(), kClusters);
        labels = Veci::Zero(data.rows());
        set_seed();
        run_lloyd(300);
    }
    map<int,vector<Index>> createClusterDict() {
        map<int,vector<Index>> clusterDict;
        for (int i=0; i<kClusters; ++i){
            clusterDict[i] = vector<Index>();
        }
        for (int i=0; i<labels.size(); ++i){
            clusterDict[labels[i]].push_back(i);
        }
        return clusterDict;
    }
    pair<double, double> computeScores() {
        double ch = calinskiHarabaszScore(data, labels);
        double db = daviesBouldinScore(data, labels);
        return std::make_pair(ch, db);
    }
    Veci getLabels() {
        return labels;
    }
    Mat getCenters() {
        return centers;
    }
};