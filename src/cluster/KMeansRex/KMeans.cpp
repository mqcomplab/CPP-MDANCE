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
#include "mersenneTwister2002.h"

// ====================================================== Utility Functions
void KmeansNANI::set_seed() {
    init_genrand( seed );
}

void KmeansNANI::set_vectorization_threshold(int threshold){
    // set vectorization threshold for distance matrix computation
    if (threshold < 1){
        throw std::invalid_argument("Threshold must be at least 1");
    }
    if (threshold == std::numeric_limits<int>::infinity()) {
        throw std::invalid_argument("Threshold must be finite");
    }
    threshold = threshold;
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
    Mat topCCdata = data(topIndices, Eigen::placeholders::all);
    vector<Index> idx = diversitySelection(topCCdata, 100, mt, nAtoms, isComp);
    centers = topCCdata(idx,Eigen::placeholders::all);
}

void KmeansNANI::init_Mu() {
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
        centers = data(idx,Eigen::placeholders::all);
        break;

    case MD::KinitType::DivSelect:
        idx = diversitySelection(data, percentage, mt, nAtoms, true);
        centers = data(idx,Eigen::placeholders::all);
        break;
    }
    // only take first kClusters centers
    if (centers.rows() > kClusters){
        centers = centers(Eigen::seq(0, kClusters-1), Eigen::placeholders::all).eval();
    }
}

//======================================================= Update Assignments Z
void KmeansNANI::pairwise_distance( Mat &X, Mat &Mu, Mat &Dist ) {
    int N = X.rows();
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
        Dist.colwise() += X.square().rowwise().sum();
    }
}


double KmeansNANI::assignClosest(int idx) {
    double totalDist = 0;
    int minRowID;

    pairwise_distance( data, centers, dist );
    //std::cout<<std::setprecision(10)<<dist.row(1321)<<std::endl;

    for (int nn=0; nn<data.rows(); nn++) {
        totalDist += dist.row(nn).minCoeff( &minRowID );
        labels(nn,0) = minRowID;
    }
    return totalDist;
}

// ======================================================= Update Locations Mu
double KmeansNANI::calcMu() {
    //Mu = Mat::Zero(Mu.rows(), Mu.cols());
    Mat new_centers = Mat::Zero(2, data.row(0).size());
    Vec NperCluster = Vec::Zero(centers.rows());
    for (int nn=0; nn<data.rows(); nn++) {
        new_centers.row((int) labels(nn,0)) += data.row(nn);
        NperCluster[(int) labels(nn,0)] += 1;
    }  
    NperCluster += MD::EPSILON_DIV; // avoid division-by-zero
    for (int k=0; k < centers.rows(); k++) {
        new_centers.row(k) /= NperCluster(k);
    }

    double centerShift = 0;
    for (int i=0; i < centers.rows(); i++){
        double diff = (new_centers.row(i) - centers.row(i)).square().sum();
        centerShift += diff;
    }

    centers = new_centers;

    return centerShift;
}

// ======================================================= Overall Lloyd Alg.
void KmeansNANI::run_lloyd(int Niter, float tol)  {
    double prevDist,totalDist = 0;

    // TODO: store the labels at each frame
    for (int iter=0; iter<Niter; iter++) {
        totalDist = assignClosest(iter);
        double centerShift = calcMu();
        std::cout<<centerShift<<std::endl;
        if (centerShift <= tol) {
            std::cout<<centerShift<<tol<<std::endl;
            break;
        }
        prevDist = totalDist;
    }
}


// based off of scikit kmeans
/*
void KmeansNANI::run_lloyd(Mat data, Mat centersInit, int Niter, float tol){
    int nClust=centers.rows();
    int nFeat=centers.cols();
    int n=data.rows();

    Mat centers=centersInit;
    Mat newCenters=Mat::Zero(nClust, nFeat);
    Veci labels=Veci::Zero(n);
    Veci oldLabels=labels;

    Vec weightClusters=Vec::Zero(nClust);
    Vec centerShift=Vec::Zero(nClust);

    for (int i=0; i<Niter; i++){
        lloyd ll{
            data,
            sampleWeight,
            centers,
            newCenters,
            weightClusters,
            labels,
            centerShift
        };
        lloyd_iter()
    }
}

void KmeansNANI::lloyd_iter(lloyd& ll){
    int nSamp = ll.data.rows();
    int nFeat = ll.data.cols();
    int nClust = ll.oldCenters.rows();

    int nSampChunk = nSamp > CHUNK_SIZE ? CHUNK_SIZE : nSamp;
    int nChunk = nSamp / nSampChunk;
    int nSampRem = nSamp % nSampChunk;

    int chunkIdx;
    int start, end;

    Vec centerSqNorm = ll.oldCenters.square().rowwise().sum();

    Mat centersNewChunk = Mat::Zero(nClust, nFeat);
    Vec weightClustersChunk = Vec::Zero(nClust);
    Mat pairwiseDistChunk = Mat::Zero(nSamp, nClust);

    #pragma omp parallel for schedule(static)
    for (int chunkIdx=0; chunkIdx < nChunk; ++chunkIdx){
        start = chunkIdx * nSampChunk;
        if (chunkIdx == nChunk - 1 && nSampRem > 0){
            end = start + nSampRem;
        }
        else{
            end = start + nSampChunk;
        }
        
        chunk ch{
            ll.data(Eigen::seq(start, end-1), Eigen::placeholders::all),
            ll.sampleWeight(Eigen::seq(start, end-1)),
            ll.oldCenters,
            centerSqNorm,
            ll.labels(Eigen::seq(start, end-1)),
            centersNewChunk,
            weightClustersChunk,
            pairwiseDistChunk
        };
        updateChunk(ch);
    }

    // update teh centers
    for (int i=0; i<nClust; i++){
        ll.weightClusters[i] += weightClustersChunk[i];
        for (int j=0; j<nFeat; j++){
            ll.newCenters(i,j) += centersNewChunk(i,j);
        }
    }

    relocateEmptyClusters(
        ll.data, 
        ll.sampleWeight,
        ll.oldCenters,
        ll.newCenters,
        ll.weightClusters,
        ll.labels
    );

    averageCenters(
        ll.newCenters,
        ll.weightClusters
    );

    centerShift(
        ll.oldCenters,
        ll.newCenters,
        ll.centerShift
    );
}

void KmeansNANI::updateChunk(chunk& ch){
    int nSamp = ch.labels.size();
    int nClust = ch.oldCenters.rows();
    int nFeat = ch.oldCenters.cols();

    float sqDist, minSqDist;
    int i, j, k, label;

    for (int i=0; i<nSamp; i++){
        for (int j=0; j<nClust; j++){
            ch.pairwiseDistances(i,j) = ch.centerSqNorm[j];
        }
    }
    ch.pairwiseDistances += -2 * ch.data * ch.oldCenters.transpose(); 

    for (int i=0; i<nSamp; i++){
        minSqDist = ch.pairwiseDistances(i,0);
        for (int j=1; j<nClust; j++){
            sqDist = ch.pairwiseDistances(i,j);
            if (sqDist < minSqDist){
                minSqDist = sqDist;
                label = j;
            }
        }
        ch.labels[i] = label;

        ch.weightClusters[label] += ch.sampleWeight[i];
        for (int k=0; k<nFeat; k++){
            ch.newCenters(j,k) += ch.data(i,k) + ch.sampleWeight[i];
        }
    }
}
void KmeansNANI::relocateEmptyClusters(Mat data, Vec sampleWeight, Mat oldCenters, Mat& newCenters, Vec& weightClusters, Veci labels){
    vector<int> emptyClusters;
    for(int i=0; i<weightClusters.size(); i++){
        if (weightClusters[i]==0){
            emptyClusters.push_back(i);
        }
    }
    int nEmpty=emptyClusters.size();

    if (nEmpty==0){
        return;
    }

    int nFeat = data.cols();
    Vec distances(data.rows());
    for(int i=0; i<data.rows(); i++){
        distances[i] = (data.row(i) - oldCenters.row(labels[i])).square().sum();
        //https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_k_means_lloyd.pyx
    }
    std::sort(distances.data(), distances.data()+distances.size(), std::greater<double>());
    Veci farFromCenter=distances.head(nEmpty);

    if(distances.head(1).data()==0){
        return;
    }

    int newClusterIdx;
    int oldClusterIdx;
    for(int i=0; i<nEmpty; i++){
        newClusterIdx=emptyClusters[i];
        int farIdx=farFromCenter[i];
        float weight=sampleWeight[farIdx];
        oldClusterIdx=labels[farIdx];

        for(int k=0; k<nFeat; k++){
            newCenters(oldClusterIdx, k)-=data(farIdx,k)*weight;
            newCenters(newClusterIdx, k)=data(farIdx,k)*weight;
        }

        weightClusters[newClusterIdx]=weight;
        weightClusters[oldClusterIdx]-=weight;
    }
}
void KmeansNANI::averageCenters(Mat& centers, Vec weightClusters){
    int nClust=centers.rows();
    int nFeat=centers.cols();
    float alpha;
    Eigen::Index argmaxWeight;
    double maxWeight=weightClusters.maxCoeff(&argmaxWeight);

    for(int j=0; j<nClust; j++){
        if(weightClusters[j]>0){
            alpha=1.0/weightClusters[j];
            for(int k=0; k<nFeat; k++){
                centers(j,k)*=alpha;
            }
        }
        else{
            for(int k=0; k<nFeat; k++){
                centers(j,k)=centers(argmaxWeight,k);
            }
        }
    }
}
void KmeansNANI::centerShift(Mat oldCenters, Mat newCenters, Vec& centerShift){
    int nClust=oldCenters.rows();
    int nFeat=oldCenters.cols();

    for(int j=0; j<nClust; j++){
        centerShift[j]=euclidean(
            newCenters.col(j),
            oldCenters.col(j),
            nFeat,
            false
        );
    }
}
float KmeansNANI::euclidean(Vec a, Vec b, int nFeat, bool squared){
    int n=nFeat/4;
    int rem=nFeat%4;
    float result=0;

    for(int i=0; i<n; i++){
        result+=(
            (a[0+4*i]-b[0+4*i]) * (a[0+4*i]-b[0+4*i]) + 
            (a[1+4*i]-b[1+4*i]) * (a[1+4*i]-b[1+4*i]) + 
            (a[2+4*i]-b[2+4*i]) * (a[2+4*i]-b[2+4*i]) + 
            (a[3+4*i]-b[3+4*i]) * (a[3+4*i]-b[3+4*i])
        );
    }

    for(int i=0;i<rem; i++){
        result+=(a[i+4*n]-b[i+4*n]) * (a[i+4*n]-b[i+4*n]);
    }

    if (squared==true){
        return result;
    }
    return std::sqrt(result);
}
*/

KmeansNANI::KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, MD::KinitType kinit, int nAtoms, int percentage, int vectThreshold, int seed) : data(data), kClusters(kClusters), mt(mt), nAtoms(nAtoms), kinit(kinit), seed(seed), percentage(percentage) {
    centers = Mat::Zero(kClusters, data.cols());
    dist = Mat::Zero(data.rows(), kClusters);
    labels = Veci::Zero(data.rows());
    set_vectorization_threshold(vectThreshold);
    set_seed();
    init_Mu();
    run_lloyd(300);
}
KmeansNANI::KmeansNANI(ArrayXXd data, int kClusters, MD::Metric mt, Mat centers, int nAtoms, int percentage, int vectThreshold, int seed) : data(data), kClusters(kClusters), mt(mt), nAtoms(nAtoms), seed(seed), percentage(percentage), centers(centers) {
    dist = Mat::Zero(data.rows(), kClusters);
    labels = Veci::Zero(data.rows());
    set_vectorization_threshold(vectThreshold);
    set_seed();

    Vec mean = data.colwise().mean();
    Vec var = ((data.rowwise() - mean.transpose()).square().colwise().sum() / data.rows()).transpose();
    float tol = var.mean() * 0.0001;
    std::cout<<tol<<std::endl;
    run_lloyd(300, tol);
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
