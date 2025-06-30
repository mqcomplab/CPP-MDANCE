#include "nani.h";

void pairwiseDistance(MatrixXd& data, MatrixXd& mu, MatrixXd& dist){
    int n = data.rows();
    int d = data.cols();
    int k = mu.rows();

    // For small dims D, for loop is noticeably faster than fully vectorized.
    // Odd but true.  So we do fastest thing 
    if (d <= 16){
        for (int kk=0; kk<k; ++kk){
            dist.col(kk) = (data.rowwise() - mu.row(kk)).array().square().rowwise().sum();
        }
    } else {
        dist = -2*(data * mu.transpose());
        dist.rowwise() += mu.array().square().rowwise().sum().transpose().row(0);
    }
}

double assignClosest(MatrixXd& data, MatrixXd& mu, VectorXi& labels, MatrixXd& dist){
    double totalDist = 0;
    int minRowID;

    pairwiseDistance(data, mu, dist);

    for (int nn=0; nn<data.rows(); ++nn){
        totalDist += dist.row(nn).minCoeff( &minRowID);
        labels(nn) = minRowID;
    }
    return totalDist;
}

void calcMu(MatrixXd& data, MatrixXd& mu, VectorXi& labels){
    mu.fill(0);
    VectorXd nPerCluster(mu.rows());
    nPerCluster.fill(1e-100);  // avoid division by 0
    for (int nn=0; nn<data.rows(); ++nn){
        mu.row(labels(nn)) += data.row(nn);
        ++nPerCluster[labels(nn)];
    }
    for (int k=0; k < mu.rows(); ++k) {
        mu.row(k) /= nPerCluster(k);
    }
}

void kMeans(MatrixXd& data, MatrixXd& mu, VectorXi& labels, int iterations){
    double prevDist, totalDist = 0;
    MatrixXd dist(data.rows(), mu.rows());

    for (int iter=0; iter<iterations; ++iter){
        totalDist = assignClosest(data, mu, labels, dist);
        calcMu(data, mu, labels);
        if (prevDist == totalDist) {
            break;
        }
        prevDist = totalDist;
    }
}

MatrixXd getTop(MatrixXd& data, int percentage, Metric mt, int M){
    int nMax = data.rows() * percentage / 100;
    VectorXd compSimResults = compSim(data, mt, M);
    sortvd compSims = sortCompSims(data.rows(), compSimResults);
    MatrixXd topCCdata(nMax,data.row(0).size());
    for(int i=1; i<=nMax; ++i){
        topCCdata.row(nMax-i) = data.row(compSims[data.rows()-i].second);
    }
    return topCCdata;
}

MatrixXd initKmean(MatrixXd& data, int nClusters, Metric mt, int M, InitiateKTypes type, int percentage=0){
    MatrixXd initiators;
    MatrixXd topCCdata;
    if (percentage <=0 || percentage > 100) {
        cerr << "percentage must be an integer [0, 100].\n";
        return;
    }
    switch (type){
        case stratAll: initiators = data(diversitySelection(data, percentage, mt, M, strat), Eigen::all); break;
        case divSelect: initiators = data(diversitySelection(data, percentage, mt, M, medoid), Eigen::all); break;
        case vanillakpp: break;
        case kMeanspp: break;
        case random: break;
        case InitiateKTypes::compSim: topCCdata = getTop(data, percentage, mt, M); initiators = topCCdata(diversitySelection(topCCdata, 100, mt, M, medoid), Eigen::all); break;
        case stratReduced: topCCdata = getTop(data, percentage, mt, M); initiators = topCCdata(diversitySelection(topCCdata, 100, mt, M, strat), Eigen::all); break;
    }
    return initiators(Eigen::seq(0,nClusters));
}
