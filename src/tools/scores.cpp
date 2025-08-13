#include "scores.h"

/* Returns the Calinski-Harabasz score for a given cluster array.
 * 
 * Parameters:
 *  - data: A feature array of size (nSamples, nFeatures)
 *  - labels: A list of labels.
 * 
 * Returns: the CH score
 * 
 * Reference: This code does not include any of the parameter checking: https://github.com/scikit-learn/scikit-learn/blob/c5497b7f7eacfaff061cf68e09bcd48aa93d4d6b/sklearn/metrics/cluster/_unsupervised.py#L325
*/
double calinskiHarabaszScore(MatrixXd data, VectorXi labels) {
    std::map<int, vector<int>> clusters;

    for (int i=0; i<labels.size(); ++i) {
        if (clusters.find(labels[i]) == clusters.end()){
            vector<int> k = {i};
            clusters[labels[i]] = k;
        } else {
            clusters[labels[i]].emplace_back(i);
        }
    }

    double extraDisp = 0;
    double intraDisp = 0;
    VectorXd mean = data.colwise().sum() / data.rows();
    
    for (auto k=clusters.begin(); k != clusters.end(); ++k) {
        MatrixXd cluster = data(*k,Eigen::all);
        VectorXd clusterMean = cluster.colwise().sum() / cluster.rows();

        extraDisp += cluster.rows() * (clusterMean - mean).array().square().sum();

        for (int i=0; i<cluster.rows(); ++i) {
            intraDisp += (cluster.row(i) - clusterMean).array().square().sum();
        }
    }

    if (intraDisp == 0)
        return 1;
    
    return extraDisp * (data.rows() - clusters.size()) / (intraDisp * (clusters.size() - 1.0));
}

/* Returns the Davies-Bouldin score for a given cluster array.
 * 
 * Parameters:
 *  - data: A feature array of size (nSamples, nFeatures)
 *  - labels: A list of labels.
 * 
 * Returns: the DB score
 * 
 * Reference: This code does not include any of the parameter checking: https://github.com/scikit-learn/scikit-learn/blob/c5497b7f7eacfaff061cf68e09bcd48aa93d4d6b/sklearn/metrics/cluster/_unsupervised.py#L396
*/
double daviesBouldinScore(MatrixXd data, VectorXi labels) {
    std::map<int, vector<int>> clusters;

    for (int i=0; i<labels.size(); ++i) {
        if (clusters.find(labels[i]) == clusters.end()){
            vector<int> k = {i};
            clusters[labels[i]] = k;
        } else {
            clusters[labels[i]].emplace_back(i);
        }
    }

    MatrixXd centroids(clusters.size(), data.row(0).size());
    VectorXd intraDists(clusters.size());
    MatrixXd scores(clusters.size(), clusters.size());
    VectorXd maxScores(clusters.size());
    int c=0;

    for (auto k=clusters.begin(); k != clusters.end(); ++k) {
        MatrixXd cluster = data(*k,Eigen::all);
        centroids.row(c) = cluster.colwise().sum() / cluster.rows();
        intraDists[c] = 0;
        for (int i=0; i<cluster.rows(); ++i) {
            intraDists[c] += std::sqrt((cluster.row(i) - centroids.row(c)).array().square().sum());
        }
        intraDists[c] /= cluster.rows();

        scores.row(c)[c] = 0;
        maxScores[c] = 0;
        for (int i=0; i<c; ++i) {
            scores.row(c)[i] = (intraDists[i] + intraDists[c]) / std::sqrt((centroids.row(i) - centroids.row(c)).array().square().sum());
            scores.row(i)[c] = scores.row(c)[i];

            if (scores.row(i)[c] > maxScores[i])
                maxScores[i] = scores.row(i)[c];
            if (scores.row(c)[i] > maxScores[c])
                maxScores[c] = scores.row(c)[i];
        }

        ++c;
    }    
    
    return maxScores.sum() / maxScores.size();
}