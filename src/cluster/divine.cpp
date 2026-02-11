#include "divine.h"


//ready for testing
void Divine::divisiveAlgorithm() {
    int minFrames = std::max(1, (int)round(threshold * data.rows()));

    while (true) {
        if(end==0){
            if(clusters.size() >= kClusters)    break;
        }
        else if(end==1){
            bool flag=false;
            for(Index i=0; i<clusters.size(); i++){
                if(clusters[i].size()!=1)   flag=true;
            }
            if(!flag)   break;
        }
        vector<bool> failedSplits(clusters.size(), false);
        bool didSplit = false;
        while (!didSplit) {
            Index clusterToSplit = selectClusterToSplit(failedSplits);
            if (clusterToSplit < 0) {
                throw std::runtime_error("No more cluster splits possible that would yield valid subclusters");
            } 
            didSplit = splitCluster(clusterToSplit, minFrames);
            failedSplits[clusterToSplit] = !didSplit;
        }
    }
};
Index Divine::selectClusterToSplit(vector<bool>& failedSplits) {
    Index topCluster = -1;
    double bestScore = -1;

    for (Index i = 0; i < clusters.size(); ++i) {
        if (failedSplits[i] || clusters[i].size() < 2) {
            continue;
        }

        double score = -1;
        Mat subdata = data(clusters[i], Eigen::placeholders::all);
        if (splitType == MD::DivineSplit::MSD) {
            score = extendedComparison(subdata, 0, nAtoms, false, mt);
        } else if (splitType == MD::DivineSplit::Radius) {
            Index medoidIdx = calculateMedoid(subdata, nAtoms, mt);
            Vec medoid = subdata.row(medoidIdx);
            Vec dists = (subdata.rowwise() - medoid.transpose()).square().rowwise().sum() / nAtoms;
            score = dists.maxCoeff();
        } else if (splitType == MD::DivineSplit::WeightedMSD) {
            score = clusters[i].size() * extendedComparison(subdata, 0, nAtoms, false, mt);  
        }

        if (score > bestScore) {
            bestScore = score;
            topCluster = i;
        }
    }
    return topCluster;
};
bool Divine::splitCluster(Index clusterToSplit, int minFrames) {
    if (clusters[clusterToSplit].size() < 2) {
        std::cerr<<"Cannot split a cluster with less than 2 points";
        return false;
    }

    Veci subdataIndices;
    subdataIndices.resize(clusters[clusterToSplit].size());
    for(int i=0; i<clusters[clusterToSplit].size(); i++){
        subdataIndices[i]=clusters[clusterToSplit][i];
    }
    Mat subdata = data(subdataIndices, Eigen::placeholders::all);
    
    if (anchorType == MD::DivineAnchors::NANI) {
        KmeansNANI kmeans(subdata, 2, mt, kinit, nAtoms, percentage);
        ArrayXi sublabels = kmeans.getLabels();
        vector<Index> cluster1, cluster2;
        for (Index i = 0; i < sublabels.size(); ++i) {
            if (sublabels[i] == 0) {
                cluster1.push_back(clusters[clusterToSplit][i]);
            } else {
                cluster2.push_back(clusters[clusterToSplit][i]);
            }
        }

        clusters[clusterToSplit] = cluster1;
        clusters.push_back(cluster2);
    } else if (anchorType == MD::DivineAnchors::OutlierPair) {
        Index outlierIdx = calculateOutlier(subdata, nAtoms, mt);
        Vec anchorA = subdata.row(outlierIdx);
        Vec dists = (subdata.rowwise() - anchorA.transpose()).square().rowwise().sum() / nAtoms;
        Index idxFurthest;
        dists.maxCoeff(&idxFurthest);
        Vec anchorB = subdata.row(idxFurthest);

        Mat dataC = data(clusters[clusterToSplit], Eigen::placeholders::all);
        Vec dA = (dataC.rowwise() - anchorA.transpose()).square().rowwise().sum() / nAtoms;
        Vec dB = (dataC.rowwise() - anchorB.transpose()).square().rowwise().sum() / nAtoms;

        vector<Index> initialMask;
        vector<Index> notInitialMask;
        initialMask.reserve(subdata.rows());
        notInitialMask.reserve(subdata.rows());
        for (Index i = 0; i < dA.size(); ++i) {
            if (dA[i] < dB[i]) {
                initialMask.push_back(clusters[clusterToSplit][i]);
            } else {
                notInitialMask.push_back(clusters[clusterToSplit][i]);
            }
        }

        if (refine) {
            Mat groupA = subdata(initialMask, Eigen::placeholders::all);
            Mat groupB = subdata(notInitialMask, Eigen::placeholders::all);
            Index medoidA = groupA.size() <= 2 ? 0 : calculateMedoid(groupA, nAtoms, mt);
            Index medoidB = groupB.size() <= 2 ? 0 : calculateMedoid(groupB, nAtoms, mt);
            Mat initiators = Mat::Zero(2, data.row(0).size());
            initiators.row(0) = groupA.row(medoidA);
            initiators.row(1) = groupB.row(medoidB);
            KmeansNANI kmeans(subdata, 2, mt, initiators, nAtoms);
            Veci sublabels = kmeans.getLabels();
            vector<Index> cluster1, cluster2;
            for (Index i = 0; i < sublabels.size(); ++i) {
                if (sublabels[i] == 0) {
                    cluster1.push_back(clusters[clusterToSplit][i]);
                } else {
                    cluster2.push_back(clusters[clusterToSplit][i]);
                }
            }
            set<int> uniqueLabels;
            for(auto i:sublabels){
                uniqueLabels.insert(i);
            }
            if(uniqueLabels.size() < 2){
                std::cerr<<"K-Means refinement failed to find two distinct clusters."<<std::endl;
                clusters[clusterToSplit] = initialMask;
                clusters.push_back(notInitialMask);
            }
            else{
                clusters[clusterToSplit] = cluster1;
                clusters.push_back(cluster2);
            }

        } else {
            clusters[clusterToSplit] = initialMask;
            clusters.push_back(notInitialMask);
            
        }
    } else if (anchorType == MD::DivineAnchors::SplinterPair) {
        if(subdata.rows() < 2){
            return false;
        }

        Index splinterIdx = calculateOutlier(subdata, nAtoms, mt);
        Vec splinterPoint = subdata.row(splinterIdx);

        Index medoidIdx = calculateMedoid(subdata, nAtoms, mt);
        Vec medoidPoint = subdata.row(medoidIdx);

        vector<Index> splinterGroup = {subdataIndices[splinterIdx]};
        vector<Index> mainGroup;
        splinterGroup.reserve(subdata.rows() - 1);
        mainGroup.reserve(subdata.rows() - 1);

        for (Index i = 0; i < subdata.rows(); ++i) {
            if (i == splinterIdx){
                continue;
            }

            double dS = (subdata.row(i).transpose() - splinterPoint).square().sum() / nAtoms;
            double dM = (subdata.row(i).transpose() - medoidPoint).square().sum() / nAtoms;

            if (dS < dM) {
                splinterGroup.push_back(subdataIndices[i]);
            } else {
                mainGroup.push_back(subdataIndices[i]);
            }
        }
        if (refine) {
            Mat groupA = subdata(mainGroup, Eigen::placeholders::all);
            Mat groupB = subdata(splinterGroup, Eigen::placeholders::all);
            Index medoidA = splinterGroup.size() <= 2 ? 0 : calculateMedoid(groupA, nAtoms, mt);
            Index medoidB = groupB.size() <= 2 ? 0 : calculateMedoid(groupB, nAtoms, mt);
            Mat initiators = Mat::Zero(2, data.row(0).size());
            initiators.row(0) = groupA.row(medoidA);
            initiators.row(1) = groupB.row(medoidB);
            KmeansNANI kmeans(subdata, 2, mt, initiators, nAtoms);
            Veci sublabels = kmeans.getLabels();
            vector<Index> cluster1, cluster2;
            for (Index i = 0; i < sublabels.size(); ++i) {
                if (sublabels[i] == 0) {
                    cluster1.push_back(clusters[clusterToSplit][i]);
                } else {
                    cluster2.push_back(clusters[clusterToSplit][i]);
                }
            }
            set<int> uniqueLabels;
            for(auto i:sublabels){
                uniqueLabels.insert(i);
            }
            if(uniqueLabels.size() < 2){
                std::cerr<<"K-Means refinement failed to find two distinct clusters."<<std::endl;
                clusters[clusterToSplit] = cluster1;
                clusters.push_back(cluster2);
            }
            else{
                clusters[clusterToSplit] = cluster1;
                clusters.push_back(cluster2);
            }

        } else {
            clusters[clusterToSplit] = mainGroup;
            clusters.push_back(splinterGroup);   
        }
    }
    return true;
};
Divine::Divine(Mat data, MD::DivineSplit splitType, 
    MD::DivineAnchors anchorType, MD::KinitType kinit, 
    int end, int k, bool refine, int nAtoms, double threshold, int percentage): 
    data(data), splitType(splitType), anchorType(anchorType), kinit(kinit), 
    refine(refine), nAtoms(nAtoms), threshold(threshold), percentage(percentage), 
    mt(MD::Metric::MSD), end(end) {
    if (k == 0) {
        kClusters = data.rows();
    } else {
        kClusters = k;
    }

    vector<Index> initCluster;
    for(int i=0; i<data.rows(); i++){
        initCluster.emplace_back(i);
    }
    clusters.emplace_back(initCluster);
    divisiveAlgorithm();
    labels.assign(data.rows(), -1);
    createLabels(data.rows());
};

vector<vector<Index>> Divine::getClusters(){
    return clusters;
}
vector<int> Divine::getLabels(){
    return labels;
}
vector<pair<double, double>> Divine::getScores(){
    return scores;
}

pair<double, double> Divine::computeScores(Veci labels, Mat data){
    set<int> uniqueLabels;
    for(auto i:labels){
        uniqueLabels.insert(i);
    }

    if(uniqueLabels.size() <= 1 || uniqueLabels.size() >= data.rows()){
        return pair<double, double>(0,0);
    }

    double chScore = calinskiHarabaszScore(data, labels);
    double dbScore = daviesBouldinScore(data, labels);
    return pair<double, double>(chScore, dbScore);
}

void Divine::createLabels(int nTotal){
    for(int i=0; i<clusters.size(); i++){
        for(int j:clusters[i]){
            labels[j] = i;
        }
    }
}
