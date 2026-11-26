#include "divine.h"


//ready for testing
void Divine::divisiveAlgorithm() {
    /* 
        Main loop for recursively splitting clusters.
    */
    int minFrames = std::max(1, (int)std::floor(threshold * data.rows()));

    int maxIter=data.rows()+1;
    int counter=1;
    bool done=true;
    while (done) {
        MD::progress(kClusters > 0 ? std::min(1.0, (double)clusters.size() / kClusters) : 0.0, "DIVINE");
        //stopping conditions
        if(counter>maxIter)   break;
        counter++;
        if(end==0){
            if(clusters.size() >= kClusters)    break;
        }
        else if(end==1){
            //if all clusters are size of 1, then break
            bool flag=false;
            for(Index i=0; i<clusters.size(); i++){
                if(clusters[i].size()!=1)   flag=true;
            }
            if(!flag)   break;
        }

        vector<bool> failedSplits(clusters.size(), false);
        bool didSplit = false;
        while (!didSplit) {
            //determine which cluster to split
            Index clusterToSplit = selectClusterToSplit(failedSplits);
            if (clusterToSplit < 0) {
                // no more splits would yield valid subclusters
                done=false;
                break;
            } 
            //split the selected cluster into sub-clusters
            didSplit = splitCluster(clusterToSplit, minFrames);
            //update failedSplits
            failedSplits[clusterToSplit] = !didSplit;
        }
    }
};
Index Divine::selectClusterToSplit(vector<bool>& failedSplits) {
    /*
        This function selects the cluster with the highest score for a criteria.

        Parameters
        -------------
        failedSplits: indices of clusters previously deemed unsplittable.

        Returns
        -------------
        Index: index of the cluster to split, -1 if no suitable split found.
    */
    Index topCluster = -1;
    double bestScore = -1;
    // Each cluster's score is computed once: it changes only when its frames
    // do (splitCluster forgets the split one; a new one is new). Every
    // iteration rescored every cluster from a copy of its frames, O(N D) a
    // split and O(N^2 D) with end mode "points".
    scoreCache.resize(clusters.size(), -2);

    for (Index i = 0; i < (Index)clusters.size(); ++i) {
        if (failedSplits[i] || clusters[i].size() < 2) {
            continue;
        }
        if (scoreCache[i] != -2) {
            if (scoreCache[i] > bestScore) {
                bestScore = scoreCache[i];
                topCluster = i;
            }
            continue;
        }

        double score = -1;
        Mat subdata = data(clusters[i], Eigen::all);
        if (splitType == MD::DivineSplit::MSD) {
            score = extendedComparison(subdata, 0, nAtoms, false, mt);
        } else if (splitType == MD::DivineSplit::Radius) {
            Index medoidIdx = calculateMedoid(subdata, nAtoms, mt);
            Vec medoid = subdata.row(medoidIdx);
            Vec dists = (subdata.rowwise() - medoid.transpose()).square().rowwise().sum() / nAtoms;

            score = dists.maxCoeff();
        } else if (splitType == MD::DivineSplit::WeightedMSD) {
            // (scoreCache[i] is set below)
            score = clusters[i].size() * extendedComparison(subdata, 0, nAtoms, false, mt);  
        }

        scoreCache[i] = score;
        if (score > bestScore) {
            bestScore = score;
            topCluster = i;
        }
    }
    return topCluster;
};
bool Divine::splitCluster(Index clusterToSplit, int minFrames) {
    /*
        This functioni splits the specified cluster into two subclusters.

        Parameters
        -------------
        clusterToSplit: index of the cluster to split.
        
        Returns
        -------------
        bool: whether or not split was successful or not
    */
    if (clusters[clusterToSplit].size() < 2) {
        return false;
    }
    // (the split cluster's score is computed again at the next selection:
    // its frames change, or it is marked failed)
    scoreCache.resize(clusters.size(), -2);
    scoreCache[clusterToSplit] = -2;

    //subset the indices of the cluster to be split
    Veci subdataIndices;
    subdataIndices.resize(clusters[clusterToSplit].size());
    for(int i=0; i<clusters[clusterToSplit].size(); i++){
        subdataIndices[i]=clusters[clusterToSplit][i];
    }
    Mat subdata = data(subdataIndices, Eigen::all);
    
    if (anchorType == MD::DivineAnchors::NANI) {
        // calculate the minimum percentage so that at least two points are selected. 
        int minPercentage = std::ceil(2.0 / subdata.rows() * 100); 
        ArrayXi sublabels;
        // check if minPercentage exceeds 100%. This shouldn't happen, but just in case, we set it to 100% if it does.
        if (minPercentage > 100){
            minPercentage = 100;
        }
        // if the user-specified percentage is less than the minimum percentage, we adjust it to the minimum percentage.
        if (percentage < minPercentage) {
            // too low a percentage to select two points: the least that does
             KmeansNANI kmeans(subdata, 2, mt, kinit, nAtoms, minPercentage);
            sublabels = kmeans.getLabels();
        }
       else{
            KmeansNANI kmeans(subdata, 2, mt, kinit, nAtoms, percentage);
            sublabels = kmeans.getLabels();
       }
         
        vector<Index> cluster1, cluster2;
        for (Index i = 0; i < sublabels.size(); ++i) {
            if (sublabels[i] == 0) {
                cluster1.push_back(clusters[clusterToSplit][i]);
            } else {
                cluster2.push_back(clusters[clusterToSplit][i]);
            }
        }
        // check if min frame condition is satisfied
        if(cluster1.size()<minFrames || cluster2.size()<minFrames){
            return false;
        }

        //merge the two newly split clusters to clusters vector
        clusters[clusterToSplit] = cluster1;
        clusters.push_back(cluster2);
    } else if (anchorType == MD::DivineAnchors::OutlierPair) {
        Index outlierIdx = calculateOutlier(subdata, nAtoms, mt);
        Vec anchorA = subdata.row(outlierIdx);
        //calculate distances between subdata rows and anchorA row
        Vec dists = (subdata.rowwise() - anchorA.transpose()).square().rowwise().sum() / nAtoms;
        Index idxFurthest;
        
        //find row that is the farthest away from anchorA row based on dists
        dists.maxCoeff(&idxFurthest);
        Vec anchorB = subdata.row(idxFurthest);

        //calculate distances between subdata and each anchor row
        Vec dA = (subdata.rowwise() - anchorA.transpose()).square().rowwise().sum() / nAtoms;
        Vec dB = (subdata.rowwise() - anchorB.transpose()).square().rowwise().sum() / nAtoms;

        vector<Index> initialMask;
        vector<Index> notInitialMask;
        initialMask.reserve(subdata.rows());
        notInitialMask.reserve(subdata.rows());
        for (Index i = 0; i < dA.size(); ++i) {
            if (dA[i] < dB[i]) {
                initialMask.push_back(i);
            } else {
                notInitialMask.push_back(i);
            }
        }

        if (refine) {
            // Identical frames (or NaN) put every frame on one side: the
            // refinement then read the first row of an empty group (a crash).
            if (initialMask.empty() || notInitialMask.empty()) {
                return false;
            }
            Mat groupA = subdata(initialMask, Eigen::all);
            Mat groupB = subdata(notInitialMask, Eigen::all);

            // (rows, not coefficients; the medoid of one or two frames is the
            // first, which calculateMedoid also returned for two)
            Index medoidA = groupA.rows() <= 2 ? 0 : calculateMedoid(groupA, nAtoms, mt);
            Index medoidB = groupB.rows() <= 2 ? 0 : calculateMedoid(groupB, nAtoms, mt);

            Mat initiators = Mat::Zero(2, data.row(0).size());
            initiators.row(0) = groupA.row(medoidA);
            initiators.row(1) = groupB.row(medoidB);

            KmeansNANI kmeans(subdata, 2, mt, initiators, nAtoms);
            Veci sublabels = kmeans.getLabels();
            
            //find number of unique labels
            set<int> uniqueLabels;
            for(auto i:sublabels){
                uniqueLabels.insert(i);
            }
            if(uniqueLabels.size() < 2){
                // the refinement found one cluster: the split before it is kept
                vector<Index> cluster1, cluster2;
                for (auto i : initialMask){
                    cluster1.push_back(subdataIndices[i]);
                }
                for (auto i : notInitialMask){
                    cluster2.push_back(subdataIndices[i]);
                }
                // check if min frame condition is satisfied
                if(cluster1.size()<minFrames || cluster2.size()<minFrames){
                    return false;
                }
                // add new clusters to the clusters vector
                clusters[clusterToSplit] = cluster1;
                clusters.push_back(cluster2);
            }
            else{
                vector<Index> cluster1, cluster2;
                for (Index i = 0; i < sublabels.size(); ++i) {
                    if (sublabels[i] == 0) {
                        cluster1.push_back(clusters[clusterToSplit][i]);
                    } else {
                        cluster2.push_back(clusters[clusterToSplit][i]);
                    }
                }
                // check if min frame condition is satisfied
                if(cluster1.size()<minFrames || cluster2.size()<minFrames){
                    return false;
                }
                // add new clusters to the clusters vector
                clusters[clusterToSplit] = cluster1;
                clusters.push_back(cluster2);
            }

        } else {
            if(initialMask.size()<minFrames || notInitialMask.size()<minFrames){
                return false;
            }
            
            vector<Index> cluster1, cluster2;
            for (auto i : initialMask){
                cluster1.push_back(subdataIndices[i]);
            }
            for (auto i : notInitialMask){
                cluster2.push_back(subdataIndices[i]);
            }
            clusters[clusterToSplit] = cluster1;
            clusters.push_back(cluster2);
            
        }
    } else if (anchorType == MD::DivineAnchors::SplinterPair) {
        if(subdata.rows() < 2){
            return false;
        }

        Index splinterIdx = calculateOutlier(subdata, nAtoms, mt);
        Vec splinterPoint = subdata.row(splinterIdx);

        Index medoidIdx = calculateMedoid(subdata, nAtoms, mt);
        Vec medoidPoint = subdata.row(medoidIdx);

        //split cluster indices into splinterGroup and mainGroup
        vector<Index> splinterGroup = {splinterIdx};
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
                splinterGroup.push_back(i);
            } else {
                mainGroup.push_back(i);
            }
        }
        if (refine) {
            if (mainGroup.empty() || splinterGroup.empty()) {
                return false;
            }
            Mat groupA = subdata(mainGroup, Eigen::all);
            Mat groupB = subdata(splinterGroup, Eigen::all);

            Index medoidA = groupA.rows() <= 2 ? 0 : calculateMedoid(groupA, nAtoms, mt);
            Index medoidB = groupB.rows() <= 2 ? 0 : calculateMedoid(groupB, nAtoms, mt);

            Mat initiators = Mat::Zero(2, data.row(0).size());
            initiators.row(0) = groupA.row(medoidA);
            initiators.row(1) = groupB.row(medoidB);

            KmeansNANI kmeans(subdata, 2, mt, initiators, nAtoms);
            Veci sublabels = kmeans.getLabels();

            //find number of unique labels
            set<int> uniqueLabels;
            for(auto i:sublabels){
                uniqueLabels.insert(i);
            }
            vector<Index> cluster1, cluster2;
            if(uniqueLabels.size() < 2){
                // use mainGroup and splinterGroup before refinement as the new clusters
                
                if(mainGroup.size()<minFrames || splinterGroup.size()<minFrames){
                    return false;
                }
                for (Index i : mainGroup){
                    cluster1.push_back(subdataIndices[i]);
                }
                for (Index i : splinterGroup){
                    cluster2.push_back(subdataIndices[i]);
                }
                clusters[clusterToSplit] = cluster1;
                clusters.push_back(cluster2);
            }
            else{
                for (Index i = 0; i < sublabels.size(); ++i) {
                    if (sublabels[i] == 0) {
                        cluster1.push_back(clusters[clusterToSplit][i]);
                    } else {
                        cluster2.push_back(clusters[clusterToSplit][i]);
                    }
                }
                // check if min frame condition is satisfied
                if(cluster1.size()<minFrames || cluster2.size()<minFrames){
                    return false;
                }
                clusters[clusterToSplit] = cluster1;
                clusters.push_back(cluster2);
            }
            

        } else {
            vector<Index> cluster1, cluster2;
            // check if min frame condition is satisfied
            if(mainGroup.size()<minFrames || splinterGroup.size()<minFrames){
                return false;
            }
            for (Index i : mainGroup){
                cluster1.push_back(subdataIndices[i]);
            }
            for (Index i : splinterGroup){
                cluster2.push_back(subdataIndices[i]);
            }
            // add cluster1 and cluster2 to the clusters vector
            clusters[clusterToSplit] = cluster1;
            clusters.push_back(cluster2);   
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
    //initialize clusters
    vector<Index> initCluster;
    for(int i=0; i<data.rows(); i++){
        initCluster.emplace_back(i);
    }
    if(initCluster.size()){
        clusters.emplace_back(initCluster);
    }
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
