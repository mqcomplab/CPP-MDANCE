#include "divine.h"
#include "pybind.h"
#include <fstream>
#include <string>

//ready for testing
void Divine::divisiveAlgorithm() {
    /* 
        Main loop for recursively splitting clusters.
    */
    int minFrames = std::max(1, (int)round(threshold * data.rows()));

    int maxIter=data.rows()+1;
    int counter=1;
    bool done=true;
    while (done) {
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
                std::cerr<<"No more cluster splits possible that would yield valid subclusters."<<std::endl;
                done=false;
                break;
            } 
            //split the selected cluster into sub-clusters
            didSplit = splitCluster(clusterToSplit, minFrames, counter);
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
bool Divine::splitCluster(Index clusterToSplit, int minFrames, int counter) {
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
        std::cerr<<"Cannot split a cluster with less than 2 points";
        return false;
    }

    //subset the indices of the cluster to be split
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

        //merge the two newly split clusters to clusters vector
        clusters[clusterToSplit] = cluster1;
        clusters.push_back(cluster2);
        
        if(clusters.size()>=6001){
            std::cout<<clusters.size()<<std::endl;
        }
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

            //KmeansNANI kmeans(subdata, 2, mt, initiators, nAtoms);
            //Veci sublabels = kmeans.getLabels();
            Veci sublabels=wrapperKmeans(subdata, initiators);

            vector<Index> cluster1, cluster2;
            for (Index i = 0; i < sublabels.size(); ++i) {
                if (sublabels[i] == 0) {
                    cluster1.push_back(clusters[clusterToSplit][i]);
                } else {
                    cluster2.push_back(clusters[clusterToSplit][i]);
                }
            }

            //find number of unique labels
            set<int> uniqueLabels;
            for(auto i:sublabels){
                uniqueLabels.insert(i);
            }
            if(initialMask.size()<minFrames || notInitialMask.size()<minFrames){
                return false;
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
            if(initialMask.size()<minFrames || notInitialMask.size()<minFrames){
                return false;
            }

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

        //split cluster indices into splinterGroup and mainGroup
        //vector<Index> splinterGroup = {subdataIndices[splinterIdx]};
        vector<Index> splinterGroup = {splinterIdx};
        vector<Index> mainGroup;
        splinterGroup.reserve(subdata.rows() - 1);
        mainGroup.reserve(subdata.rows() - 1);

        for (Index i = 0; i < subdata.rows(); ++i) {
            if (i == splinterIdx){
            //if (subdataIndices[i]==splinterIdx){
                continue;
            }
            double dS = (subdata.row(i).transpose() - splinterPoint).square().sum() / nAtoms;
            double dM = (subdata.row(i).transpose() - medoidPoint).square().sum() / nAtoms;

            if (dS < dM) {
                //splinterGroup.push_back(subdataIndices[i]);
                 splinterGroup.push_back(i);
            } else {
                //mainGroup.push_back(subdataIndices[i]);
                mainGroup.push_back(i);
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

            //KmeansNANI kmeans(subdata, 2, mt, initiators, nAtoms);
            //Veci sublabels = kmeans.getLabels();

            Veci sublabels=wrapperKmeans(subdata, initiators);

            vector<Index> cluster1, cluster2;
            for (Index i = 0; i < sublabels.size(); ++i) {
                if (sublabels[i] == 0) {
                    cluster1.push_back(subdataIndices[i]);
                } else {
                    cluster2.push_back(subdataIndices[i]);
                }
            }
            
            //find number of unique labels
            set<int> uniqueLabels;
            for(auto i:sublabels){
                uniqueLabels.insert(i);
            }

            if(uniqueLabels.size() < 2){
                std::cerr<<"K-Means refinement failed to find two distinct clusters."<<std::endl;
                clusters[clusterToSplit] = mainGroup;
                clusters.push_back(splinterGroup);
            }
            else{
                clusters[clusterToSplit] = cluster1;
                clusters.push_back(cluster2);
            }

            //output cluster indices
            std::string name="cluster1-" + std::to_string(counter) + ".csv";
            std::ofstream outFile(name);
            for (const auto& element:cluster1){
                outFile<<element<<"\n";
            }

            std::string name2="cluster2-" + std::to_string(counter) + ".csv";
            std::ofstream outFile2(name2);
            for (const auto& element:cluster2){
                outFile2<<element<<"\n";
            }

            if(mainGroup.size()<minFrames || splinterGroup.size()<minFrames){
                return false;
            }

        } else {
            if(mainGroup.size()<minFrames || splinterGroup.size()<minFrames){
                return false;
            }
            
            clusters[clusterToSplit]={};
            for(auto i: mainGroup){
                clusters[clusterToSplit].push_back(subdataIndices[i]);
            }
            clusters.push_back({});
            for(auto i: splinterGroup){
                clusters.back().push_back(subdataIndices[i]);
            }
            //clusters[clusterToSplit] = mainGroup;
            //clusters.push_back(splinterGroup);   
            std::cout<<"cluster size "<<clusters.size()<<std::endl;
        }
    }
    return true;
};

/*
save python mdance labels results for each iteration and use it for C++ mdance
    see if it fixes stuff, if it does then kmeans is the issue
    if it not then c++ divine is the issue
pdb and gdb: compare kmeans labels**
   
compare divine iteration between python and c++

error: add termination condition if divine cannot find two split clusters then it should stop
    
*/

Veci Divine::wrapperKmeans(Mat X, Mat initiators) {
    return runKmeans(X, initiators);
}

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
