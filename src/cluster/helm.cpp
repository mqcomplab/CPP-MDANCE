#include <stdexcept>

#include "helm.h"
#include "../tools/bts.h"
#include "../tools/scores.h"

Mat Helm::makeDataByRow(Vec a, Vec b){
    //a and b need to be same length
    //a will be csum, and b will be sqsum
    Mat data(2, a.size());
    data.row(0) = a;
    data.row(1) = b;

    return data;
}

vector<HCTree> Helm::trimClusters(){
    /*
        Trims the intial clusters based on the trimVal or trimK

        returns
        ------------
        map of clusters
    */
    vector<pair<double, int>> clusterMsds;
    int i=0;
    for(int i=0; i<clusterTree.size(); i++){
        Vec cSum = clusterTree.at(i).getRootCSum();
        Vec sqSum = clusterTree.at(i).getRootSQSum();
        int Nik = clusterTree.at(i).getRootNObjects();

        if(Nik < minSamples){
            continue;
        }

        Mat data = makeDataByRow(cSum, sqSum);
        clusterMsds.emplace_back(pair<double, int>(extendedComparison(data, Nik, nAtoms, true, mt), i));
    }

    sort(clusterMsds.begin(), clusterMsds.end());

    //trim the clusters based on the trim_k or trim_val
    vector<HCTree> newClusterTree;
    if(trimK){
        this->trimIncoming = clusterMsds.size()-trimK;
        if (trimK >= clusterMsds.size()-1){
            throw std::runtime_error("trimK is too large!");
        }
        else if(trimK >= clusterMsds.size()/2){
            std::cerr<<"trimK is more than 50% of the clusters. This may lead to poor clustering"<<std::endl;
        }
        for(int i=0; i<clusterMsds.size()-trimK; i++){
            int index = clusterMsds[i].second;
            if (index < 0 || index >= clusterTree.size()) {
                throw std::runtime_error("Index out of bounds when trimming clusters.");
            }
            newClusterTree.emplace_back(clusterTree[index]);
        }
    } 
    else if(trimVal){
        this->trimIncoming = 0;
        for(auto i:clusterMsds){
            if(i.first < trimVal){
                trimIncoming++;
            }
        }

        for(auto i:clusterMsds){
            if(i.first < trimVal){
                newClusterTree.emplace_back(clusterTree[i.second]);
            }
        }
    }

    return newClusterTree;
}

vector<HCTree> Helm::genNewClusters(int ZIdx){
    /*
        Generates new cluster by merging two most similar clusters.

        Parameters
        ------------
        previous_clusters: contains info about clusters in kth iteration
    */
    int previousSize = clusterTree.size();
    vector<HCTree> previousClusters = clusterTree;
    if(clusterDists.rows()==0 && clusterDists.cols()==0){
        genClusterDists(previousClusters);
    }
    else{
        Vec distsToNewCluster = Vec::Constant(previousSize-1, INFINITY);
        for(int i=0; i < previousSize - 1; i++){
            float helmSim = calcHelmSim(previousClusters[i], previousClusters[previousSize - 1]); // assumes new cluster is at the end of clusterTree!
            distsToNewCluster[i] = helmSim;
        }

        //Add new cluster to distance matrix
        //np hstack 
        Mat temp1(clusterDists.rows(), clusterDists.cols() + 1);
        temp1.leftCols(clusterDists.cols()) = clusterDists;
        temp1.col(clusterDists.cols()) = distsToNewCluster;
        
        //np vstack
        Mat temp2(temp1.rows() + 1, temp1.cols());
        temp2.topRows(temp1.rows()) = temp1;
        temp2.row(temp1.rows()).setConstant(INFINITY);
        clusterDists = temp2;
    }   

    //Find the two most similar clusters
    Index minRow, minCol;       //minRow and minCol refer to two different clusters
    
    float mergeDist = clusterDists.minCoeff(&minRow, &minCol);
    // add check for minRow and minCol being same. This can cause UB or crash. 
    if (minRow == minCol){
        throw std::runtime_error("Got same cluster idx for the two most similar clusters.");
    }
    //Merge the two most similar clusters
    Vec cSum, sqSum;
    if(alignMeth == MD::AlignMethod::Kron){
        //will add later
        // todo add kron alignment
        ;
    }
    else{
        cSum = previousClusters[minRow].getRootCSum() + previousClusters[minCol].getRootCSum();
        sqSum = previousClusters[minRow].getRootSQSum() + previousClusters[minCol].getRootSQSum();
    }

    Vec cSumik = cSum;
    Vec sqSumik = sqSum;
    int Nik = previousClusters[minRow].getRootNObjects() + previousClusters[minCol].getRootNObjects();
    if(alignMeth != MD::AlignMethod::None){
        // todo add kron alignment
        //aligned combine clusters
        //continuation of kron
    }

    //Save the new clusters after mergin
    vector<HCTree> newClusters;
    newClusters.reserve(previousClusters.size()+1);
    for(int i=0; i<previousClusters.size(); i++){
        if(i==minRow || i==minCol){;}
        else{
            newClusters.emplace_back(previousClusters[i]);
        }
    }
    // int nIndMinRow = previousClusters[minRow].getIndices().size();
    // int nIndMinCol = previousClusters[minCol].getIndices().size();
    // Veci indicesik(nIndMinRow + nIndMinCol);
    // indicesik(Eigen::seq(0, nIndMinRow-1)) = previousClusters[minRow].getIndices();
    // indicesik(Eigen::seq(nIndMinRow, indicesik.size()-1)) = previousClusters[minCol].getIndices();

    //Two different ways of saving the new cluster
    if(alignMeth != MD::AlignMethod::None){
        // todo add kron alignment
        //newClusters.push_back(Cluster(indicesik, cSumik, sqSumik, Nik, aligned));
        //kron
        ;
    }
    else{
        // update zMatrix with new merged cluster info
        int idxMinRow = previousClusters[minRow].getRootIdx();
        int idxMinCol = previousClusters[minCol].getRootIdx();
        int nClustsMerged = previousClusters[minRow].getRootIndices().size() + previousClusters[minCol].getRootIndices().size();
        updateZMatrix(idxMinRow, idxMinCol, nClustsMerged);

        // merge clusters and add to newClusters
        previousClusters[minRow].combineTrees(previousClusters[minCol], ZIdx);
        newClusters.emplace_back(previousClusters[minRow]);
    }

    //Remove distances of merged clusters
    Veci clustersToKeep(clusterDists.rows()-2);
    int ind=0;
    for(int i=0; i<clusterDists.rows(); i++){
        if(i!= minRow && i!=minCol){
            clustersToKeep[ind] = i;
            ind++;
        }
        
    }
    Mat clusterDists_temp1 = clusterDists(Eigen::placeholders::all, clustersToKeep);
    Mat clusterDists_temp2 = clusterDists_temp1(clustersToKeep, Eigen::placeholders::all);
    clusterDists = clusterDists_temp2;

    if(eps==-1 || mergeDist < eps){
        return newClusters;
    }
    return vector<HCTree>();
}

void Helm::updateZMatrix(int idxA, int idxB, int mergedClusts){
    if (zMatrix.rows() == 0 && zMatrix.cols() == 0){
        zMatrix = Mat::Zero(1,4);
        zMatrix(0, 0) = idxA;
        zMatrix(0, 1) = idxB;
        zMatrix(0, 2) = 1;
        zMatrix(0, 3) = mergedClusts;
    }
    else{
        Vec zMatrixRow(4);
        int distZMatrix = zMatrix.rows() + 1;
        zMatrixRow << idxA, idxB, distZMatrix, mergedClusts;
        zMatrix.conservativeResize(zMatrix.rows()+1, zMatrix.cols());
        zMatrix.row(zMatrix.rows()-1) = zMatrixRow;
    }
}

float Helm::calcHelmSim(HCTree& firstTree, HCTree& secondTree){
    /*
        Calculates the similarity between two clusters. These are the roots of the two trees that are passed. 

        Parameters
        ------------
        firstTree: HCTree object containing info about first cluster at root
        secondTree: HCTree object containing info about second cluster at root
    */

    Vec cSumA = firstTree.getRootCSum();
    Vec cSumB = secondTree.getRootCSum();
    Vec sqSumA = firstTree.getRootSQSum();
    Vec sqSumB = secondTree.getRootSQSum();
    int nA = firstTree.getRootNObjects();
    int nB = secondTree.getRootNObjects();

    Mat dataA = makeDataByRow(cSumA, sqSumA);
    double simA = extendedComparison(dataA, nA, nAtoms, true, mt);
    Mat dataB = makeDataByRow(cSumB, sqSumB);
    double simB = extendedComparison(dataB, nB, nAtoms, true, mt);

    Vec cSum;
    Vec sqSum;
    if(alignMeth == MD::AlignMethod::Kron){
        // todo add kron alignment
        //when alignMeth is not None, previousCluster[i][3]=input_clusters
        ;

    } else{
        cSum = cSumA + cSumB;
        sqSum = sqSumA + sqSumB;
    }

    int n = nA + nB;
    Mat data = makeDataByRow(cSum, sqSum);
    double sim = extendedComparison(data, n, nAtoms, true, mt);

    //Different merging schemes for determining which clusters to merge
    float helmSim;
    if (mergeScheme == MD::MergeScheme::Intra){
        helmSim = sim;
    }
    else if(mergeScheme == MD::MergeScheme::Inter){
        helmSim = ((sim*pow(n,2)) - (simA*pow(nA,2)) - (simB*pow(nB,2)))/(nA*nB);
    }
    else if(mergeScheme == MD::MergeScheme::Half){
        helmSim = sim - ((simA + simB)/2);
    }

    return helmSim;
}

    //finished
void Helm::genClusterDists(vector<HCTree>& previousClusters){
    /*
        Generates pairwise similairty matrix for initial clusters

        Parameters
        -------------
        previousClusters: contains info about clusters in kth iteration

        Returns
        -------------
        pairwise similarity matrix (void)
    */

    clusterDists = Mat::Zero(previousClusters.size(), previousClusters.size()) + INFINITY;
    for(int i=0; i<previousClusters.size(); i++){
        for(int j=i+1; j<previousClusters.size(); j++){
            float helmSim = calcHelmSim(previousClusters[i], previousClusters[j]);
            clusterDists(i, j) = helmSim;
        }
    }
}

//finished
//todo: update to new strucure

// Mat Helm::initialPairwiseMatrix(vector<Cluster>& previousClusters){
//     /*
//         Generates pairwise similarity matrix for the initial clusters
        
//         Parameters
//         -------------
//         previousClusters: contains the info about clusters in kth iteration

//         Results
//         --------------
//         returns pairwise similarity matrix
//     */

//     //Optimally trim the initial clusters step
//     if(trimStart){
//         clusterMap = trimClusters();
//     }

//     //extracting all keys of clusterMap
//     vector<int> keys;
//     for(auto pair:clusterMap){
//         keys.emplace_back(pair.first);
//     }
//     sort(keys.begin(), keys.end());
//     int n = keys[0];

//     previousClusters = clusterMap[n];

//     Mat distances(n,n);
//     for(int i=0; i<n; i++){
//         for(int j=0; j<n; j++){
//             if(i==j){
//                 distances(i,j) = 0;
//             }
//             else{
//                 float helmSim = calcHelmSim(previousClusters[i], previousClusters[j]);
//                 distances(i,j) = helmSim;
//             }
//         }
//     }

//     distances = refineDisMatrix(distances);
//     return distances;
// }

    //todo gen_link_matrix()
    //i will assume that a link matrix is ArrayXXd (or eq. Mat)

//finished
// todo: convert to new HCTree structure
// map<int, vector<Cluster>> Helm::linkMatrixToClusterMap(){
//     vector<int> naniSizes;
//     for(auto it=clusterMap.begin(); it!=clusterMap.end(); it++){
//         for(auto clust:it->second){
//             naniSizes.emplace_back(clust.getN());
//         }
//     }

//     //cluster IDs
//     int k = linkMatrix.size() + 1;
//     vector<vector<int>> vecCluster;
//     for(int i=0; i<k; i++){
//         vecCluster.emplace_back(vector<int>{i});
//     }

//     map<int, vector<vector<int>>> clusterInds;
//     vector<vector<int>> copyVecCluster = vecCluster;
//     clusterInds[k] = copyVecCluster;

//     for(int i=0; i<linkMatrix.size(); i++){
//         vector<vector<int>> levelClusters;

//         auto row=linkMatrix.row(i);            
//         int c1 = row[0];
//         int c2 = row[1];

//         vector<int> newVec(vecCluster[c1]);
//         for(int i:vecCluster[c2]){
//             newVec.emplace_back(i);
//         }
//         vecCluster.emplace_back(newVec);
//         int newK = k-i-1;

//         for(auto clust:clusterInds[k-i]){
//             //clust here is vector<int>

//             if(clust == vecCluster[c1]){;}
//             else if(clust == vecCluster[c2]){;}
//             else{
//                 levelClusters.emplace_back(clust);
//             } 
//         }
//         levelClusters.emplace_back(vecCluster.back());
//         clusterInds[newK] = levelClusters;
//     }

//     map<int, vector<Cluster>> clusters;
//     for(auto it=clusterInds.begin(); it!=clusterInds.end(); it++){
//         clusters[it->first] = vector<Cluster>();
//         for(auto clust:it->second){
//             int n_mols=0;
//             for(int i:clust){
//                 n_mols+=naniSizes[i];
//             }

//             //convert clust to Veci type
//             Veci clust_eigen(clust.size());
//             for(int i=0; i<clust.size(); i++){
//                 clust_eigen(i) = clust[i];
//             }
//             clusters[it->first].emplace_back(Cluster(clust_eigen, Vec::Zero(clust.size()),Vec::Zero(clust.size()), n_mols));
//         }
//     }
//     return clusters;
// }

Helm::Helm(vector<HCTree> clusterTree, int nAtoms, MD::Metric mt, 
        MD::MergeScheme mergeScheme, int nClusters, float eps, 
        bool trimStart, MD::AlignMethod alignMeth, 
        float minSamples, MD::Link link,
        float trimVal, float trimK,
        bool savePairwiseSum,
        string inputTop, string inputTraj){
    this->clusterTree = clusterTree;
    this->mt = mt;
    this->nAtoms = nAtoms;
    this->mergeScheme = mergeScheme;
    this->nClusters = nClusters;
    this->eps = eps;
    this->trimStart = trimStart;
    this->alignMeth = alignMeth;
    this->minSamples = minSamples;
    this->link = link;
    this->trimVal = trimVal;
    this->trimK = trimK;
    this->savePairwiseSum = savePairwiseSum;
    this->inputTop = inputTop;
    this->inputTraj = inputTraj;

    this->totalIncoming = clusterTree.size();;
    totalSum = 0;
    for(auto tree:clusterTree){
        this->totalSum += tree.getRootNObjects();
    }

    //check end conditions
    if((this->nClusters==0 && this->eps==-1) || 
        (this->nClusters > 0 && this->eps!=-1)){
            throw std::invalid_argument("You must provide either nClusters or eps, but not both.");
        }

    if(this->trimStart && !(this->trimVal || this->trimK)){
        throw std::invalid_argument("If trimStart is true, then either trimVal or trimK must be provided.");
    }
    if(this->trimVal && this->trimK){
        throw std::invalid_argument("You can only provided either trimVal or trimK, but not both.");
    }

    if(this->minSamples < 0){
        throw std::invalid_argument("minSamples must be greater than 0.");
    }
    else if(0 < this->minSamples && this->minSamples < 1){
        this->minSamples = int(this->minSamples * this->totalSum);
    }
    else if(this->minSamples >= 1){
        this->minSamples = int(this->minSamples);
    }

}
//finished
vector<HCTree> Helm::run(){
    /*
        Performs HELM clustering of initial clusters

        Returns
        -----------
        map of clusters ( map<int, vector<vector<Cluster>>>)
    */

    if(nClusters == 0){
        nClusters = 1;
    }
    if(link == MD::Link::Ward){
        // todo add ward linkage
        //gen_link_matrix();
        //return linkMatrixToClusterMap();
    }
    if(trimStart){
        clusterTree = trimClusters();
    }

    //perform clustering
    int n = clusterTree.size();
    int currentZInd = clusterTree.size();
    if (nClusters == n) {
        std::cerr << "Number of clusters is already equal to nClusters. No clustering will be performed." << std::endl;
        return clusterTree;
    }
    while(n>1){
        // vector<Cluster> previousClusters = clusterMap[n];
        vector<HCTree> newClusters = genNewClusters(currentZInd);
        if (!newClusters.empty()){
            clusterTree = newClusters;
        }
        currentZInd +=1; // increment Z index for new merged cluster
        //termination conditions
        if(n==(nClusters+1) || newClusters.empty()){
            break;
        }
        n-=1;
    }
    return clusterTree;
}

pair<double, double> Helm::computeScores(vector<HCTree> clusters, Mat data){
    /*
        Computes Calinksi-Harabasz and Davies-Bouldin scores of clusters
        using random labeling

        Returns
        -------------
        pair: first element is Calinkski, second element Davies-Bouldin
    */

    //vector<Veci> clusterIndices;
    vector<int> label;
    int count=0;

    for(auto c:clusters){
        //clusterIndices.emplace_back(c.getIndices());
        int Nik=c.getRootNObjects();
        label.insert(label.end(), Nik, count);
        count++;
    }
    
    set<int> temp;
    for(int i:label){
        temp.insert(i);
    }
    if(temp.size()==1){
        return pair<double, double>{-1,-1};
    }
    else{
        Veci labels = Eigen::Map<Veci>(label.data(), label.size());
        double chScore = calinskiHarabaszScore(data, labels);
        double dbScore = daviesBouldinScore(data, labels);

        return pair<double, double>{chScore, dbScore};
    }
}

Mat Helm::getZMatrix(){
    /* 
        Converts the cluster dictionary to a linkage matrix Z

        Returns
        -------------
        Z matrix (Mat)
    */
    if (zMatrix.rows() == 0 && zMatrix.cols() == 0){
        std::cerr<<"Z matrix is empty. No clusters were merged."<<std::endl;
    }
    return zMatrix;
}

