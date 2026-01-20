#include <stdexcept>

#include "../tools/bts.h"
#include "../tools/cluster.h"
#include "../tools/scores.h"
#include "../tools/types.h"

class Helm{
    map<int, vector<Cluster>> clusterMap;
    int nAtoms;
    int nClusters;
    float eps;          //if None, then eps=-1
    bool trimStart;
    float trimVal;
    float trimK;
    float minSamples;
    int trimIncoming;
    MD::Metric mt;
    MD::AlignMethod alignMeth;
    MD::MergeScheme mergeScheme;
    MD::Link link;
    Mat clusterDists;
    Mat linkMatrix;
    int totalIncoming;
    bool savePairwiseSum;
    string inputTop;
    string inputTraj;
    int totalSum;

    Mat makeDataByRow(Vec a, Vec b){
        //a and b need to be same length
        //a will be csum, and b will be sqsum
        Mat data(2, a.size());
        data.row(0) = a;
        data.row(1) = b;

        return data;
    }

    map<int, vector<Cluster>> trimClusters(){
        /*
            Trims the intial clusters based on the trimVal or trimK

            returns
            ------------
            map of clusters
        */
        vector<pair<double, int>> clusterMsds;
        int i=0;
        for(int i=0; i<clusterMap[totalIncoming].size(); i++){
            Vec cSum = clusterMap[totalIncoming].at(i).getCsum();
            Vec sqSum = clusterMap[totalIncoming].at(i).getSQsum();
            int Nik = clusterMap[totalIncoming].at(i).getN();

            if(Nik < minSamples){
                continue;
            }

            Mat data = makeDataByRow(cSum, sqSum);
            clusterMsds.emplace_back(pair<double, int>(extendedComparison(data, Nik, nAtoms, true, mt), i));
        }

        sort(clusterMsds.begin(), clusterMsds.end());

        //trim the clusters based on the trim_k or trim_val
        map<int, vector<Cluster>> newClusterMap;
        if(trimK){
            this->trimIncoming = clusterMsds.size()-trimK;
            newClusterMap[trimIncoming] = vector<Cluster>();
            if (trimK >= clusterMsds.size()-1){
                throw std::runtime_error("trimK is too large!");
            }
            else if(trimK >= clusterMsds.size()/2){
                std::cerr<<"trimK is more than 50% of the clusters. This may lead to poor clustering"<<std::endl;
            }
            for(int i=0; i<clusterMsds.size()-trimK; i++){
                int index = clusterMsds[i].second;
                if (index < 0 || index >= clusterMap[totalIncoming].size()) {
                    throw std::runtime_error("Index out of bounds when trimming clusters.");
                }
                newClusterMap[trimIncoming].emplace_back(clusterMap[totalIncoming][index]);
            }
        } 
        else if(trimVal){
            this->trimIncoming = 0;
            for(auto i:clusterMsds){
                if(i.first < trimVal){
                    trimIncoming++;
                }
            }

            newClusterMap.emplace(trimIncoming, vector<Cluster>());
            for(auto i:clusterMsds){
                if(i.first < trimVal){
                    newClusterMap[trimIncoming].emplace_back(clusterMap[totalIncoming][i.second]);
                }
            }
        }

        return newClusterMap;
    }

    vector<Cluster> genNewClusters(vector<Cluster>& previousClusters){
        /*
            Generates new cluster by merging two most similar clusters.

            Parameters
            ------------
            previous_clusters: contains info about clusters in kth iteration
        */

        if(clusterDists.rows()==0 && clusterDists.cols()==0){
            genClusterDists(previousClusters);
        }
        else{
            Vec distsToNewCluster = Vec::Constant(previousClusters.size()-1, INFINITY);
            for(int i=0; i<previousClusters.size()-1; i++){
                float helmSim = calc(previousClusters, i, previousClusters.size()-1);
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
            cSum = previousClusters[minRow].getCsum() + previousClusters[minCol].getCsum();
            sqSum = previousClusters[minRow].getSQsum() + previousClusters[minCol].getSQsum();
        }

        Vec cSumik = cSum;
        Vec sqSumik = sqSum;
        int Nik = previousClusters[minRow].getN() + previousClusters[minCol].getN();
        if(alignMeth != MD::AlignMethod::None){
            // todo add kron alignment
            //aligned combine clusters
            //continuation of kron
        }

        //Save the new clusters after mergin
        vector<Cluster> newClusters;
        newClusters.reserve(previousClusters.size()+1);
        for(int i=0; i<previousClusters.size(); i++){
            if(i==minRow || i==minCol){;}
            else{
                newClusters.emplace_back(previousClusters[i]);
            }
        }
        int nIndMinRow = previousClusters[minRow].getIndices().size();
        int nIndMinCol = previousClusters[minCol].getIndices().size();
        Veci indicesik(nIndMinRow + nIndMinCol);
        indicesik(Eigen::seq(0, nIndMinRow-1)) = previousClusters[minRow].getIndices();
        indicesik(Eigen::seq(nIndMinRow, indicesik.size()-1)) = previousClusters[minCol].getIndices();

        //Two different ways of saving the new cluster
        if(alignMeth != MD::AlignMethod::None){
            // todo add kron alignment
            //newClusters.push_back(Cluster(indicesik, cSumik, sqSumik, Nik, aligned));
            //kron
            ;
        }
        else{
            newClusters.emplace_back(Cluster(indicesik, cSumik, sqSumik, Nik));
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
        return vector<Cluster>();
    }

    float calc(vector<Cluster>& previousClusters, int i, int j){
        /*
            Calculates the similarity between two clusters

            Parameters
            ------------
            previousClusters: contains info about clusters in kth iteration
            i: index of first cluster
            j: index of second cluster
        */

        Vec cSumA = previousClusters[i].getCsum();
        Vec cSumB = previousClusters[j].getCsum();
        Vec sqSumA = previousClusters[i].getSQsum();
        Vec sqSumB = previousClusters[j].getSQsum();
        int nA = previousClusters[i].getN();
        int nB = previousClusters[j].getN();

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

        int n = previousClusters[i].getN() + nB;
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
    void genClusterDists(vector<Cluster>& previousClusters){
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
                float helmSim = calc(previousClusters, i, j);
                clusterDists(i, j) = helmSim;
            }
        }
    }

    //finished
    Mat initialPairwiseMatrix(vector<Cluster>& previousClusters){
        /*
            Generates pairwise similarity matrix for the initial clusters
            
            Parameters
            -------------
            previousClusters: contains the info about clusters in kth iteration

            Results
            --------------
            returns pairwise similarity matrix
        */

        //Optimally trim the initial clusters step
        if(trimStart){
            clusterMap = trimClusters();
        }

        //extracting all keys of clusterMap
        vector<int> keys;
        for(auto pair:clusterMap){
            keys.emplace_back(pair.first);
        }
        sort(keys.begin(), keys.end());
        int n = keys[0];

        previousClusters = clusterMap[n];

        Mat distances(n,n);
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                if(i==j){
                    distances(i,j) = 0;
                }
                else{
                    float helmSim = calc(previousClusters, i, j);
                    distances(i,j) = helmSim;
                }
            }
        }

        distances = refineDisMatrix(distances);
        return distances;
    }

    //todo gen_link_matrix()
    //i will assume that a link matrix is ArrayXXd (or eq. Mat)

    //finished
    map<int, vector<Cluster>> linkMatrixToClusterMap(){
        vector<int> naniSizes;
        for(auto it=clusterMap.begin(); it!=clusterMap.end(); it++){
            for(auto clust:it->second){
                naniSizes.emplace_back(clust.getN());
            }
        }

        //cluster IDs
        int k = linkMatrix.size() + 1;
        vector<vector<int>> vecCluster;
        for(int i=0; i<k; i++){
            vecCluster.emplace_back(vector<int>{i});
        }

        map<int, vector<vector<int>>> clusterInds;
        vector<vector<int>> copyVecCluster = vecCluster;
        clusterInds[k] = copyVecCluster;

        for(int i=0; i<linkMatrix.size(); i++){
            vector<vector<int>> levelClusters;

            auto row=linkMatrix.row(i);            
            int c1 = row[0];
            int c2 = row[1];

            vector<int> newVec(vecCluster[c1]);
            for(int i:vecCluster[c2]){
                newVec.emplace_back(i);
            }
            vecCluster.emplace_back(newVec);
            int newK = k-i-1;

            for(auto clust:clusterInds[k-i]){
                //clust here is vector<int>

                if(clust == vecCluster[c1]){;}
                else if(clust == vecCluster[c2]){;}
                else{
                    levelClusters.emplace_back(clust);
                } 
            }
            levelClusters.emplace_back(vecCluster.back());
            clusterInds[newK] = levelClusters;
        }

        map<int, vector<Cluster>> clusters;
        for(auto it=clusterInds.begin(); it!=clusterInds.end(); it++){
            clusters[it->first] = vector<Cluster>();
            for(auto clust:it->second){
                int n_mols=0;
                for(int i:clust){
                    n_mols+=naniSizes[i];
                }

                //convert clust to Veci type
                Veci clust_eigen(clust.size());
                for(int i=0; i<clust.size(); i++){
                    clust_eigen(i) = clust[i];
                }
                clusters[it->first].emplace_back(Cluster(clust_eigen, Vec::Zero(clust.size()),Vec::Zero(clust.size()), n_mols));
            }
        }
        return clusters;
    }

    public:
        Helm(map<int, vector<Cluster>> clusterMap, int nAtoms, MD::Metric mt = MD::Metric::MSD, 
                MD::MergeScheme mergeScheme = MD::MergeScheme::Inter, int nClusters = 0, float eps = -1, 
                bool trimStart = false, MD::AlignMethod alignMeth = MD::AlignMethod::None, 
                float minSamples = 0.01, MD::Link link = MD::Link::None,
                float trimVal=0, float trimK=0,
                bool savePairwiseSum = false,
                string inputTop ="", string inputTraj =""){
            
            this->clusterMap = clusterMap;
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

            vector<int> keys;
            for(auto pair:clusterMap){
                keys.emplace_back(pair.first);
            }
            sort(keys.begin(), keys.end());
            this->totalIncoming = keys[0];
            totalSum = 0;
            for(auto clust:clusterMap[totalIncoming]){
                this->totalSum += clust.getN();
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
        map<int, vector<Cluster>> run(){
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
                clusterMap = trimClusters();
            }

            //perform clustering
            vector<int> keys;
            for(auto pair:clusterMap){
                keys.emplace_back(pair.first);
            }
            sort(keys.begin(), keys.end());
            int n = keys[0];

            while(n>1){
                vector<Cluster> previousClusters = clusterMap[n];
                vector<Cluster> newClusters = genNewClusters(previousClusters);
                clusterMap[n-1] = newClusters;

                //termination conditions
                if(n==(nClusters+1) || newClusters.empty()){
                    break;
                }
                n-=1;
            }
            return clusterMap;
        };

         pair<double, double> computeScores(vector<Cluster> clusters, Mat data){
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
                int Nik=c.getN();
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
        };

        Mat calculateZMatrix(map<int, vector<Cluster>> clusterMap){
            /* 
                Converts the cluster dictionary to a linkage matrix Z

                Returns
                -------------
                Z matrix (Mat)
            */
            vector<Veci> indices_clusters; // contains all unique clusters, and merged clusters
            vector<Cluster> initial_clusters = clusterMap.rbegin()->second;
            for (int i = 0; i < initial_clusters.size(); i++) {
                indices_clusters.push_back(initial_clusters[i].getIndices());
            }

            Mat zMatrix(clusterMap.size()-1, 4);
            int row = 0;
            
            // iterate through clusterMap in reverse order to get merging order
            // clusterMap is sorted by keys (No. clusters) in ascending order, thus we must traverse it in reverse
            for(auto it = clusterMap.rbegin(); it != clusterMap.rend(); ++it){
                vector<Cluster> current_clusters = it->second;
                for (auto cluster : current_clusters){
                    // convert vector to set so that it is easier to compare them
                    Veci inds = cluster.getIndices();
                    std::set inds_set(std::begin(inds), std::end(inds));
                    if (inds.size() != inds_set.size()){
                        throw std::runtime_error("indices of current cluster has duplicate members!");
                    }
                    // check if this cluster already exists in indices_clusters. This means it is not a new merged cluster
                    bool found = false;
                    for (int i = 0; i < indices_clusters.size(); i++){
                        Veci ind_prev_cluster = indices_clusters[i];
                        std::set<int> ind_prev_cluster_set(std::begin(ind_prev_cluster), std::end(ind_prev_cluster));

                        if (ind_prev_cluster.size() != ind_prev_cluster_set.size()){
                            throw std::runtime_error("indices of cluster has duplicate members!");
                        }
                        if (ind_prev_cluster_set == inds_set){
                            found = true;
                            break;
                        }
                    }
                    // extract data for Z matrix for the merged cluster
                    if (!found){
                        for (int i = 0; i < indices_clusters.size(); i++){
                            for (int j = i+1; j < indices_clusters.size(); j++){
                                Veci ind_i = indices_clusters[i];
                                Veci ind_j = indices_clusters[j];
                                set<int> set_i(std::begin(ind_i), std::end(ind_i));
                                set<int> set_j(std::begin(ind_j), std::end(ind_j));
                                set<int> set_union_ij;
                                std::set_union(set_i.begin(), set_i.end(), set_j.begin(), set_j.end(), std::inserter(set_union_ij, set_union_ij.begin()));
                                if (set_union_ij == inds_set){
                                    // found two clusters being merged
                                    zMatrix(row, 0) = i; // index of first cluster being merged
                                    zMatrix(row, 1) = j; // index of second cluster being merged
                                    zMatrix(row, 2) = row+1; // "distance" between clusters (using row number as proxy)
                                    zMatrix(row, 3) = inds_set.size(); // number of cluster members
                                    row +=1;
                                    break;
                                }
                            }
                        }
                        indices_clusters.push_back(inds); // add new merged cluster to list
                    }
                }
                // fixme: there are no checks to ensure that there is only one new merged cluster per iteration! 
        }
        return zMatrix;
    };
};

