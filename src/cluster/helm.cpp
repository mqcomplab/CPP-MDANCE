#include "../tools/types.h"
#include "../tools/bts.h"
#include "../tools/cluster.h"

class Helm{
    map<int, vector<Cluster>> clusterMap;
    MD::Metric mt;
    int nAtoms;
    int nClusters;
    float eps;          //if None, then eps=-1
    bool trimStart;
    float trimVal;
    int trimK;
    float minSamples;
    int trimIncoming;
    MD::AlignMethod alignMeth;
    MD::MergeScheme mergeScheme;
    Mat clusterDists;
    Mat linkMatrix;
    int totalIncoming;

    void run(){

    };
    Mat makeDataByRow(Vec a, Vec b){
        //a and b need to be same length
        Mat data(2, a.size());
        data.row(0) = a;
        data.row(1) = b;

        return data;
    }
    //finished
    map<int, vector<Cluster>> trimClusters(){
        vector<pair<double, int>> clusterMsds;
        int i=0;
        for(int i=0; i<clusterMap[totalIncoming].size(); i++){
            Vec cSum = clusterMap[totalIncoming].at(i).getCsum();
            Vec sqSum = clusterMap[totalIncoming].at(i).getSQsum();
            int Nik = clusterMap[totalIncoming].at(i).getN();

            if(Nik < minSamples){
                continue;
            }

            Mat data(2, cSum.size());
            data.row(0) = cSum;
            data.row(1) = sqSum;
            clusterMsds.emplace_back(pair<double, int>(extendedComparison(data, Nik, nAtoms, true, mt), i));
        }

        sort(clusterMsds.begin(), clusterMsds.end());

        //trim the clusters based on the trim_k or trim_val
        map<int, vector<Cluster>> newClusterMap;
        if(trimK){
            trimIncoming = clusterMsds.size()-trimK;
            newClusterMap[trimIncoming] = vector<Cluster>();
            if (trimK >= clusterMsds.size()-1){
                std::cerr<<"trimK is too large!"<<std::endl;
                return;
            }
            else if(trimK = clusterMsds.size()/2){
                //change this to warning
                std::cerr<<"trimK is more than 50% of the clusters. This may lead to poor clustering"<<std::endl;
            }
            for(int i=0; i<clusterMsds.size()-trimK; i++){
                newClusterMap[trimIncoming].emplace_back(clusterMap[totalIncoming][i]);
            }
        } 
        else if(trimVal){
            int trimIncoming = 0;
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
            Vec distsToNewCluster(previousClusters.size()-1);
            for(int i=0; i<previousClusters.size()-1; i++){
                float helmSim = calc(previousClusters, i, previousClusters.size()-1);
                distsToNewCluster[i] = helmSim;
            }

            //Add new cluster to distance matrix
            //clusterDists = 
            //clussterDists =
        }   

        //Find the two most similar clusters
        Index minRow, minCol;       //minRow and minCol indicate two different clusters
        float mergeDist = clusterDists.minCoeff(&minRow, &minCol);

        //Merge the two most similar clusters
        Vec cSum, sqSum;
        if(alignMeth == MD::AlignMethod::Kron){
            //add stuff
        }
        else{
            cSum = previousClusters[minRow].getCsum() + previousClusters[minCol].getCsum();
            sqSum = previousClusters[minRow].getSQsum() + previousClusters[minCol].getSQsum();
        }

        Vec cSumik = cSum;
        Vec sqSumik = sqSum;
        int Nik = previousClusters[minRow].getN() + previousClusters[minCol].getN();
        if(alignMeth != MD::AlignMethod::None){
            //aligned combine clusters
        }

        //Save the new clusters after mergin
        vector<Cluster> newClusters;
        for(int i=0; i<previousClusters.size(); i++){
            if(i==minRow || i==minCol)  continue;
            else{
                newClusters.push_back(previousClusters[i]);
            }
        }
        Veci indicesik = previousClusters[minRow].getIndices() + previousClusters[minCol].getIndices();

        //Two different ways of saving the new cluster
        if(alignMeth != MD::AlignMethod::None){
            //newClusters.push_back(Cluster(indicesik, cSumik, sqSumik, Nik, aligned));
        }
        else{
            newClusters.push_back(Cluster(indicesik, cSumik, sqSumik, Nik));
        }

        //Remove distances of merged clusters
        vector<int> clustersToKeep;
        for(int i=0; i<clusterDists.size(); i++){
            if(i!= minRow && i!=minCol){
                clustersToKeep.push_back(i);
            }
        }
        Veci clustersToKeepVec = Veci(clustersToKeep.data(), clustersToKeep.size());
        clusterDists = clusterDists(Eigen::placeholders::all, clustersToKeepVec);
        clusterDists = clusterDists(clustersToKeepVec, Eigen::placeholders::all);

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
            //when alignMeth is not None, previousCluster[i][3]=input_clusters


        } else{
            cSum = cSumA + cSumB;
            sqSum = sqSumA + sqSumB;
        }

        int n = previousClusters[i].getN() + nB;
        Mat data = makeDataByRow(cSum, sqSum);
        double sim = extendedComparison(data, n, nAtoms, true, mt);

        //Different merging schemes for determining which clusters to merge
        float helmSim;
        if (mergeScheme == MD::MergeScheme::intra){
            helmSim = sim;
        }
        else if(mergeScheme == MD::MergeScheme::inter){
            helmSim = ((sim*pow(n,2)) - (simA*pow(nA,2)) - (simB*pow(nB,2)))/(nA*nB);
        }
        else if(mergeScheme == MD::MergeScheme::half){
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

        clusterDists(previousClusters.size(), previousClusters.size());
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

    //gen_link_matrix()
    //i will assume that a link matrix is ArrayXXd (or eq. Mat)

    //can rewrite vectors into Arrays if need be
    void linkMatrixToClusterMap(){
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
        }
    }


};