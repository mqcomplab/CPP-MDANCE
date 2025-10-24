#include "../tools/types.h"
#include "../tools/bts.h"

class Helm{
    map<int, Mat> clusterMap;
    MD::Metric mt;
    int nAtoms;
    int nClusters;
    float eps;
    bool trimStart;
    float trimVal;
    int trimK;
    float minSamples;
    int trimIncoming;

    void run(){

    };

    //i know there are bugs, i will fix them later
    void trimClusters(){
        vector<pair<double, int>> clusterMsds;
        int i=0;
        for(auto it=clusterMap.begin(); it!=clusterMap.end(); it++){
            Vec cSum = it->second;  //place holder 
            Vec sqSum = it->second;
            int Nik = it->second;

            if(Nik < minSamples){
                i++;
                continue;
            }

            Mat data(2, cSum.size());
            data.col(0) = cSum;
            data.col(1) = sqSum;
            double sim = extendedComparison(data, Nik, nAtoms, true, mt);
            clusterMsds.emplace_back(make_pair(sim, i));
            i++;
        }

        sort(clusterMsds.begin(), clusterMsds.end());

        //trim the clusters based on the trim_k or trim_val
        map<int, int> newClusterMap;
        if(trimK){
            trimIncoming = clusterMsds.size()-trimK;
            if (trimK >= clusterMsds.size()-1){
                std::cerr<<"trimK is too large!"<<std::endl;
                //reeturn something
            }
            else if(trimK = clusterMsds.size()/2){
                //change this to warning
                std::cerr<<"trimK is more than 50/% of the clusters. This may lead to poor clustering"<<std::endl;
            }
        } 
    }
};