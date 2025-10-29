//this cluster h file is to help define clusterDict object in helm.cpp
#pragma once
#include "types.h"

class Cluster{
    private:
        /*
        |   indices: cluster indices of merged clusters
        |   c_sum: feature array of the column-wise sum of data
        |   sq_sum: feature array of the column-wise of squared data
        |   n: number of elements in cluster

        */
        Veci indices;
        Vec cSum; 
        Vec sqSum;
        int n;
        
    public:
        Cluster();
        Cluster(Veci indices, Vec cSum, Vec sqSum, int n);
        Veci getIndices();
        Vec getCsum();
        Vec getSQsum();
        int getN();
        
};