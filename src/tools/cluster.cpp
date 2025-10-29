#include "cluster.h"

Cluster::Cluster(){
    this->indices;
    this->cSum;
    this->sqSum;
    this->n;
}
Cluster::Cluster(Veci indices, Vec cSum, Vec sqSum, int n){
    this->indices = indices;
    this->cSum = cSum;
    this->sqSum = sqSum;
    this->n = n;
}

//getter functions
Veci Cluster::getIndices(){
    return indices;
}
Vec Cluster::getCsum(){
    return cSum;
}
Vec Cluster::getSQsum(){
    return sqSum;
}
int Cluster::getN(){
    return n;
}