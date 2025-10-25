#include "cluster.h"

Cluster::Cluster(){

}

//getter functions
vector<int> Cluster::getIndices(){
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