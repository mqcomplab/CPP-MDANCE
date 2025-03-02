#include "bts.h"
#include "../support/types.h"

/* Inputs
 * ------------
 * data: the feature array of size (n_samples, n_features)
 * M : number of atoms in the MD system. M = 1 for non-MD systems.
*/
float msd(MatrixXf& data, int M){
    int N = data.rows();
    if (N == 1)
        return 0;
    MatrixXf sqData = data.array() * data.array();
    VectorXf cSum = data.colwise().sum();
    VectorXf sqSum = sqData.colwise().sum();
    return msd(N, M, cSum, sqSum);
}
/* Inputs
 * ------------
 * N : The number of frames
 * M : number of atoms in the MD system. M = 1 for non-MD systems.
 * cSum : A feature array of the column-wise sum of the data.
 * sq_sum : A feature array of the column-wise sum of the squared data. 
*/
float msd(int N, int M, VectorXf& cSum, VectorXf& sqSum) {
    if (N == 1)
        return 0;
    float msd = (2 * (N * sqSum.array() - cSum.array() * cSum.array()) / (N*N) ).sum();
    return msd / M;
}

float extendedComparison(MatrixXf& data, DataType dt=full, Metric mt=MSD, int N=0, int M=1, int cThreshold=0, Weight wt=fraction) {
    if (N == 0){
       N=data.size(); 
    }
    if (dt==full){

    }
}
/* Inputs
 * ------------
 * data: the feature array of size (n_samples, n_features)
 * M : number of atoms in the MD system. M = 1 for non-MD systems.
*/
VectorXf compSim(MatrixXf& data, Metric mt, int M=1){
    int N = data.rows();
    MatrixXf sqData = data.array() * data.array();
    VectorXf cSum = data.colwise().sum();
    VectorXf sqSum = sqData.colwise().sum();
    MatrixXf compC = (-data).colwise()+cSum;
    MatrixXf compSq = (-sqData).colwise()+sqSum;

    if (mt == MSD){

    } else {
        
    }

    VectorXf compMsd = (2*((N-1) * compSq.array() - compC.array() * compC.array())).rowwise().sum();
    return compMsd / M;
}

Index calcMedoid(MatrixXf& data, int M=1){
    Index maxIdx;
    compSim(data,M).maxCoeff(&maxIdx);
}

Index calcOutlier(MatrixXf& data, int M=1){
    Index minIdx;
    compSim(data,M).minCoeff(&minIdx);
}

Index getNewIndexN (MatrixXf& data, Metric mt=MSD, MatrixXf& selectedCondensed, int n, VectorXi& selectFromN, VectorXf* sqSelectedCondensed=nullptr,  int M=0) {
    ++n;
    float minVal = -1;
    int idx = data.size()+1;
    for(int i : selectFromN){
        
    }
}