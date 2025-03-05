#include "bts.h"
#include "esim.h"

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

float extendedComparison(MatrixXf& data, DataType dt=full, Metric mt=MSD, int N=0, int M=1, Threshold cThreshold=Threshold(), Weight wt=fraction) {
    if (N == 0){
       N=data.size(); 
    }
    VectorXf cSum;
    VectorXf sqSum;
    if (dt==full){
        cSum = data.colwise().sum();
        if (mt==MSD){
            MatrixXf sqData = data.array() * data.array();
            sqSum = sqData.colwise().sum();
        }
    }
    else if (dt == condensed){
        cSum = data.row(0);
        if (mt == MSD) {
            sqSum = data.row(1);
        }
    }
    if (mt == MSD) {
        return msd(N, M, cSum, sqSum);
    }
    else {
        Indices idx = *(genSimIdx(cSum, N, cThreshold, wt));
        return 1 - idx.getIndex(mt);
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

    VectorXf compSims(N);

    if (mt == MSD){
        compSims = (2 * ((N-1) * compSq.array() - compC.array() * compC.array() )/ ((N-1)*(N-1)*M)).rowwise().sum();
    } else {
        for (int i=0; i<N; ++i){
            VectorXf objSq = data.row(i).array() * data.row(i).array();
            MatrixXf compData (2,N);
            compData.row(0) = cSum.array() - data.row(i).array();
            compData.row(1) = sqSum.array() - objSq.array();
            compSims[i] = extendedComparison(compData, condensed, mt, N-1, M);
        }
    }

   return compSims;
}

Index calcMedoid(MatrixXf& data, Metric mt, int M=1){
    Index maxIdx;
    compSim(data,mt,M).maxCoeff(&maxIdx);
    return maxIdx;
}

Index calcOutlier(MatrixXf& data, Metric mt, int M=1){
    Index minIdx;
    compSim(data,mt,M).minCoeff(&minIdx);
    return minIdx;
}

MatrixXf trimOutliers(MatrixXf& data, float fracToTrim, Metric mt, int M, TrimCriteria ct=TrimCriteria::compSim) {
    long n = lround(data.size() * fracToTrim);
    trimOutliers(data, n, mt, M, ct);
}
MatrixXf trimOutliers(MatrixXf& data, long nToTrim, Metric mt, int M, TrimCriteria ct=TrimCriteria::compSim) {
    switch(ct) {
        case TrimCriteria::compSim: return trimOutliersCompSim(data, nToTrim, mt, M);
        case simToMedoid: return trimOutliersCompSim(data, nToTrim, mt, M);
    }
}
MatrixXf trimOutliersCompSim(MatrixXf& data, long nToTrim, Metric mt, int M) {
    VectorXf cSum = data.colwise().sum();
    VectorXf sqSumTotal = (data.array() * data.array()).colwise().sum();
    std::pair<float, int> compSims[data.size()];
    for (int i=0; i<data.size(); ++i){
        VectorXf c = cSum.array() - data.row(i).array();
        VectorXf sq = sqSumTotal.array() - data.row(i).array()*data.row(i).array();
        MatrixXf compData (2,data.row(0).size());
        compData.row(0) = c;
        compData.row(1) = sq;
        
        compSims[i].second = i;
        compSims[i].first = extendedComparison(compData,condensed, mt, data.size()-1, M);
    }
    std::sort(compSims,compSims+data.size());

    VectorXi indices(data.size()-nToTrim);
    for(int i=nToTrim;i<data.size(); ++i) {
        indices[i-nToTrim] = compSims[i].second;
    }
    return data(Eigen::all,indices);
}
MatrixXf trimOutliersMedoid(MatrixXf& data, long nToTrim, Metric mt, int M) {
    int medoidIdx = calcMedoid(data,mt,M);
    // TODO: Check whether medoid should be deleted so we have nToTrim+1 values removed in the ennd
    std::pair<float, int> compSims[data.size()-1];
    for (int i = 0; i<data.size(); ++i){
        if (i!=medoidIdx){
            MatrixXf compData (2,data.row(0).size());
            compData.row(0) = data.row(i);
            compData.row(1) = data.row(medoidIdx);

            compSims[i].second = i;
            compSims[i].first = extendedComparison(compData, full, mt, 0, M);
        }
    }

    std::sort(compSims,compSims+data.size()-1);

    VectorXi indices(data.size()-nToTrim);
    for(int i=0;i<data.size()-nToTrim; ++i) {
        indices[i] = compSims[i].second;
    }
    return data(Eigen::all,indices);
}

VectorXi* diversitySelection(MatrixXf& data, int percent, Metric mt, int M=1, InitiateDiversity id=medoid){
    VectorXi seed(1);
    switch(id){
        case medoid:
            seed[0]=calcMedoid(data, mt, M);;
        case outlier:
            seed[0]=calcOutlier(data, mt, M);;
        case random:
            seed[0]=(int)floor(rand()*M);
    }
    return diversitySelection(data, percent, mt, M, seed);
}
VectorXi* diversitySelection(MatrixXf& data, int percent, Metric mt, int M=1, VectorXi& selectedN){
    int nMax = M * percent / 100;
    if (nMax > M){
        cerr << "Percentage is too high!\n";
        return;
    }
    MatrixXf selection = data(Eigen::all,selectedN);
    VectorXf selectedCondensed = selection.colwise().sum();
    VectorXf sqSelectedCondensed (data.row(0).size());
    if (mt == MSD){
        MatrixXf sqSelection = selection.array() * selection.array();
        sqSelectedCondensed = sqSelection.colwise().sum();
    } 

    while (selectedN.size() < nMax) {
        // TODO: figure out more efficient way to index
        VectorXi selectFromN(data.size()-selectedN.size());
        
        int newIndexN;
        if (mt == MSD){
            newIndexN = getNewIndexN(data, mt, selectedCondensed, selectedN.size(), selectFromN, &sqSelectedCondensed, M);
            sqSelectedCondensed = sqSelectedCondensed.array() + data.row(newIndexN).array() * data.row(newIndexN).array();
        }
        else {
            newIndexN = getNewIndexN(data, mt, selectedCondensed, selectedN.size(), selectFromN);
        }
        selectedCondensed = selectedCondensed.array() + data.row(newIndexN).array();
        selectedN.conservativeResize(selectedN.size()+1);
        selectedN[selectedN.size()-1] = newIndexN;
    }
}


Index getNewIndexN (MatrixXf& data, Metric mt=MSD, VectorXf& selectedCondensed, int n, VectorXi& selectFromN, VectorXf* sqSelectedCondensed=nullptr,  int M=0) {
    ++n;
    float minVal = -1; // TODO: why is it callled minVal if it's the max?
    int idx = data.size()+1;
    for(int i : selectFromN){
        int simIndex;
        if (mt == MSD) {
            MatrixXf compData (2,data.row(0).size());
            compData.row(0) = selectedCondensed.array() + data.row(i).array();
            compData.row(1) = sqSelectedCondensed->array() + data.row(i).array() * data.row(i).array();
            simIndex = extendedComparison(compData, condensed, mt, n, M);
        }
        else {
            MatrixXf compData (1,data.row(0).size());
            compData.row(0) = selectedCondensed.array() + data.row(i).array();
            simIndex = extendedComparison(compData, condensed, mt, n);
        }

        if (simIndex > minVal) {
            minVal = simIndex;
            idx = i;
        }
    }
    return idx;
}