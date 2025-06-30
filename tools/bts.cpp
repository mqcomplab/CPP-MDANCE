#include "bts.h"
#include "esim.h"

/* Inputs
 * ------------
 * data: the feature array of size (n_samples, n_features)
 * M : number of atoms in the MD system. M = 1 for non-MD systems.
*/
double msd(MatrixXd& data, int M){
    int N = data.rows();
    if (N == 1)
        return 0;
    MatrixXd sqData = data.array().square();
    VectorXd cSum = data.colwise().sum();
    VectorXd sqSum = sqData.colwise().sum();
    return msd(N, M, cSum, sqSum);
}
/* Inputs
 * ------------
 * N : The number of frames
 * M : number of atoms in the MD system. M = 1 for non-MD systems.
 * cSum : A feature array of the column-wise sum of the data.
 * sq_sum : A feature array of the column-wise sum of the squared data. 
*/
double msd(int N, int M, VectorXd& cSum, VectorXd& sqSum) {
    if (N == 1)
        return 0;
    VectorXd meanSqSum = sqSum.array() / N;
    VectorXd meanCSum = cSum.array() / N;
    double msd = (meanSqSum.array() - meanCSum.array().square()).sum();
    return 2*msd / M;
}

double extendedComparison(MatrixXd& data, DataType dt, Metric mt, int N, int M, Threshold cThreshold, int wt) {
    if (N == 0){
       N=data.rows(); 
    }
    VectorXd cSum;
    VectorXd sqSum;
    if (dt==full){
        cSum = data.colwise().sum();
        if (mt==MSD){
            MatrixXd sqData = data.array().square();
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
        Indices idx = genSimIdx(cSum, N, cThreshold, wt);
        return 1 - idx.getIndex(mt);
    }
}

/* Inputs
 * ------------
 * data: the feature array of size (n_samples, n_features)
 * M : number of atoms in the MD system. M = 1 for non-MD systems.
*/
VectorXd compSim(MatrixXd& data, Metric mt, int M){
    int N = data.rows();
    MatrixXd sqData = data.array().square();
    VectorXd cSum = data.colwise().sum();
    VectorXd sqSum = sqData.colwise().sum();
    MatrixXd compC = ((-data).rowwise()+cSum.transpose()) / (N-1);
    MatrixXd compSq = ((-sqData).rowwise()+sqSum.transpose()) / (N-1);

    VectorXd compSims(N);

    if (mt == MSD){
        compSims = (2 * (compSq.array() - compC.array().square())/ M).rowwise().sum();
    } else {
        for (int i=0; i<N; ++i){
            VectorXd objSq = data.row(i).array().square();
            MatrixXd compData (2,N);
            compData.row(0) = cSum.array() - data.row(i).array();
            compData.row(1) = sqSum.array() - objSq.array();
            compSims[i] = extendedComparison(compData, condensed, mt, N-1, M);
        }
    }

   return compSims;
}

sortvd sortCompSims(int n, VectorXd& compSimResults) {
    sortvd compSims;
    compSims.reserve(n);
    for(int i=0; i<n; ++i){
        compSims.emplace_back(compSimResults[i], i);
    }
    
    std::sort(compSims.begin(),compSims.end());

    return compSims;
}

Index calcMedoid(MatrixXd& data, Metric mt, int M){
    Index maxIdx;
    compSim(data,mt,M).maxCoeff(&maxIdx);
    return maxIdx;
}

Index calcOutlier(MatrixXd& data, Metric mt, int M){
    Index minIdx;
    compSim(data,mt,M).minCoeff(&minIdx);
    return minIdx;
}

MatrixXd trimOutliers(MatrixXd& data, double fracToTrim, Metric mt, int M, TrimCriteria ct) {
    long n = std::floor(data.rows() * fracToTrim);
    return trimOutliers(data, n, mt, M, ct);
}
MatrixXd trimOutliers(MatrixXd& data, long nToTrim, Metric mt, int M, TrimCriteria ct) {
    switch(ct) {
        case TrimCriteria::compSimilarity: return trimOutliersCompSim(data, nToTrim, mt, M);
        case simToMedoid: return trimOutliersMedoid(data, nToTrim, mt, M);
    }
    return trimOutliersCompSim(data, nToTrim, mt, M); // TODO: better default
}
MatrixXd trimOutliersCompSim(MatrixXd& data, long nToTrim, Metric mt, int M) {
    VectorXd cSum = data.colwise().sum();
    VectorXd sqSumTotal = data.array().square().colwise().sum();
    vector<std::pair<double, int>> compSims;
    compSims.reserve(data.rows()-1);
    for (int i=0; i<data.rows(); ++i){
        VectorXd c = cSum.array().transpose() - data.row(i).array();
        VectorXd sq = sqSumTotal.array().transpose() - data.row(i).array().square();
        MatrixXd compData (2,data.row(0).size());
        compData.row(0) = c;
        compData.row(1) = sq;
        
        compSims.emplace_back(std::pair<double, int>(extendedComparison(compData, condensed, mt, data.rows()-1, M),i));
    }
    std::sort(compSims.begin(),compSims.end());
    vector<int> indices;
    indices.reserve(data.rows()-nToTrim);
    for(int i=nToTrim;i<data.rows(); ++i) {
        indices.emplace_back(compSims[i].second);
    }
    std::sort(indices.begin(),indices.end());
    return data(indices, Eigen::all);
}
MatrixXd trimOutliersMedoid(MatrixXd& data, long nToTrim, Metric mt, int M) {
    int medoidIdx = calcMedoid(data,mt,M);
    // TODO: Check whether medoid should be deleted so we have nToTrim+1 values removed in the end
    vector<std::pair<double, int>> compSims;
    compSims.reserve(data.rows()-1);
    for (int i = 0; i<data.rows(); ++i){
        if (i!=medoidIdx){
            MatrixXd compData (2,data.row(0).size());
            compData.row(0) = data.row(i);
            compData.row(1) = data.row(medoidIdx);

            compSims.emplace_back(std::pair<double, int>(extendedComparison(compData, full, mt, 0, M),i));
        }
    }

    std::sort(compSims.begin(),compSims.end());

    vector<int> indices;
    indices.reserve(data.rows()-nToTrim);
    for(int i=nToTrim;i<data.rows(); ++i) {
        indices.emplace_back(compSims[i].second);
    }
    std::sort(indices.begin(),indices.end());
    return data(indices, Eigen::all);
}

void diversitySelection(MatrixXd& data, VectorXi& selected, int nMax, Metric mt, int M){

    VectorXi indicesToSelect(nMax);
    if (nMax > 1) {
        double step = (data.rows() - 1) * 1.0 / (nMax - 1);
        for(int i=0; i<nMax; ++i){
            indicesToSelect[i] = (int)round(i * step);
        }
    }
    indicesToSelect[0] = 0;
    VectorXd compSims = compSim(data, mt, M);
    sortvd sortedComps = sortCompSims(data.rows(), compSims);
    for(int i=0; i<nMax; ++i){
        selected[i] = sortedComps[indicesToSelect[i]].second;
    }
}

VectorXi diversitySelection(MatrixXd& data, int percent, Metric mt, int M, InitiateDiversity id){
    int N = data.rows();
    int nMax = N * percent / 100;
    if (nMax > N){
        cerr << "Percentage is too high!\n";
        return;
    }
    if (nMax < 1){
        cerr << "Percentage is too low!\n";
        return;
    }
    VectorXi seed(nMax);
    switch(id){
        case strat: diversitySelection(data, seed, nMax, mt, M); return seed;
        case medoid:
            seed[0] = calcMedoid(data, mt, M); break;
        case outlier:
            seed[0] = calcOutlier(data, mt, M); break;
        case randomly:
            seed[0] = (int)floor(rand()*M); break;
    }
    diversitySelection(data, seed, 1, nMax, mt, M);
    return seed;
}
VectorXi diversitySelection(MatrixXd& data, int percent, Metric mt, int M, VectorXi& start){
    int N = data.rows();
    int nMax = N * percent / 100;
    if (nMax > N){
        cerr << "Percentage is too high!\n";
        return;
    }
    VectorXi seed(nMax);
    seed.head(start.size()) = start;
    diversitySelection(data, seed, nMax, start.size(), mt, M);
    return seed;
}
void diversitySelection(MatrixXd& data, VectorXi& selectedN, int initSize, int nMax, Metric mt, int M){
    MatrixXd selection = data(selectedN,Eigen::all);
    RowVectorXd selectedCondensed = selection.colwise().sum();
    RowVectorXd sqSelectedCondensed (data.row(0).size());
    if (mt == MSD){
        MatrixXd sqSelection = selection.array().square();
        sqSelectedCondensed = sqSelection.colwise().sum();
    } 

    set<int> selectFromN;
    for (int i=0; i<data.rows(); ++i){
        selectFromN.insert(selectFromN.end(), i);
    }

    for (int i: selectedN){
        selectFromN.erase(i);
    }

    while (initSize < nMax) {
        // TODO: figure out more efficient way to index

        int newIndexN;
        if (mt == MSD){
            newIndexN = getNewIndexN(data, selectedCondensed, initSize, selectFromN, sqSelectedCondensed, M);
            sqSelectedCondensed = sqSelectedCondensed.array() + data.row(newIndexN).array().square();
        }
        else {
            newIndexN = getNewIndexN(data, selectedCondensed, initSize, selectFromN, mt);
        }
        selectedCondensed = selectedCondensed.array() + data.row(newIndexN).array();
        selectedN[initSize] = newIndexN;
        selectFromN.erase(newIndexN);
        ++initSize;
    }
}

Index getNewIndexN (MatrixXd& data, RowVectorXd& selectedCondensed, int n, set<int>& selectFromN, Metric mt,  int M) {
    ++n;
    double minVal = -1; // TODO: why is it callled minVal if it's the max?
    int idx = data.rows()+1;
    for(int i : selectFromN){
        int simIndex;
        MatrixXd compData (1,data.row(0).size());
        compData.row(0) = selectedCondensed.array() + data.row(i).array();
        simIndex = extendedComparison(compData, condensed, mt, n);
        

        if (simIndex > minVal) {
            minVal = simIndex;
            idx = i;
        }
    }
    return idx;
}

Index getNewIndexN (MatrixXd& data, RowVectorXd& selectedCondensed, int n, set<int>& selectFromN, RowVectorXd& sqSelectedCondensed,  int M) {
    ++n;
    double minVal = -1; // TODO: why is it callled minVal if it's the max?
    int idx = data.rows()+1;
    for(int i : selectFromN){
        double simIndex;
        MatrixXd compData (2,data.row(0).size());
        VectorXd test1 = selectedCondensed.array() + data.row(i).array(); 
        VectorXd test2 = sqSelectedCondensed.array() + data.row(i).array().square();
        compData.row(0) = test1;
        compData.row(1) = test2;
        simIndex = extendedComparison(compData, condensed, Metric::MSD, n, M);
        
        if (simIndex > minVal) {
            minVal = simIndex;
            idx = i;
        }
    }
    return idx;
}

MatrixXd refineDisMatrix(MatrixXd& data) {
    if (data.rows() != data.row(0).size()){
        cerr << "Input must be a square matrix!\n";
        return;
    }
    MatrixXd distances = data.array() + data.transpose().array() * 0.5  - data.minCoeff();
    for (int i=0; i<data.rows(); ++i) {
        distances.row(i)[i] = 0;
    }
    return distances;
}