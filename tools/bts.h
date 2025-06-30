#pragma once
#include "../support/types.h"

double msd(MatrixXd& data, int M);
double msd(int N, int M, VectorXd& cSum, VectorXd& sqSum);
double extendedComparison(MatrixXd& data, DataType dt=full, Metric mt=MSD, int N=0, int M=1, Threshold cThreshold=Threshold(), int wt=-1);
VectorXd compSim(MatrixXd& data, Metric mt, int M=1);
sortvd sortCompSims(int n, VectorXd& compSimResults);
Index calcMedoid(MatrixXd& data, Metric mt, int M=1);
Index calcOutlier(MatrixXd& data, Metric mt, int M=1);
MatrixXd trimOutliers(MatrixXd& data, double fracToTrim, Metric mt, int M, TrimCriteria ct=TrimCriteria::compSimilarity);
MatrixXd trimOutliers(MatrixXd& data, long nToTrim, Metric mt, int M, TrimCriteria ct=TrimCriteria::compSimilarity);
MatrixXd trimOutliersCompSim(MatrixXd& data, long nToTrim, Metric mt, int M);
MatrixXd trimOutliersMedoid(MatrixXd& data, long nToTrim, Metric mt, int M);
VectorXi diversitySelection(MatrixXd& data, int percent, Metric mt, int M, InitiateDiversity id);
VectorXi diversitySelection(MatrixXd& data, vector<int>& selectedN, int nMax, Metric mt, int M);
Index getNewIndexN (MatrixXd& data, RowVectorXd& selectedCondensed, int n, set<int>& selectFromN, Metric mt=MSD,  int M=0);
Index getNewIndexN (MatrixXd& data, RowVectorXd& selectedCondensed, int n, set<int>& selectFromN, RowVectorXd& sqSelectedCondensed,  int M=0);
MatrixXd refineDisMatrix(MatrixXd& data);