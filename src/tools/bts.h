#pragma once
#include "types.h"
#include "esim.h"
#include <cstdlib>

double meanSqDev(MatrixXd& data, int nAtoms = 1);
double msdCondensed(VectorXd& cSum, VectorXd& sqSum, Index N, int nAtoms = 1);
double extendedComparison(MatrixXd& data, Index N = 0, int nAtoms = 1, bool isCondensed = false, Metric mt = Metric::MSD, Threshold cThreshold = Threshold(ThresholdType::None, 1), int wPower = 0);
VectorXd calculateCompSim(MatrixXd& data, int nAtoms = 1, Metric mt = Metric::MSD);
Index calculateMedoid(MatrixXd& data, int nAtoms = 1, Metric mt = Metric::MSD);
Index calculateMedoid(VectorXd& data, int nAtoms = 1, Metric mt = Metric::MSD);
Index calculateOutlier(MatrixXd& data, int nAtoms = 1, Metric mt = Metric::MSD);
Index calculateOutlier(VectorXd& data, int nAtoms = 1, Metric mt = Metric::MSD);
MatrixXd trimOutliers(MatrixXd& data, int nTrimmed, int nAtoms = 1, bool isMedoid = false, Metric mt = Metric::MSD);
MatrixXd trimOutliers(MatrixXd& data, float nTrimmed, int nAtoms = 1, bool isMedoid = false, Metric mt = Metric::MSD);
vector<Index> diversitySelection(MatrixXd& data, int percentage, Metric mt, int nAtoms = 1, bool isCompSim = false, startSeed start = startSeed::Medoid);
vector<Index> diversitySelection(MatrixXd& data, int percentage, Metric mt, int nAtoms, bool isCompSim, vector<Index>& start);