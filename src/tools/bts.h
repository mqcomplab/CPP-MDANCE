#pragma once
#include "types.h"
#include "esim.h"
#include <cstdlib>

double meanSqDev(ArrayXXd& data, int nAtoms = 1);
double msdCondensed(ArrayXd& cSum, ArrayXd& sqSum, Index N, int nAtoms = 1);
double extendedComparison(ArrayXXd& data, Index N = 0, int nAtoms = 1, bool isCondensed = false, MD::Metric mt = MD::Metric::MSD, MD::Threshold cThreshold = MD::Threshold(MD::ThresholdType::None, 1), int wPower = 0);
ArrayXd calculateCompSim(ArrayXXd& data, int nAtoms = 1, MD::Metric mt = MD::Metric::MSD);
Index calculateMedoid(ArrayXXd& data, int nAtoms = 1, MD::Metric mt = MD::Metric::MSD);
Index calculateMedoid(ArrayXd& data);
Index calculateOutlier(ArrayXXd& data, int nAtoms = 1, MD::Metric mt = MD::Metric::MSD);
Index calculateOutlier(ArrayXd& data);
ArrayXXd trimOutliers(ArrayXXd& data, int nTrimmed, int nAtoms = 1, bool isMedoid = false, MD::Metric mt = MD::Metric::MSD);
ArrayXXd trimOutliers(ArrayXXd& data, float nTrimmed, int nAtoms = 1, bool isMedoid = false, MD::Metric mt = MD::Metric::MSD);
vector<Index> diversitySelection(ArrayXXd& data, int percentage, MD::Metric mt = MD::Metric::MSD, int nAtoms = 1, bool isCompSim = false, MD::StartSeed start = MD::StartSeed::Medoid);
vector<Index> diversitySelection(ArrayXXd& data, int percentage, MD::Metric mt, int nAtoms, vector<Index>& start);
Index getNewIndexN(ArrayXXd& data, MD::Metric mt, ArrayXXd& selectedCondensed, int N, set<Index>& selectFromN, int nAtoms);
ArrayXi repSample(ArrayXXd& data, MD::Metric mt = MD::Metric::MSD, int nAtoms = 1, int nBins = 10, int nSamples = 100, bool hardCap = false);