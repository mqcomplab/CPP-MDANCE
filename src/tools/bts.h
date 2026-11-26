#pragma once

#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "esim.h"
#include "types.h"

double meanSqDev(const ArrayXXd& data, int nAtoms = 1);
double msdCondensed(const ArrayXd& cSum, const ArrayXd& sqSum, Index N, int nAtoms = 1);
double extendedComparison(const ArrayXXd& data, Index N = 0, int nAtoms = 1, bool isCondensed = false, MD::Metric mt = MD::Metric::MSD, MD::Threshold cThreshold = MD::Threshold(MD::ThresholdType::None, 1), int wPower = 0);
ArrayXd calculateCompSim(const ArrayXXd& data, int nAtoms = 1, MD::Metric mt = MD::Metric::MSD);
double compSimMSDRow(const ArrayXXd& data, Index i, const ArrayXd& cSum, const ArrayXd& sqSum, int nAtoms);
vector<int> topCompSimOrder(const ArrayXXd& data, int m, int nAtoms, MD::Metric mt);
Index calculateMedoid(const ArrayXXd& data, int nAtoms = 1, MD::Metric mt = MD::Metric::MSD);
Index calculateMedoid(const ArrayXd& data);
Index calculateOutlier(const ArrayXXd& data, int nAtoms = 1, MD::Metric mt = MD::Metric::MSD);
Index calculateOutlier(const ArrayXd& data);
ArrayXXd trimOutliers(const ArrayXXd& data, int nTrimmed, int nAtoms = 1, bool isMedoid = false, MD::Metric mt = MD::Metric::MSD);
ArrayXXd trimOutliers(const ArrayXXd& data, float nTrimmed, int nAtoms = 1, bool isMedoid = false, MD::Metric mt = MD::Metric::MSD);
vector<Index> diversitySelection(const ArrayXXd& data, int percentage, MD::Metric mt = MD::Metric::MSD, int nAtoms = 1, bool isCompSim = false, MD::StartSeed start = MD::StartSeed::Medoid);
vector<Index> diversitySelection(const ArrayXXd& data, int percentage, MD::Metric mt, int nAtoms, vector<Index>& start);
// Buffers getNewIndexN reuses for every candidate.
struct DiversityScratch {
    ArrayXd c, s, tc, ts;
    explicit DiversityScratch(Index d) : c(d), s(d), tc(d), ts(d) {}
};
Index getNewIndexN(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& rows,
                   MD::Metric mt, const ArrayXXd& selectedCondensed, int N,
                   const vector<Index>& selectFrom, int nAtoms, DiversityScratch& scratch);
ArrayXi repSample(const ArrayXXd& data, MD::Metric mt = MD::Metric::MSD, int nAtoms = 1, int nBins = 10, int nSamples = 100, bool hardCap = true);
ArrayXi repSample(const ArrayXXd& data, MD::Metric mt, int nAtoms, int nBins, double nSamples, bool hardCap = true);
ArrayXXd refineDisMatrix(const ArrayXXd& data);
ArrayXXd alignTraj(const ArrayXXd& data, int nAtoms, MD::AlignMethod = MD::AlignMethod::None);
