#include "bts.h"

#include <cmath>
#include <functional>


/* O(N) Mean square deviation(MSD) calculation for n-ary objects.
 *  
 * Parameters:
 *  - data: A feature array of size (nSamples, nFeatures)
 *  - nAtoms: Number of atoms in the Molecular Dynamics (MD) system. nAtoms=1 for non-MD systems.
 * 
 * Returns: normalized MSD value
 * 
 * Reference: https://github.com/mqcomplab/MDANCE/blob/9a895e72d71fee1d1a4fad1700a806473dff2f71/src/mdance/tools/bts.py#L14
*/ 
double meanSqDev(const ArrayXXd& data, int nAtoms){
    Index N = data.rows();
    if (N == 1)
        return 0;
    ArrayXd cSum = data.colwise().sum();
    ArrayXd sqSum = data.square().colwise().sum();
    return msdCondensed(cSum, sqSum, N, nAtoms);
}

/* Condensed version of Mean square deviation (MSD) calculation for n-ary objects
 * 
 * Parameters:
 *  - cSum: A feature array of the column-wise sum of the data (nFeatures)
 *  - sqSum: A feature array of the column-wise sum of the squared data (nFeatures)
 *  - N: Number of data points
 *  - nAtoms: Number of atoms in the Molecular Dynamics (MD) system. nAtoms=1 for non-MD systems.
 * 
 * Returns: normalized MSD value.
 * 
 * Reference: https://github.com/mqcomplab/MDANCE/blob/9a895e72d71fee1d1a4fad1700a806473dff2f71/src/mdance/tools/bts.py#L54
*/
double msdCondensed(const ArrayXd& cSum, const ArrayXd& sqSum, Index N, int nAtoms){
    if (N == 1)
        return 0;
    // The following is a step-by-step explanation of what we are returning. May need to use this instead if we run into overflow issues?
    // ArrayXd meanSqSum = sqSum / N;
    // ArrayXd meanCSum = cSum / N;
    // return msd = (meanSqSum - meanCSum.square()).sum() * 2.0 / nAtoms;
    if (N <= 0) {
        throw std::invalid_argument("N must be positive and non-zero.");
    }
    if (nAtoms <= 0) {
        throw std::invalid_argument("nAtoms must be positive and non-zero.");
    }
    // The denominator is formed in double: N * N * nAtoms in int overflowed
    // well below N = 46340 once nAtoms > 1 (10 000 frames of 22 atoms), and
    // gave a negative or garbage MSD. Where the int product did not overflow
    // it was exact, and so is the double one: the values are unchanged.
    return (double)2.0 * (sqSum * N - cSum.square()).sum() / ((double)N * (double)N * (double)nAtoms);
}

/* O(N) Extended comparison function for n-ary objects.
 *
 * Parameters:
 * - data: A feature array which can take on multiple formats
 *    --> if (!isCondensed): an array of size (nSamples, nFeatures)
 *    --> if (isCondensed): a ArrayXXd with 1 row (cSum)
 *    --> if (isCondensed): a ArrayXXd with 2 rows (cSum, sqSum)
 * - isCondensed: Controls type of data (see above)
 * - mt: The metric to use when calculating distance between n objects in an array
 * - N: Number of data points.
 * - nAtoms: Number of atoms in the Molecular Dynamics (MD) system. nAtoms=1 for non-MD systems
 * - cThreshold: CCoincidence threshold for calculatinig extended similarity
 * - wPower: Controls the type of weight function for calculating extended similarity
 *    --> if (wPower): use fraction method
 *    --> else: use powerN method where N=wPower
 * 
 * Returns: Extended comparison value.
 * 
 * Reference: https://github.com/mqcomplab/MDANCE/blob/main/src/mdance/tools/bts.py#L96
*/
double extendedComparison(const ArrayXXd& data, Index N, int nAtoms, bool isCondensed, MD::Metric mt, MD::Threshold cThreshold, int wPower) {
    // N = 0 (the default) means the rows of the data. The similarity indices
    // were computed with N = 0 objects then (-inf for RR on any data); only
    // the MSD, which counts the rows itself, was right.
    if (!isCondensed && N <= 0)
        N = data.rows();
    if (isCondensed && N <= 0)
        throw std::invalid_argument("extendedComparison: condensed data needs the number of objects N.");
    // Handle default initialization of MD::Threshold
    if (cThreshold.type == MD::ThresholdType::None)
        cThreshold.value = N % 2;
    // Data check
    if (isCondensed){
        if (data.rows() > 2){
            throw std::runtime_error("Data must have at most two rows: either (cSum) or (cSum, sqSum)");
        }
        ArrayXd cSum = data.row(0);
        if (mt == MD::Metric::MSD){
            ArrayXd sqSum = data.row(1);
            return msdCondensed(cSum, sqSum, N, nAtoms);
        } else {
            MD::Indices idx = genSimIdx(cSum, N, cThreshold, wPower);
            return 1 - idx.getIndex(mt);
        }
    } else {
        if (mt == MD::Metric::MSD) {
            return meanSqDev(data, nAtoms);
        } else {
            ArrayXd cSum = data.colwise().sum();
            MD::Indices idx = genSimIdx(cSum, N, cThreshold, wPower);
            return 1 - idx.getIndex(mt);
        }
    }

}

/* O(N) Complementary similarity calculation for n-ary objects.
 *
 * Parameters:
 *  - data: A feature array of shape (nSamples, nFeatures).
 *  - nAtoms: Number of atoms in the Molecular Dynamics (MD) system. nAtoms=1 for non-MD systems.
 *  - mt: The metric to use when calculating distance between n objects in an array.
 * 
 * Returns: Vector (N) of complementary similarities for each object
 * 
 * Reference: https://github.com/mqcomplab/MDANCE/blob/main/src/mdance/tools/bts.py#L190
*/
ArrayXd calculateCompSim(const ArrayXXd& data, int nAtoms, MD::Metric mt) {
    Index N = data.rows();
    if (N == 0) {
        return ArrayXd();
    }
    if (nAtoms <= 0) {
        throw std::invalid_argument("nAtoms must be positive and non-zero.");
    }

    ArrayXd cSum = data.colwise().sum();
    ArrayXd sqSum = data.square().colwise().sum();

    ArrayXd compSims(N);

    if (mt == MD::Metric::MSD){
        // Each row's value is summed over the columns in order, one column
        // of the data at a time (contiguous), into one accumulator per row:
        // no temporary the size of the data (the expression this replaces
        // made three: 2 GB for 100 000 frames of 300 atoms), and a row's
        // value does not depend on where the row is (Eigen's row-wise sum
        // grouped rows into packets and summed the last rows otherwise).
        // compSimMSDRow computes one row's value the same way.
        compSims.setZero();
        const double n1 = (double)(N-1);
        for (Index j = 0; j < data.cols(); ++j) {
            const double* col = data.col(j).data();
            const double cj = cSum[j], sj = sqSum[j];
            for (Index i = 0; i < N; ++i) {
                const double x = col[i];
                const double cc = (cj - x) / n1;
                const double sq = (sj - x * x) / n1;
                compSims[i] += 2 * (sq - cc * cc) / nAtoms;
            }
        }
    } else {
        // (one buffer for every row, not two rows allocated per row)
        ArrayXd comp(data.cols());
        for (Index i=0; i<N; ++i){
            comp = cSum - data.row(i).transpose();
            MD::Threshold th(MD::ThresholdType::None, 1);
            th.value = (N-1) % 2;
            MD::Indices idx = genSimIdx(comp, N-1, th, 0);
            compSims[i] = 1 - idx.getIndex(mt);
        }
    }

   return compSims;
}

// One row's MSD complementary similarity, as calculateCompSim sums it.
double compSimMSDRow(const ArrayXXd& data, Index i, const ArrayXd& cSum, const ArrayXd& sqSum, int nAtoms) {
    const double n1 = (double)(data.rows() - 1);
    double acc = 0;
    for (Index j = 0; j < data.cols(); ++j) {
        const double x = data(i, j);
        const double cc = (cSum[j] - x) / n1;
        const double sq = (sqSum[j] - x * x) / n1;
        acc += 2 * (sq - cc * cc) / nAtoms;
    }
    return acc;
}

/* The first m rows in decreasing order of complementary similarity: what
 * sorting every row by it (std::sort, cs[a] > cs[b]) puts first.
 *
 * With the MSD, each row's value is written by algebra first,
 * (2/nAtoms) [(SQ - |x|^2)/n - (|c|^2 - 2 c.x + |x|^2)/n^2], n = N - 1, one
 * matrix-vector product for every row, and bounded with its rounding error
 * and calculateCompSim's. Only the rows whose bound reaches the m-th best
 * are computed exactly. Where their exact values leave no tie in the first
 * m + 1, the order is the one the sort gives; else (and for the other
 * metrics) every row is computed and sorted, as before.
 */
vector<int> topCompSimOrder(const ArrayXXd& data, int m, int nAtoms, MD::Metric mt) {
    const Index N = data.rows();
    if (m > N) m = (int)N;
    if (m <= 0) return vector<int>();
    auto full = [&]() {
        ArrayXd cs = calculateCompSim(data, nAtoms, mt);
        vector<int> order(N);
        for (Index i = 0; i < N; ++i) order[i] = (int)i;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return cs[a] > cs[b]; });
        order.resize(m);
        return order;
    };
    if (mt != MD::Metric::MSD || N < 64 || (Index)m * 4 > N || nAtoms <= 0) {
        return full();
    }
    ArrayXd cSum = data.colwise().sum();
    ArrayXd sqSum = data.square().colwise().sum();
    const double n1 = (double)(N - 1);
    const double SQ = sqSum.sum();
    const double cc = cSum.square().sum();
    ArrayXd xsq = data.square().rowwise().sum();
    ArrayXd cx = (data.matrix() * cSum.matrix()).array();
    const double gam = 8.0 * (double)(data.cols() + 16) * std::numeric_limits<double>::epsilon();
    const double k = 2.0 / nAtoms;
    ArrayXd approx(N), err(N);
    for (Index i = 0; i < N; ++i) {
        approx[i] = k * ((SQ - xsq[i]) / n1 - (cc - 2 * cx[i] + xsq[i]) / (n1 * n1));
        err[i] = gam * k * ((SQ + xsq[i]) / n1 + (cc + 2 * std::sqrt(cc * xsq[i]) + xsq[i]) / (n1 * n1));
        if (!std::isfinite(approx[i]) || !std::isfinite(err[i])) return full();
    }
    // the m-th largest lower bound
    vector<double> lower(N);
    for (Index i = 0; i < N; ++i) lower[i] = approx[i] - err[i];
    std::nth_element(lower.begin(), lower.begin() + (m - 1), lower.end(), std::greater<double>());
    const double L = lower[m - 1];
    vector<pair<double, int>> near;
    for (Index i = 0; i < N; ++i) {
        if (approx[i] + err[i] >= L) near.emplace_back(compSimMSDRow(data, i, cSum, sqSum, nAtoms), (int)i);
    }
    std::sort(near.begin(), near.end(), [](const pair<double, int>& x, const pair<double, int>& y) {
        return x.first > y.first;
    });
    // no tie among the first m, nor between the m-th and the next
    for (int q = 0; q < m && q + 1 < (int)near.size(); ++q) {
        if (!(near[q].first > near[q + 1].first)) return full();
    }
    vector<int> order(m);
    for (int q = 0; q < m; ++q) order[q] = near[q].second;
    return order;
}

/* O(N) medoid calculation for n-ary objects.
 *
 * Parameters:
 *  - data: A feature array of shape (nSamples, nFeatures)
 *     --> Can also be a vector (N) of complementary similarities for each object. Useful when calculating medoid and outlier so you only calculate compSims once.
 *  - nAtoms: Number of atoms in the Molecular Dynamics (MD) system. nAtoms=1 for non-MD systems.
 *  - mt: The metric to use when calculating distance between n objects in an array. 
 * 
 * Returns: The index of the medoid in the dataset.
 * 
 * Reference: https://github.com/mqcomplab/MDANCE/blob/main/src/mdance/tools/bts.py#L241
*/
Index calculateMedoid(const ArrayXXd& data, int nAtoms, MD::Metric mt) {
    ArrayXd compSims = calculateCompSim(data, nAtoms, mt);
    return calculateMedoid(compSims);
}
Index calculateMedoid(const ArrayXd& data) {
    if (data.size() == 0) {
        throw std::invalid_argument("medoid of an empty set of frames.");
    }
    Index maxIdx;
    data.maxCoeff(&maxIdx);
    return maxIdx;
}

/* O(N) outlier calculation for n-ary objects.
 *
 * Parameters:
 *  - data: A feature array of shape (nSamples, nFeatures)
 *     --> Can also be a vector (N) of complementary similarities for each object. Useful when calculating medoid and outlier so you only calculate compSims once.
 *  - nAtoms: Number of atoms in the Molecular Dynamics (MD) system. nAtoms=1 for non-MD systems.
 *  - mt: The metric to use when calculating distance between n objects in an array. 
 * 
 * Returns: The index of the medoid in the dataset.
 * 
 * Reference: https://github.com/mqcomplab/MDANCE/blob/main/src/mdance/tools/bts.py#L271
*/
Index calculateOutlier(const ArrayXXd& data, int nAtoms, MD::Metric mt) {
    ArrayXd compSims = calculateCompSim(data, nAtoms, mt);
    return calculateOutlier(compSims);
}
Index calculateOutlier(const ArrayXd& data) {
    if (data.size() == 0) {
        throw std::invalid_argument("outlier of an empty set of frames.");
    }
    Index minIdx;
    data.minCoeff(&minIdx);
    return minIdx;
}

/* O(N * log(nTrimmed)) method of trimming a desired percentage of outliers (most dissimilar) from a feature array.
 *
 * Parameters:
 *  - data: A feature array of shape (nSamples, nFeatures).
 *  - nTrimmed: The desired # of outliers to be removed. Can be a number (int), or a percentage (float).
 *  - nAtoms: Number of atoms in the Molecular Dynamics (MD) system. nAtoms=1 for non-MD systems.
 *  - isMedoid: Criterion to use for data trimming
 *     --> if (!isMedoid): remove most dissimilar objects based on complement similarity.
 *     --> if (isMedoid): remove most dissimilar objects based on similarity to the medoid.
 * 
 *  Returns: A feature array with the desired outliers removed.
 * 
 *  Reference: https://github.com/mqcomplab/MDANCE/blob/main/src/mdance/tools/bts.py#L301
*/
ArrayXXd trimOutliers(const ArrayXXd& data, int nTrimmed, int nAtoms, bool isMedoid, MD::Metric mt) {
    Index N = data.rows();
    if (nTrimmed <= 0) {
        return data;
    }
    if (nTrimmed >= N) {
        return ArrayXXd(0, data.cols());
    }

    Index medoidIdx = -1;
    ArrayXd cSum;
    ArrayXd sqSumTotal;
    if (isMedoid) {
        medoidIdx = calculateMedoid(data, nAtoms, mt);
    } else {
        cSum = data.colwise().sum();
        sqSumTotal = data.square().colwise().sum();
    }

    ArrayXXd compData(2, data.cols());
    auto trimValue = [&](Index i) {
        if (isMedoid) {
            // Dissimilarity between frame i and the medoid (a 2-object
            // comparison; the medoid itself scores 0 and is never trimmed).
            compData.row(0) = data.row(i);
            compData.row(1) = data.row(medoidIdx);
            return extendedComparison(compData, 2, nAtoms, false, mt);
        }
        // Complementary similarity: dissimilarity of the data without frame i.
        compData.row(0) = cSum.transpose() - data.row(i);
        compData.row(1) = sqSumTotal.transpose() - data.row(i).square();
        return extendedComparison(compData, N - 1, nAtoms, true, mt);
    };

    // Select the nTrimmed outliers in O(N log nTrimmed) with a bounded heap.
    // comp_sim trims the LOWEST complementary similarities (a max-heap keeps
    // the nTrimmed smallest values); sim_to_medoid trims the HIGHEST
    // distances to the medoid (a min-heap keeps the nTrimmed largest).
    vector<pair<double, Index>> heap;
    heap.reserve(nTrimmed);
    auto cmp = [isMedoid](const pair<double, Index>& a, const pair<double, Index>& b) {
        return isMedoid ? a.first > b.first : a.first < b.first;
    };
    for (Index i = 0; i < N; ++i) {
        double val = trimValue(i);
        if ((Index)heap.size() < nTrimmed) {
            heap.emplace_back(val, i);
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else if (isMedoid ? (val > heap.front().first) : (val < heap.front().first)) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = pair<double, Index>(val, i);
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }

    vector<bool> trimmed(N, false);
    for (auto& p : heap) {
        trimmed[p.second] = true;
    }
    vector<Index> indices;
    indices.reserve(N - nTrimmed);
    for (Index i = 0; i < N; ++i) {
        if (!trimmed[i]) {
            indices.push_back(i);
        }
    }
    return data(indices, Eigen::all);
}
ArrayXXd trimOutliers(const ArrayXXd& data, float nTrimmed, int nAtoms, bool isMedoid, MD::Metric mt) {
    int num = std::floor(data.rows() * nTrimmed);
    if (num == 0)
        return data;
    return trimOutliers(data, num, nAtoms, isMedoid, mt);
}

/* O(N) method of selecting the most diverse subset of a data matrix using the complementary similarity.
 *
 * Parameters:
 *  - data: A feature array of shape (nSamples, nFeatures).
 *  - percentage: Indicates the percentage of data to be selected.
 *  - mt: The metric to use when calculating distance between n objects in an array
 *  - nAtoms: Number of atoms in the Molecular Dynamics (MD) system. nAtoms=1 for non-MD systems.
 *  - isCompSim: The method to use for diversity selection.
 *     --> if (!isCompSim): Uses stratified sampling.
 *     --> if (isCompSim): Maximizes the MSD between the selected objects and the rest of the data.
 *  - start: The initial seed for initiating diversity selection.
 *     --> You can also specify the seed indices as a vector<Index>
 * 
 * Returns: A vector of indices of the diversity selected data (in order selected).
 * 
 * Reference: https://github.com/mqcomplab/MDANCE/blob/016bd9aff30d1c2add26b36bfcf64aa665a34a1d/src/mdance/tools/bts.py#L376
*/
vector<Index> diversitySelection(const ArrayXXd& data, int percentage, MD::Metric mt, int nAtoms, bool isCompSim, MD::StartSeed start){
    if (data.rows() == 0) {
        return vector<Index>();
    }
    if (percentage < 0) {
        throw std::invalid_argument("diversity selection: the percentage must not be negative.");
    }
    if (isCompSim) {
        vector<Index> seed;
        switch(start) {
            case MD::StartSeed::Medoid: seed.push_back(calculateMedoid(data, nAtoms, mt)); break;
            case MD::StartSeed::Outlier: seed.push_back(calculateOutlier(data, nAtoms, mt)); break;
            case MD::StartSeed::Random: seed.emplace_back(rand() % data.rows()); break;  // (not reachable from the C API: rand() is global)
        }
        return diversitySelection(data, percentage, mt, nAtoms, seed);
    }
    Index N = data.rows();
    long long nMaxL = (long long)N * percentage / 100;
    if (nMaxL > N) {
        throw std::runtime_error("Percentage is too high for the given matrix size");
    }
    int nMax = (int)nMaxL;
    if (nMax == 0) {
        return vector<Index>();
    }
    vector<Index> indices (nMax);
    if (nMax == 1)
        indices[0] = 0;
    else {
        double step = (static_cast<double>(N - 1)) / (nMax -1);
        for (int i = 0; i < nMax; ++i){
            indices[i] = std::round(i * step);
        }
    } 
    ArrayXd compSims = calculateCompSim(data, nAtoms, mt);
    vector<pair<double,int>> compSimArray;
    compSimArray.reserve(compSims.size());
    for (int i=0; i<compSims.size(); ++i){
        compSimArray.emplace_back(-compSims[i],i);
    }
    std::sort(compSimArray.begin(), compSimArray.end());

    for(int i=0; i<nMax; ++i){
        indices[i] = compSimArray[indices[i]].second;
    }
    return indices;

}
vector<Index> diversitySelection(const ArrayXXd& data, int percentage, MD::Metric mt, int nAtoms, vector<Index>& indices){
    Index nTotalI = data.rows();
    if (nTotalI == 0) {
        return indices;
    }
    for (Index i : indices) {
        if (i < 0 || i >= nTotalI) {
            throw std::out_of_range("diversity selection: seed index out of range.");
        }
    }
    ArrayXXd selection = data(indices, Eigen::all);
    ArrayXXd selected (mt == MD::Metric::MSD ? 2 : 1, data.cols());
    selected.row(0) = selection.colwise().sum();
    if (mt == MD::Metric::MSD) {
        selected.row(1) = selection.square().colwise().sum();
    }

    int nTotal = (int)nTotalI;
    // (in 64 bits: 100 000 frames times a percentage overflowed an int)
    int nMax = (int)std::min<long long>((long long)nTotal * percentage / 100, nTotal);

    // The candidates, in increasing order (the set this replaces iterated
    // them so, and the first of equal values wins).
    vector<char> taken(nTotal, 0);
    for (Index i : indices) taken[i] = 1;
    vector<Index> selectFrom;
    selectFrom.reserve(nTotal);
    for (int i=0; i<nTotal; ++i) {
        if (!taken[i]) selectFrom.push_back(i);
    }

    // Each candidate's row is read at every step: a row of the column-major
    // data is strided by nTotal doubles, a cache miss per value. One
    // row-major copy, made once.
    using RowMat = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMat rows = data;
    DiversityScratch scratch(data.cols());

    // With the MSD, each step is narrowed first: every candidate's MSD with
    // the selection is written by algebra, n S + n |x|^2 - |c|^2 - 2 c.x -
    // |x|^2 (S the selection's sum of squares, c its column sums), one
    // matrix-vector product for all of them, and bounded with its rounding
    // error and the exact sum's. Only the candidates whose bound reaches the
    // best one's are then computed exactly, as getNewIndexN always did, in
    // the same order: the frame chosen is the same, the work a fraction.
    const bool narrow = (mt == MD::Metric::MSD);
    ArrayXd xsq, approx, cx;
    vector<Index> near;
    if (narrow) {
        xsq.resize(nTotal);
        for (int i = 0; i < nTotal; ++i) xsq[i] = rows.row(i).square().sum();
        near.reserve(nTotal);
    }
    const double gam = 4.0 * (double)(data.cols() + 16) * std::numeric_limits<double>::epsilon();

    indices.reserve(nMax);
    while ((int)indices.size() < nMax && !selectFrom.empty()) {
        MD::checkCancel();
        const vector<Index>* cands = &selectFrom;
        if (narrow && selectFrom.size() > 8) {
            const double n = (double)indices.size() + 1;
            ArrayXd c = selected.row(0).transpose();
            const double S = selected.row(1).sum();
            const double cc = c.square().sum();
            cx = (rows.matrix() * c.matrix()).array();
            double lo = -std::numeric_limits<double>::infinity();
            bool finite = std::isfinite(S) && std::isfinite(cc);
            approx.resize(selectFrom.size());
            ArrayXd err(selectFrom.size());
            for (size_t q = 0; q < selectFrom.size() && finite; ++q) {
                Index i = selectFrom[q];
                double a = n * S + n * xsq[i] - cc - 2 * cx[i] - xsq[i];
                double t = n * S + n * xsq[i] + cc + 2 * std::sqrt(cc * xsq[i]) + xsq[i];
                double e = gam * t;
                if (!std::isfinite(a) || !std::isfinite(e)) { finite = false; break; }
                approx[q] = a;
                err[q] = e;
                lo = std::max(lo, a - e);
            }
            if (finite) {
                near.clear();
                for (size_t q = 0; q < selectFrom.size(); ++q) {
                    if (approx[q] + err[q] >= lo) near.push_back(selectFrom[q]);
                }
                cands = &near;
            }
        }
        Index newIndexN = getNewIndexN(rows, mt, selected, (int)indices.size(), *cands, nAtoms, scratch);
        if (newIndexN < 0) {
            // every candidate scored NaN: the coordinates are not finite.
            // (This used to pick "frame" ncols, past the end of the data.)
            throw std::runtime_error("diversity selection: the similarity of every candidate is not a number (coordinates that are not finite?).");
        }

        selected.row(0) += data.row(newIndexN);
        if (mt == MD::Metric::MSD)
            selected.row(1) = selected.row(1) + data.row(newIndexN).square();

        selectFrom.erase(std::lower_bound(selectFrom.begin(), selectFrom.end(), newIndexN));
        indices.push_back(newIndexN);
    }

    return indices;
}

/* Extract the new index to add to the list of selected indices.
 *
 * Parameters:
 *  - rows: the feature array (nSamples, nFeatures), row-major.
 *  - mt: The metric to use when calculating distance between n objects in an array
 *  - selectedCondensed: A fingerprint feature array that can take on multiple shapes:
 *     --> if (mt == MSD): a ArrayXXd with 2 rows (cSum, sqSum)
 *     --> else: a ArrayXXd with 1 row (cSum)
 *  - N: number of selected objects
 *  - selectFrom: indices to select from, in increasing order
 *  - nAtoms: Number of atoms in the Molecular Dynamics (MD) system. nAtoms=1 for non-MD systems.
 *
 * Returns: index of the new fingerprint to add to the selected indices, or
 * -1 when no candidate's value is a number.
 *
 * The value of each candidate is computed as extendedComparison computed it
 * (the same sums, in the same buffers' order), without allocating: a 2-row
 * matrix and two vectors per candidate made this the slowest step of k-means
 * seeding, eQual and the diversity pick.
 *
 * Reference: https://github.com/mqcomplab/MDANCE/blob/016bd9aff30d1c2add26b36bfcf64aa665a34a1d/src/mdance/tools/bts.py#L489
*/
Index getNewIndexN(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& rows,
                   MD::Metric mt, const ArrayXXd& selectedCondensed, int N,
                   const vector<Index>& selectFrom, int nAtoms, DiversityScratch& sc) {
    // Number of fingerprints already selected and the new one to add
    int nTotal = N + 1;

    double maxVal = -std::numeric_limits<double>::infinity();
    Index idx = -1;

    const Index D = rows.cols();
    sc.c = selectedCondensed.row(0).transpose();
    if (mt == MD::Metric::MSD) sc.s = selectedCondensed.row(1).transpose();
    MD::Threshold th(MD::ThresholdType::None, 1);
    th.value = nTotal % 2;

    for (Index i : selectFrom){
        Eigen::Map<const ArrayXd> x(rows.row(i).data(), D);
        double simIdx;
        sc.tc = sc.c + x;
        if (mt == MD::Metric::MSD) {
            sc.ts = sc.s + x.square();
            simIdx = msdCondensed(sc.tc, sc.ts, nTotal, nAtoms);
        } else {
            MD::Indices ix = genSimIdx(sc.tc, nTotal, th, 0);
            simIdx = 1 - ix.getIndex(mt);
        }

        if (simIdx > maxVal){
            maxVal = simIdx;
            idx = i;
        }
    }
    return idx;
}

/* Representative sampling according to compSim values: Divides the range of comp_sim values in nbins and then uniformly selects n_samples molecules, consecutively taking one from each bin
 *
 * Parameters:
 *  - data: A feature array of shape (nSamples, nFeatures).
 *  - mt: The metric to use when calculating distance between n objects in an array
 *  - nAtoms: Number of atoms in the Molecular Dynamics (MD) system. nAtoms=1 for non-MD systems.
 *  - nBins: Number of bins to divide the compSim values.
 *  - nSamples: Number of samples to be selected.
 *  - hardCap: whether the number of samples will be *exactly* nSamples
 * 
 * Returns: List of indices of the sampled objects in the original data
 * 
 * Reference: https://github.com/mqcomplab/MDANCE/blob/016bd9aff30d1c2add26b36bfcf64aa665a34a1d/src/mdance/tools/bts.py#L652
*/
ArrayXi repSample(const ArrayXXd& data, MD::Metric mt, int nAtoms, int nBins, double nSamples, bool hardCap) {
    // nSamples in (0,1) is a fraction of the data; anything >= 1 is a count.
    if (nSamples < 1) {
        return repSample(data, mt, nAtoms, nBins, (int)(data.rows() * nSamples), hardCap);
    }
    return repSample(data, mt, nAtoms, nBins, (int)nSamples, hardCap);
}
ArrayXi repSample(const ArrayXXd& data, MD::Metric mt, int nAtoms, int nBins, int nSamples, bool hardCap) {
    Index N = data.rows();
    if (nSamples <= 0 || N == 0) {
        return ArrayXi();
    }
    if (nBins < 1) {
        throw std::invalid_argument("nBins must be at least 1");
    }

    ArrayXd compSims = calculateCompSim(data, nAtoms, mt);
    // (compSim value, original index), sorted by value
    vector<pair<double,int>> compSimArray;
    compSimArray.reserve(N);
    for (int i=0; i<N; ++i){
        compSimArray.emplace_back(compSims[i], i);
    }
    std::sort(compSimArray.begin(), compSimArray.end());

    double mi = compSimArray.front().first;
    double ma = compSimArray.back().first;
    double binWidth = (ma - mi) / nBins;

    // Distribute the objects across nBins bins of equal compSim width. When
    // every value is identical the reference puts everything in the last bin.
    vector<vector<int>> bins(nBins);
    for (auto& p : compSimArray) {
        int b = binWidth > 0 ? (int)((p.first - mi) / binWidth) : nBins - 1;
        if (b < 0) b = 0;
        if (b >= nBins) b = nBins - 1;  // the maximum value lands in the last bin
        bins[b].push_back(p.second);
    }

    // Round-robin over the bins, taking the next unused object of each bin per
    // pass. With hardCap the collection stops exactly at nSamples; otherwise
    // the current pass is completed first (so slightly more than nSamples may
    // be returned), as in the Python reference.
    vector<int> sampled;
    sampled.reserve(nSamples);
    for (int depth = 0; (int)sampled.size() < nSamples; ++depth) {
        bool added = false;
        for (int b = 0; b < nBins; ++b) {
            if ((int)bins[b].size() > depth) {
                sampled.push_back(bins[b][depth]);
                added = true;
                if (hardCap && (int)sampled.size() >= nSamples)
                    break;
            }
        }
        if (!added) {
            break;  // every bin is exhausted: fewer than nSamples objects exist
        }
    }

    return Eigen::Map<ArrayXi>(sampled.data(), sampled.size());
}

/* Refine a distance matrix by setting the diagonal to zero and symmetrizing the matrix
 *
 * Parameters:
 *  - data: A feature array of shape (nSamples, nFeatures).
 * 
 * Returns: A refined 2D matrix.
 * 
 * Reference: https://github.com/mqcomplab/MDANCE/blob/016bd9aff30d1c2add26b36bfcf64aa665a34a1d/src/mdance/tools/bts.py#L720
*/
ArrayXXd refineDisMatrix(const ArrayXXd& data) {
    if (data.rows() == 1 || data.cols() == 1) {
        throw std::invalid_argument("Matrix must be 2D.");
    }
    if (data.rows() != data.cols()) {
        throw std::invalid_argument("Matrix must be square.");
    }

    ArrayXXd distances = (data + data.transpose()) / 2;
    distances = distances - distances.minCoeff();
    for (int i=0; i<distances.rows(); ++i) {
        distances.row(i)[i] = 0;
    }
    return distances;
}


//not finished
ArrayXXd alignTraj(const ArrayXXd& data, int nAtoms, MD::AlignMethod alignMeth){
    /*
        Aligns trajectory using uniforms or kronecker alignment

        Parameters
        --------------
        data: dimensions: nSamples x nFeatures
        nAtoms: number of atoms in the system
        alignMethod: {Uni, Kron}, default = None

        Returns
        --------------
        matrix of aligned data

        References
        --------------
        Klem, H., Hocky, G. M., and McCullagh M., `"Size-and-Shape Space Gaussian 
        Mixture Models for Structural Clustering of Molecular Dynamics Trajectories"`_.
        *Journal of Chemical Theory and Computation* **2022** 18 (5), 3218-3230

        .. _"Size-and-Shape Space Gaussian Mixture Models for Structural Clustering of Molecular Dynamics Trajectories":
        https://pubs.acs.org/doi/abs/10.1021/acs.jctc.1c01290
    */

    (void)nAtoms;  // stub: unused until the alignment methods are implemented

    if(alignMeth == MD::AlignMethod::None){
        return data;
    }

    return data;
}