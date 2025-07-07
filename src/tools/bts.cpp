#include "bts.h"
#include <iostream>

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
double meanSqDev(MatrixXd& data, int nAtoms){
    int N = data.rows();
    if (N == 1)
        return 0;
    VectorXd cSum = data.colwise().sum();
    VectorXd sqSum = data.array().square().colwise().sum();
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
double msdCondensed(VectorXd& cSum, VectorXd& sqSum, int N, int nAtoms){
    if (N == 1)
        return 0;
    /* The following is a step-by-step explanation of what we are returning. May need to use this instead if we run into overflow issues?
     * VectorXd meanSqSum = sqSum.array() / N;
     * VectorXd meanCSum = cSum.array() / N;
     * double msd = (meanSqSum.array() - meanCSum.array().square()).sum();
    */
    return (sqSum.array() * N - cSum.array().square()).sum() * 2.0 / (N * N * nAtoms);
}

/* O(N) Extended comparison function for n-ary objects.
 *
 * Parameters:
 * - data: A feature array which can take on multiple formats
 *    --> if (!isCondensed): an array of size (nSamples, nFeatures)
 *    --> if (isCondensed): a MatrixXd with 1 row (cSum)
 *    --> if (isCondensed): a MatrixXd with 2 rows (cSum, sqSum)
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
double extendedComparison(MatrixXd& data, int N, int nAtoms, bool isCondensed, Metric mt, Threshold cThreshold, int wPower) {
    // Handle default initialization of Threshold
    if (cThreshold.type == ThresholdType::None)
        cThreshold.value = N % 2;
    // Data check
    if (isCondensed){
        if (data.rows() > 2){
            std::cerr << "Data must have at most two rows: either (cSum) or (cSum, sqSum)" << std::endl;
            exit;
        }
        VectorXd cSum = data.row(0);
        if (mt == Metric::MSD){
            VectorXd sqSum = data.row(1);
            return msdCondensed(cSum, sqSum, N, nAtoms);
        } else {
            Indices idx = genSimIdx(cSum, N, cThreshold, wPower);
            return 1 - idx.getIndex(mt);
        }
    } else {
        if (mt == Metric::MSD) {
            return meanSqDev(data, nAtoms);
        } else {
            VectorXd cSum = data.colwise().sum();
            Indices idx = genSimIdx(cSum, N, cThreshold, wPower);
            return 1 - idx.getIndex(mt);
        }
    }

}