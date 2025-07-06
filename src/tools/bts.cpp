#include "bts.h"

/* O(N) Mean square deviation(MSD) calculation for n-ary objects.
 *  
 * Arguments:
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
 * Arguments:
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
