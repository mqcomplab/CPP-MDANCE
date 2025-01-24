#include "bts.h"

/* Inputs
 * ------------
 * data: the feature array of size (n_samples, n_features)
 * M : number of atoms in the MD system. M = 1 for non-MD systems.
*/
float msd(MatrixXf data, int M){
    int N = data.rows();
    if (N == 1)
        return 0;
    MatrixXf sq_data = data.array() * data.array();
    VectorXf c_sum = data.colwise().sum();
    VectorXf sq_sum = sq_data.colwise().sum();
    return msd(N, M, c_sum, sq_sum);
}
/* Inputs
 * ------------
 * N : The number of frames
 * M : number of atoms in the MD system. M = 1 for non-MD systems.
 * c_sum : A feature array of the column-wise sum of the data.
 * sq_sum : A feature array of the column-wise sum of the squared data. 
*/
float msd(int N, int M, VectorXf c_sum, VectorXf sq_sum) {
    if (N == 1)
        return 0;
    float msd = (2 * (N * c_sum.array() - c_sum.array() * c_sum.array()) / (N*N) ).sum();
    return msd / M;
}

VectorXf comp_sim(MatrixXf data, int M){
    int N = data.rows();
    MatrixXf sq_data = data.array() * data.array();
    VectorXf c_sum = data.colwise().sum();
    VectorXf sq_sum = sq_data.colwise().sum();
    MatrixXf comp_c = (-data).colwise()+c_sum;
    MatrixXf comp_sq = (-sq_data).colwise()+sq_sum;

    // Should replace with metric-based evaluation. For now, our only metric is MSD
    VectorXf comp_msd = (2*((N-1) * comp_sq.array() - comp_c.array() * comp_c.array())).rowwise().sum();
    return comp_msd / M;
}