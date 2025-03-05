#pragma once
#include "../support/types.h"

float msd(MatrixXf& data, int n_atoms);
float msd(MatrixXf& data, int n_atoms, VectorXf& c_sum, VectorXf& sq_sum);
VectorXf compSim(MatrixXf& data, int M);
Eigen::Index calcMedoid(MatrixXf& data, int M);
Eigen::Index calcOutlier(MatrixXf& data, int M);