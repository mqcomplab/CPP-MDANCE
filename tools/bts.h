#pragma once
#include <Eigen/Dense>
using Eigen::MatrixXf, Eigen::VectorXf;

float msd(MatrixXf data, int n_atoms);
float msd(MatrixXf data, int n_atoms, VectorXf c_sum, VectorXf sq_sum);
VectorXf comp_sim(MatrixXf data, int M);