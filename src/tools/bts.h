#pragma once
#include "../support/types.h"

double meanSqDev(MatrixXd& data, int nAtoms);
double msdCondensed(VectorXd& cSum, VectorXd& sqSum, int N, int nAtoms);