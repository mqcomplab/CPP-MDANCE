#pragma once

#include <string>
#include <Eigen/Dense>

using Eigen::ArrayXXd;

ArrayXXd readCSV(const std::string& filename);
void writeCSV(const std::string& filename, const ArrayXXd& data);
