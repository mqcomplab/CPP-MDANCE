#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <sstream>
#include <Eigen/Dense>
#include "../tools/bts.h"

Eigen::MatrixXd readCSVtoEigen(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<float> row;
        std::string value;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));  // Convert string to float
        }

        data.push_back(row);
    }
    file.close();

    // Convert to Eigen::MatrixXd
    int rows = data.size();
    int cols = rows > 0 ? data[0].size() : 0;
    Eigen::MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix(i, j) = data[i][j];

    return matrix;
}

void printMatrix(Eigen::MatrixXd& mat){
    std::cout << "[ ";
    for (int i =0; i < mat.rows(); ++i) {
      for (int j=0; j < mat.row(i).size()-1; ++j) {
        std::cout << mat.row(i)[j] << ", ";
      }
      if (i < mat.rows()-1) {
        std::cout << mat.row(i)[mat.row(i).size()-1] << "; ";
      }
      else {
        std::cout << mat.row(i)[mat.row(i).size()-1] << " ]" << std::endl;
      }
    }
}

void printVector(Eigen::VectorXd& vec){
    std::cout << "[ ";
    for (int i =0; i < vec.size()-1; ++i) {
      std::cout << vec[i] << ", ";
    }
    std::cout << vec[vec.size()-1] << " ]" << std::endl;
}

void printVector(vector<int> vec){
    std::cout << "[";
    for (int i =0; i < vec.size()-1; ++i) {
      std::cout << vec[i] << ", ";
    }
    std::cout << vec[vec.size()-1] << "]" << std::endl;
}

void run_tests(Eigen::MatrixXd data, int M){
    Eigen::MatrixXd condensedData (2,data.cols());
    MatrixXd sqData = data.array() * data.array();
    VectorXd cSum = data.colwise().sum();
    VectorXd sqSum = sqData.colwise().sum();
    condensedData.row(0) = cSum;
    condensedData.row(1) = sqSum;
    VectorXd compSimilar = compSim(data, Metric::MSD, M);
    MatrixXd trim = trimOutliers(data, 0.6, Metric::MSD, M);
    vector<int> diversity = diversitySelection(data, 40, Metric::MSD, M, InitiateDiversity::medoid);
    MatrixXd distances = refineDisMatrix(data);
    std::cout << msd(data, M) << "\n" << extendedComparison(data, DataType::full, Metric::MSD, 0, M) << "\n" << extendedComparison(condensedData, DataType::condensed, Metric::MSD, data.rows(), M) << "\n" << calcMedoid(data, Metric::MSD, M) << "\n" << calcOutlier(data, Metric::MSD, M) << "\n";
    printVector(compSimilar);
    std::cout << trim.rows() << "\n";
    printMatrix(trim);
    printVector(diversity);
    printMatrix(distances);
}


void run_tests(std::string filename, int M){
    try {
        Eigen::MatrixXd matrix = readCSVtoEigen(filename);
        run_tests(matrix, M);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    run_tests("bit.csv", 1);
    run_tests("continuous.csv", 2);
//    run_tests("sim.csv", 50);
    run_tests("small.csv", 3);
//    run_tests("1d.csv", 1);
    return 0;
}
