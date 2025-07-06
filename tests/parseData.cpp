#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <Eigen/Dense>
#include "../src/tools/bts.h"
using std::chrono::high_resolution_clock, std::chrono::duration;

Eigen::MatrixXd readCSVtoEigen(const std::string& filename) {
    std::ifstream file("data/"+filename);
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

void printVector(VectorXd& vec){
    std::cout << "[ ";
    for (int i =0; i < vec.size()-1; ++i) {
      std::cout << vec[i] << ", ";
    }
    std::cout << vec[vec.size()-1] << " ]" << std::endl;
}

void printVector(VectorXi& vec){
    std::cout << "[";
    for (int i =0; i < vec.size()-1; ++i) {
      std::cout << vec[i] << ", ";
    }
    std::cout << vec[vec.size()-1] << "]" << std::endl;
}

void run_tests(MatrixXd data, int nAtoms){
    auto start = high_resolution_clock::now();
    double msd = meanSqDev(data, nAtoms);
    auto end = high_resolution_clock::now();
    duration<double> dur = end - start;
    std::cout << meanSqDev(data, nAtoms) << std::endl;
    std::cerr << dur.count() << std::endl;
}


void run_tests(std::string filename, int nAtoms){
    try {
        Eigen::MatrixXd matrix = readCSVtoEigen(filename);
        run_tests(matrix, nAtoms);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    run_tests("bit.csv", 1);
    run_tests("continuous.csv", 2);
    run_tests("sim.csv", 50);
    run_tests("small.csv", 3);
    run_tests("1d.csv", 1);
    run_tests("mid.csv", 3);
    return 0;
}
