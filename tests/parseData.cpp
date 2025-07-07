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
        std::cout << std::setprecision(5) << mat.row(i)[j] << ", ";
      }
      if (i < mat.rows()-1) {
        std::cout << std::setprecision(5) << mat.row(i)[mat.row(i).size()-1] << "; ";
      }
      else {
        std::cout << std::setprecision(5) << mat.row(i)[mat.row(i).size()-1] << " ]" << std::endl;
      }
    }
}

void printVector(VectorXd& vec){
    std::cout << "[ ";
    for (int i =0; i < vec.size()-1; ++i) {
      std::cout << std::setprecision(5) << vec[i] << ", ";
    }
    std::cout << std::setprecision(5) << vec[vec.size()-1] << " ]" << std::endl;
}

void printVector(VectorXi& vec){
    std::cout << "[";
    for (int i =0; i < vec.size()-1; ++i) {
      std::cout << vec[i] << ", ";
    }
    std::cout << vec[vec.size()-1] << "]" << std::endl;
}

void run_tests(MatrixXd data, int nAtoms){
    // MSD test
    auto start = high_resolution_clock::now();
    double msd = meanSqDev(data, nAtoms);
    auto end = high_resolution_clock::now();
    duration<double> dur = end - start;
    std::cout << "msd: " << msd << std::endl;
    std::cerr << "msd: " << dur.count() << std::endl;

    // Full EC test
    start = high_resolution_clock::now();
    double ec = extendedComparison(data, data.rows(), nAtoms);
    end = high_resolution_clock::now();
    dur = end - start;
    std::cout << "ec: " << ec << std::endl;
    std::cerr << "ec: " << dur.count() << std::endl;

    // Condensed EC test
    VectorXd cSum = data.colwise().sum();
    VectorXd sqSum = data.array().square().colwise().sum();
    MatrixXd condensedData (2,data.cols());
    condensedData.row(0) = cSum;
    condensedData.row(1) = sqSum;

    start = high_resolution_clock::now();
    ec = extendedComparison(condensedData, data.rows(), nAtoms, true);
    end = high_resolution_clock::now();
    dur = end - start;
    std::cout << "condensed ec: " << ec << std::endl;
    std::cerr << "condensed ec: " << dur.count() << std::endl;

    // Esim EC test
    MatrixXd smallerData (1,data.cols());
    smallerData.row(0) = cSum;

    start = high_resolution_clock::now();
    ec = extendedComparison(smallerData, data.rows(), nAtoms, true, Metric::RR);
    end = high_resolution_clock::now();
    dur = end - start;
    std::cout << "esim ec: " << ec << std::endl;
    std::cerr << "esim ec: " << dur.count() << std::endl;

    // compSim test
    start = high_resolution_clock::now();
    VectorXd vec = calculateCompSim(data, nAtoms);
    end = high_resolution_clock::now();
    dur = end - start;
    std::cout << "compSim: ";
    printVector(vec);
    std::cerr << "compSim: " << dur.count() << std::endl;

    // calcMedoid test
    start = high_resolution_clock::now();
    Index idx = calculateMedoid(data, nAtoms);
    end = high_resolution_clock::now();
    dur = end - start;
    std::cout << "calcMedoid: " << idx << std::endl;
    std::cerr << "calcMedoid: " << dur.count() << std::endl;

    // calcOutlier test
    start = high_resolution_clock::now();
    idx = calculateOutlier(data, nAtoms);
    end = high_resolution_clock::now();
    dur = end - start;
    std::cout << "calcOutlier: " << idx << std::endl;
    std::cerr << "calcOutlier: " << dur.count() << std::endl;
}


void run_tests(std::string filename, int nAtoms){
    try {
        Eigen::MatrixXd matrix = readCSVtoEigen(filename);
        std::cout << "\n" << filename << "\n----------------------" << std::endl;
        std::cerr << "\n" << filename << "\n----------------------" << std::endl;
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
    // run_tests("1d.csv", 1);
    run_tests("mid.csv", 3);
    return 0;
}
