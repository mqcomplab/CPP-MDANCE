#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <Eigen/Dense>
#include "../src/tools/bts.h"
#include "test_utils.h"
using std::chrono::high_resolution_clock, std::chrono::duration, Eigen::ArrayXXf;

ArrayXXd readCSVtoEigen(const std::string& filename) {
    std::filesystem::path filePath = std::filesystem::canonical(filename);
    std::filesystem::path buildTestPath = std::filesystem::current_path();
    std::filesystem::path projectRoot = buildTestPath.parent_path().parent_path();
    std::filesystem::path dataPath = projectRoot / "tests" / "data";
    // this utility function is only intended to read files from the data directory for testing purposes
    if (filePath.string().rfind(dataPath.string(), 0) != 0){
        throw std::runtime_error("File path " + filename + " is not inside the data directory: " + dataPath.string());
    }
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath.string());
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string value;

        while (std::getline(ss, value, ',')) {
            // Convert string to double with error checking
            try {
                double val = std::stod(value);
                if (!std::isfinite(val)) throw std::runtime_error("Non-finite value");
                row.push_back(val);
            }  catch (const std::exception& e) {
                throw std::runtime_error("Invalid numeric value: " + value + " (" + e.what() + ") at row " + std::to_string(data.size()+1));
            } 
        }
        // Check for consistent number of columns
        if (!data.empty() && row.size() != data[0].size()) {
            throw std::runtime_error("Inconsistent number of columns at row " + std::to_string(data.size() + 1));
        }
        data.push_back(row);
    }
    file.close();

    // Convert to ArrayXXd
    int rows = data.size();
    int cols = rows > 0 ? data[0].size() : 0;
    ArrayXXd matrix(rows, cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix(i, j) = data[i][j];

    return matrix;
}

ArrayXXf readCSVtoEigenArr(const std::string& filename) {
    std::filesystem::path filePath = std::filesystem::canonical(filename);
    std::filesystem::path buildTestPath = std::filesystem::current_path();
    std::filesystem::path projectRoot = buildTestPath.parent_path().parent_path();
    std::filesystem::path dataPath = projectRoot / "tests" / "data";
    // this utility function is only intended to read files from the data directory for testing purposes
    if (filePath.string().rfind(dataPath.string(), 0) != 0){
        throw std::runtime_error("File path " + filename + " is not inside the data directory: " + dataPath.string());
    }
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath.string());
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
 
    // Convert to ArrayXXd
    int rows = data.size();
    int cols = rows > 0 ? data[0].size() : 0;
    ArrayXXf matrix(rows, cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix(i, j) = data[i][j];

    return matrix;
}

void printMatrix(ArrayXXd& mat){
    std::cout << "[ ";
    for (int i =0; i < mat.rows(); ++i) {
      for (int j=0; j < mat.row(i).size()-1; ++j) {
        std::cout << std::setprecision(10) << mat.row(i)[j] << ", ";
      }
      if (i < mat.rows()-1) {
        std::cout << std::setprecision(10) << mat.row(i)[mat.row(i).size()-1] << "; ";
      }
      else {
        std::cout << std::setprecision(10) << mat.row(i)[mat.row(i).size()-1] << " ]" << std::endl;
      }
    }
}

void printVector(ArrayXd& vec){
    std::cout << "[ ";
    for (int i =0; i < vec.size()-1; ++i) {
      std::cout << std::setprecision(10) << vec[i] << ",\n ";
    }
    std::cout << std::setprecision(10) << vec[vec.size()-1] << " ]" << std::endl;
}

void printVector(ArrayXi& vec){
    std::cout << "[ ";
    for (int i =0; i < vec.size()-1; ++i) {
      std::cout << vec[i] << ",\n ";
    }
    std::cout << vec[vec.size()-1] << " ]" << std::endl;
}

void printVector(vector<Index>& vec){
    std::cout << "[ ";
    for (int i =0; i < vec.size()-1; ++i) {
      std::cout << vec[i] << ", ";
    }
    std::cout << vec[vec.size()-1] << " ]" << std::endl;
}