#include "csv_io.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <iomanip>

ArrayXXd readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<double> row;
        std::string value;

        while (std::getline(ss, value, ',')) {
            try {
                double val = std::stod(value);
                if (!std::isfinite(val)) throw std::runtime_error("Non-finite value");
                row.push_back(val);
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid numeric value: " + value +
                    " (" + e.what() + ") at row " + std::to_string(data.size() + 1));
            }
        }
        if (!data.empty() && row.size() != data[0].size()) {
            throw std::runtime_error("Inconsistent number of columns at row " +
                std::to_string(data.size() + 1));
        }
        data.push_back(row);
    }
    file.close();

    int rows = data.size();
    int cols = rows > 0 ? data[0].size() : 0;
    ArrayXXd matrix(rows, cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix(i, j) = data[i][j];

    return matrix;
}

void writeCSV(const std::string& filename, const ArrayXXd& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            if (j > 0) file << ",";
            file << std::setprecision(10) << data(i, j);
        }
        file << "\n";
    }
    file.close();
}
