#include <filesystem>
#include <fstream>
#include <Eigen/Dense>

#include "../src/tools/types.h"
#include "../src/cluster/divine.h"

#include <pybind11/embed.h>

namespace py=pybind11;

ArrayXXd readCSVtoEigen_process(const std::string& filename) {
    std::filesystem::path filePath = std::filesystem::canonical(filename);
    std::filesystem::path buildTestPath = std::filesystem::current_path();
    std::filesystem::path projectRoot = buildTestPath.parent_path().parent_path();
    std::filesystem::path dataPath = projectRoot / "tests" / "data";
    // this utility function is only intended to read files from the data directory for testing purposes
    // if (filePath.string().rfind(dataPath.string(), 0) != 0){
    //     throw std::runtime_error("File path " + filename + " is not inside the data directory: " + dataPath.string());
    // }
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

int main(){
    if (!Py_IsInitialized()) {
        std::cout<<"fdsa"<<std::endl;
        py::initialize_interpreter();
    }
    std::cout<<"asdf"<<std::endl;
    //py::scoped_interpreter guard{};
    std::filesystem::path path = std::filesystem::current_path();
    path = path/"backbone.csv";
    Mat data = readCSVtoEigen_process(path.string());
    std::cout << "Data shape: " << data.rows() << " x " << data.cols() << std::endl;

    // for all runs:
    // Percentage = 10
    // N_atoms = 12
    // Sieve = 1
    int percentage = 10;
    int nAtoms = 12;
    int sieve = 1; 
    int k = 6;
    int end = 0;
    bool refine = true;
    MD::DivineSplit split = MD::DivineSplit::WeightedMSD;
    MD::DivineAnchors anchor = MD::DivineAnchors::SplinterPair;
    MD::KinitType init_type = MD::KinitType::StratAll;
    Divine model = Divine(
                data,
                split,
                anchor,
                init_type,
                end,
                k,
                refine, 
                nAtoms,
                0.0,
                percentage
            );
    vector<vector<Index>> clusters = model.getClusters();
    vector<int> labels = model.getLabels();
    std::cout << "Clusters formed: " << clusters.size() << std::endl;
    Veci labelsVec = Eigen::Map<Veci>(labels.data(), labels.size());
    pair<double, double> scores = model.computeScores(labelsVec, data);
    std::cout << k << "," << scores.first << ","<< scores.second << std::endl;
    // write labels to csv file
    std::ofstream outFile("assign_screen_splinterPair_refine_true.csv");
    if (outFile.is_open()) {
        outFile << "Frame,Cluster\n";
        for (size_t i = 0; i < labels.size(); ++i) {
            outFile << i+1 << "," << labels[i] << "\n";
        }
        outFile.close();
    } else {
        std::cerr << "Unable to open file for writing: test_calc/assign_screen_outlierPair_refine_false.csv" << std::endl;
    }
    return 0;
}


//1321 is where it first diverges