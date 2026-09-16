#include "json_output.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

std::string jsonNumber(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
}

void writeResultJSON(const std::string& filename, const ClusterResult& result) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    file << std::setprecision(10);
    file << "{\n";
    file << "  \"algorithm\": \"" << result.algorithm << "\",\n";
    file << "  \"nFrames\": " << result.nFrames << ",\n";
    file << "  \"nClusters\": " << result.nClusters << ",\n";

    // labels
    file << "  \"labels\": [";
    for (size_t i = 0; i < result.labels.size(); ++i) {
        if (i > 0) file << ", ";
        file << result.labels[i];
    }
    file << "],\n";

    // clusterSizes
    file << "  \"clusterSizes\": [";
    for (size_t i = 0; i < result.clusterSizes.size(); ++i) {
        if (i > 0) file << ", ";
        file << result.clusterSizes[i];
    }
    file << "],\n";

    // representatives
    file << "  \"representatives\": [";
    for (size_t i = 0; i < result.representatives.size(); ++i) {
        if (i > 0) file << ", ";
        file << result.representatives[i];
    }
    file << "],\n";

    // clusterMSD
    file << "  \"clusterMSD\": [";
    for (size_t i = 0; i < result.clusterMSD.size(); ++i) {
        if (i > 0) file << ", ";
        file << jsonNumber(result.clusterMSD[i]);
    }
    file << "],\n";

    // scores
    file << "  \"scores\": {\n";
    file << "    \"calinskiHarabasz\": " << jsonNumber(result.chScore) << ",\n";
    file << "    \"daviesBouldin\": " << jsonNumber(result.dbScore) << "\n";
    file << "  }";

    // zMatrix (HELM only)
    if (!result.zMatrix.empty()) {
        file << ",\n  \"zMatrix\": [\n";
        for (size_t i = 0; i < result.zMatrix.size(); ++i) {
            file << "    [";
            for (size_t j = 0; j < result.zMatrix[i].size(); ++j) {
                if (j > 0) file << ", ";
                file << jsonNumber(result.zMatrix[i][j]);
            }
            file << "]";
            if (i < result.zMatrix.size() - 1) file << ",";
            file << "\n";
        }
        file << "  ]";
    }

    file << "\n}\n";
    file.close();
}
