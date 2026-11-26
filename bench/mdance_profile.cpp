// mdance_profile: time the phases inside the C API's calls (the algorithm
// itself, then the representatives, cluster MSDs and scores every result
// carries), to see where a call's time goes.
//
//   mdance_profile data.csv natoms [k]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/tools/types.h"
#include "../src/tools/bts.h"
#include "../src/tools/scores.h"
#include "../src/tools/result_utils.h"
#include "../src/cluster/KMeansRex/KMeans.h"

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s data.csv natoms [k]\n", argv[0]); return 2; }
    int na = std::atoi(argv[2]);
    int k = argc > 3 ? std::atoi(argv[3]) : 10;
    std::ifstream in(argv[1]);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<double> r;
        std::stringstream ss(line);
        std::string v;
        while (std::getline(ss, v, ',')) r.push_back(std::stod(v));
        rows.push_back(std::move(r));
    }
    ArrayXXd X(rows.size(), rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j) X(i, j) = rows[i][j];
    std::printf("%td frames x %td columns\n", X.rows(), X.cols());

    auto t = std::chrono::steady_clock::now();
    ArrayXd cs = calculateCompSim(X, na, MD::Metric::MSD);
    std::printf("compSim                %9.1f ms\n", ms_since(t));

    t = std::chrono::steady_clock::now();
    std::vector<Index> div = diversitySelection(X, 10, MD::Metric::MSD, na, true);
    std::printf("diversity 10%% compsim  %9.1f ms (%zu)\n", ms_since(t), div.size());

    t = std::chrono::steady_clock::now();
    KmeansNANI km(X, k, MD::Metric::MSD, MD::KinitType::StratAll, na, 10);
    std::printf("KmeansNANI StratAll    %9.1f ms\n", ms_since(t));
    Veci l = km.getLabels();
    std::vector<int> labels(l.data(), l.data() + l.size());

    t = std::chrono::steady_clock::now();
    auto reps = computeRepresentatives(X, labels, k, na, MD::Metric::MSD);
    std::printf("representatives        %9.1f ms\n", ms_since(t));
    t = std::chrono::steady_clock::now();
    auto msd = computeClusterMSD(X, labels, k, na);
    std::printf("cluster MSD            %9.1f ms\n", ms_since(t));
    VectorXi vl = l.matrix();
    t = std::chrono::steady_clock::now();
    double ch = calinskiHarabaszScore(X, vl);
    std::printf("CH                     %9.1f ms\n", ms_since(t));
    t = std::chrono::steady_clock::now();
    double db = daviesBouldinScore(X, vl);
    std::printf("DB                     %9.1f ms  (ch %g db %g)\n", ms_since(t), ch, db);
    return 0;
}
