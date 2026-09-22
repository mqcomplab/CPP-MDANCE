#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <map>
#include <set>
#include <list>
#include <stdexcept>
#include <algorithm>
#include <vector>

#include "csv_io.h"
#include "json_output.h"

#include "../src/tools/types.h"
#include "../src/tools/bts.h"
#include "../src/tools/scores.h"
#include "../src/tools/parse_enums.h"
#include "../src/tools/result_utils.h"
#include "../src/cluster/KMeansRex/KMeans.h"
#ifdef MDANCE_HAS_DIVINE
#include "../src/cluster/divine.h"
#endif
#include "../src/cluster/helm.h"
#include "../src/cluster/equal.h"
#include "../src/cluster/prime.h"

// --- Argument parser ---

class Args {
    std::map<std::string, std::string> opts;
    std::set<std::string> flags;

public:
    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.substr(0, 2) == "--") {
                std::string key = arg.substr(2);
                // Check if it's a boolean flag (no value follows)
                if (key == "refine" || key == "trim-start" || key == "analysis" ||
                    key == "check-sim" || key == "reject-lowd" ||
                    key == "prime" || key == "weighted" || key == "select") {
                    flags.insert(key);
                } else if (i + 1 < argc) {
                    opts[key] = argv[++i];
                } else {
                    throw std::runtime_error("Missing value for --" + key);
                }
            }
        }
    }

    std::string get(const std::string& key) const {
        auto it = opts.find(key);
        if (it == opts.end()) throw std::runtime_error("Missing required argument: --" + key);
        return it->second;
    }

    std::string get(const std::string& key, const std::string& defaultVal) const {
        auto it = opts.find(key);
        return it != opts.end() ? it->second : defaultVal;
    }

    int getInt(const std::string& key, int defaultVal) const {
        auto it = opts.find(key);
        return it != opts.end() ? std::stoi(it->second) : defaultVal;
    }

    double getDouble(const std::string& key, double defaultVal) const {
        auto it = opts.find(key);
        return it != opts.end() ? std::stod(it->second) : defaultVal;
    }

    float getFloat(const std::string& key, float defaultVal) const {
        auto it = opts.find(key);
        return it != opts.end() ? std::stof(it->second) : defaultVal;
    }

    bool hasFlag(const std::string& key) const {
        return flags.count(key) > 0;
    }

    bool has(const std::string& key) const {
        return opts.count(key) > 0;
    }
};

// --- Print usage ---

void printUsage() {
    std::cerr << "Usage: mdance-cli --algorithm {kmeans|"
#ifdef MDANCE_HAS_DIVINE
                 "divine|"
#endif
                 "helm|equal}\n"
              << "                  --input <csv-path>\n"
              << "                  --output <json-path>\n"
              << "                  --natoms <int>\n"
              << "                  --nclusters <int>\n"
              << "                  [--metric {MSD|BUB|Fai|...}]  (default: MSD)\n"
              << "\n"
              << "KMeans options:\n"
              << "  --kinit {StratAll|StratReduced|CompSim|DivSelect|KmeansPP|Random|VanillaKmeansPP}\n"
              << "  --percentage <int>  (default: 10)\n"
              << "\n"
#ifdef MDANCE_HAS_DIVINE
              << "DIVINE options:\n"
              << "  --split {MSD|Radius|WeightedMSD}\n"
              << "  --anchors {NANI|OutlierPair|SplinterPair}\n"
              << "  --kinit {StratAll|...}\n"
              << "  --refine  (flag)\n"
              << "  --threshold <float>  (default: 0)\n"
              << "  --end-mode {k|points}  (default: k)\n"
              << "  --percentage <int>  (default: 10)\n"
              << "\n"
#endif
              << "HELM options:\n"
              << "  --initial-labels <csv-path>  (required: initial cluster labels)\n"
              << "  --merge-scheme {Intra|Inter|Half}  (default: Inter)\n"
              << "  --eps <float>  (default: -1, meaning none)\n"
              << "  --trim-start  (flag)\n"
              << "  --min-samples <float>  (default: 0.01)\n"
              << "  --trim-val <float>  (default: 0)\n"
              << "  --trim-k <float>  (default: 0)\n"
              << "\n"
              << "eQUAL options (cluster count emerges from --threshold; --nclusters ignored):\n"
              << "  --threshold <float>  (REQUIRED) radial MSD/extended-comparison cutoff\n"
              << "  --seed-method {comp_sim|medoid}  (default: medoid)\n"
              << "  --n-seeds <float>  (default: 1; (0,1)=fraction of frames, >=1=count)\n"
              << "  --percentage <int>  (default: 10; comp_sim high-density region)\n"
              << "  --check-sim  (flag)  reject+terminate when winner intra-sim > --sim-threshold\n"
              << "  --sim-threshold <float>  (required with --check-sim)\n"
              << "  --reject-lowd  (flag)  drop winners smaller than --min-samples\n"
              << "  --min-samples <float>  (default: 10; (0,1)=fraction, >=1=count)\n"
              << "  --align {none|uni|kron}  (default: none; uni/kron unimplemented)\n"
              << "\n"
              << "Analysis mode (no clustering):\n"
              << "  --analysis            compute extended-similarity (iSIM)\n"
              << "  --input <csv-path>    coordinate matrix\n"
              << "  --output <json-path>  results\n"
              << "  --natoms <int>\n"
              << "  --metric {MSD|...}    (default: MSD)\n"
              << "  --labels <csv-path>   optional per-frame cluster labels for\n"
              << "                        per-cluster compactness + outlier frames\n"
              << "\n"
              << "PRIME mode (representative/\"native\" frame prediction):\n"
              << "  --prime               predict representative frame(s)\n"
              << "  --input <csv-path>    coordinate matrix\n"
              << "  --labels <csv-path>   per-frame cluster labels (REQUIRED)\n"
              << "  --output <json-path>  7 predicted frame indices\n"
              << "  --natoms <int>\n"
              << "  --metric {RR|SM}      (default: RR)\n"
              << "  --trim-frac <float>   (default: 0.0; e.g. 0.1)\n"
              << "  --weighted            (flag) weight per-cluster score by population\n"
              << "\n"
              << "Frame-selection mode (no clustering):\n"
              << "  --select              select representative/diverse/outlier frames\n"
              << "  --input <csv-path>    coordinate matrix\n"
              << "  --output <json-path>  selected frame indices\n"
              << "  --natoms <int>\n"
              << "  --metric {MSD|...}    (default: MSD)\n"
              << "  --method {diversity|outliers|repsample|medoid|outlier}  (default: diversity)\n"
              << "  --param <float>       diversity: percentage; outliers/repsample: count(>=1) or fraction(0,1)\n"
              << "  --nbins <int>         (default: 10; repsample only)\n";
}

// --- Main ---

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    try {
        Args args;
        args.parse(argc, argv);

        // --- Extended-similarity (iSIM) analysis mode ---
        // Must be handled before args.get("algorithm") below, which throws when
        // --algorithm is absent (an analysis invocation has no --algorithm).
        if (args.hasFlag("analysis")) {
            std::string inputPath = args.get("input");
            std::string outputPath = args.get("output");
            int nAtoms = args.getInt("natoms", 1);
            std::string metricStr = args.get("metric", "MSD");
            MD::Metric mt = parseMetric(metricStr);

            std::cerr << "Reading input: " << inputPath << std::endl;
            ArrayXXd data = readCSV(inputPath);
            std::cerr << "Data: " << data.rows() << " frames x " << data.cols() << " features" << std::endl;

            std::vector<int> labels;
            if (args.has("labels")) {
                ArrayXXd ld = readCSV(args.get("labels"));
                ArrayXi lab = (ld.cols() == 1) ? ld.col(0).cast<int>()
                                               : ld.col(ld.cols() - 1).cast<int>();
                labels.assign(lab.data(), lab.data() + lab.size());
            }

            double isimAll = (data.rows() >= 2)
                ? extendedComparison(data, data.rows(), nAtoms, false, mt) : 0.0;

            int nClusters = 0;
            for (int l : labels) if (l + 1 > nClusters) nClusters = l + 1;
            std::vector<double> clusterISIM(nClusters, 0.0);
            std::vector<int> clusterOutliers(nClusters, -1);
            for (int c = 0; c < nClusters; ++c) {
                std::vector<int> mem;
                for (size_t j = 0; j < labels.size() && j < (size_t)data.rows(); ++j)
                    if (labels[j] == c) mem.push_back((int)j);
                if (mem.empty()) continue;
                if (mem.size() == 1) { clusterISIM[c] = 0.0; clusterOutliers[c] = mem[0]; continue; }
                ArrayXXd sub(mem.size(), data.cols());
                for (size_t i = 0; i < mem.size(); ++i) sub.row(i) = data.row(mem[i]);
                clusterISIM[c] = extendedComparison(sub, (Index)mem.size(), nAtoms, false, mt);
                Index ol = calculateOutlier(sub, nAtoms, mt);
                clusterOutliers[c] = mem[(int)ol];
            }

            std::cerr << "Writing output: " << outputPath << std::endl;
            std::ofstream f(outputPath);
            if (!f.is_open()) throw std::runtime_error("Could not open output for writing: " + outputPath);
            f << std::setprecision(10);
            f << "{\n";
            f << "  \"analysis\": \"isim\",\n";
            f << "  \"metric\": \"" << metricStr << "\",\n";
            f << "  \"isim\": " << jsonNumber(isimAll) << ",\n";
            f << "  \"clusterISIM\": [";
            for (size_t i = 0; i < clusterISIM.size(); ++i) { if (i) f << ", "; f << jsonNumber(clusterISIM[i]); }
            f << "],\n";
            f << "  \"clusterOutliers\": [";
            for (size_t i = 0; i < clusterOutliers.size(); ++i) { if (i) f << ", "; f << clusterOutliers[i]; }
            f << "]\n";
            f << "}\n";
            f.close();

            std::cerr << "Done. iSIM=" << isimAll << " (metric " << metricStr
                      << ") over " << data.rows() << " frames." << std::endl;
            return 0;
        }

        // --- PRIME mode: predict the representative/"native" frame ---
        if (args.hasFlag("prime")) {
            std::string inputPath = args.get("input");
            std::string outputPath = args.get("output");
            int nAtoms = args.getInt("natoms", 1);
            std::string metricStr = args.get("metric", "RR");
            MD::Metric mt = parseMetric(metricStr);
            if (mt != MD::Metric::RR && mt != MD::Metric::SM) {
                throw std::runtime_error("PRIME metric must be RR or SM.");
            }
            if (!args.has("labels")) {
                throw std::runtime_error("PRIME requires --labels (a cluster partition).");
            }
            double trimFrac = args.getDouble("trim-frac", 0.0);
            bool weighted = args.hasFlag("weighted");

            std::cerr << "Reading input: " << inputPath << std::endl;
            ArrayXXd data = readCSV(inputPath);
            ArrayXXd ld = readCSV(args.get("labels"));
            ArrayXi lab = (ld.cols() == 1) ? ld.col(0).cast<int>()
                                           : ld.col(ld.cols() - 1).cast<int>();
            std::vector<int> labels(lab.data(), lab.data() + lab.size());
            labels.resize(data.rows(), -1);

            std::cerr << "Running PRIME (metric=" << metricStr << ", trim=" << trimFrac
                      << ", weighted=" << (weighted ? "true" : "false") << ")..." << std::endl;
            Prime prime(data, labels, mt, trimFrac, weighted);
            PrimeResult pr = prime.getResult();

            std::ofstream f(outputPath);
            if (!f.is_open()) throw std::runtime_error("Could not open output for writing: " + outputPath);
            f << "{\n";
            f << "  \"analysis\": \"prime\",\n";
            f << "  \"metric\": \"" << metricStr << "\",\n";
            f << "  \"trimFrac\": " << jsonNumber(trimFrac) << ",\n";
            f << "  \"weighted\": " << (weighted ? "true" : "false") << ",\n";
            f << "  \"nClusters\": " << pr.nclusters << ",\n";
            f << "  \"pairwise\": " << pr.pairwise << ",\n";
            f << "  \"union\": " << pr.uni << ",\n";
            f << "  \"medoid\": " << pr.medoid << ",\n";
            f << "  \"outlier\": " << pr.outlier << ",\n";
            f << "  \"medoidAll\": " << pr.medoid_all << ",\n";
            f << "  \"medoidC0\": " << pr.medoid_c0 << ",\n";
            f << "  \"medoidC0Trimmed\": " << pr.medoid_c0_trimmed << "\n";
            f << "}\n";
            f.close();

            std::cerr << "PRIME done. pairwise=" << pr.pairwise << " union=" << pr.uni
                      << " medoid=" << pr.medoid << " outlier=" << pr.outlier << std::endl;
            return 0;
        }

        // --- Frame-selection tools (no clustering) ---
        if (args.hasFlag("select")) {
            std::string inputPath = args.get("input");
            std::string outputPath = args.get("output");
            int nAtoms = args.getInt("natoms", 1);
            std::string metricStr = args.get("metric", "MSD");
            MD::Metric mt = parseMetric(metricStr);
            std::string method = args.get("method", "diversity");
            double param = args.getDouble("param", 10);
            int nbins = args.getInt("nbins", 10);

            std::cerr << "Reading input: " << inputPath << std::endl;
            ArrayXXd data = readCSV(inputPath);
            int N = (int)data.rows();
            std::vector<int> indices;

            if (method == "diversity") {
                int pct = (param > 0 && param < 1) ? (int)(param * 100) : (int)param;
                if (pct < 1) pct = 1;
                if (pct > 100) pct = 100;
                std::vector<Index> idx = diversitySelection(data, pct, mt, nAtoms, true, MD::StartSeed::Medoid);
                for (Index i : idx) indices.push_back((int)i);
            } else if (method == "outliers") {
                int n = (param > 0 && param < 1) ? (int)(param * N) : (int)param;
                if (n < 0) n = 0;
                if (n > N) n = N;
                ArrayXd cs = calculateCompSim(data, nAtoms, mt);
                std::vector<std::pair<double, int>> v;
                for (int i = 0; i < cs.size(); ++i) v.emplace_back(cs[i], i);
                std::sort(v.begin(), v.end());
                for (int i = 0; i < n; ++i) indices.push_back(v[i].second);
            } else if (method == "repsample") {
                int n = (param > 0 && param < 1) ? (int)(param * N) : (int)param;
                if (n < 1) n = 1;
                ArrayXi idx = repSample(data, mt, nAtoms, nbins > 0 ? nbins : 10, n, true);
                for (int i = 0; i < idx.size(); ++i) indices.push_back(idx(i));
            } else if (method == "medoid") {
                indices.push_back((int)calculateMedoid(data, nAtoms, mt));
            } else if (method == "outlier") {
                indices.push_back((int)calculateOutlier(data, nAtoms, mt));
            } else {
                throw std::runtime_error("Unknown --method: " + method +
                    " (use diversity|outliers|repsample|medoid|outlier).");
            }

            std::cerr << "Writing output: " << outputPath << std::endl;
            std::ofstream f(outputPath);
            if (!f.is_open()) throw std::runtime_error("Could not open output for writing: " + outputPath);
            f << "{\n";
            f << "  \"analysis\": \"select\",\n";
            f << "  \"method\": \"" << method << "\",\n";
            f << "  \"metric\": \"" << metricStr << "\",\n";
            f << "  \"count\": " << indices.size() << ",\n";
            f << "  \"indices\": [";
            for (size_t i = 0; i < indices.size(); ++i) { if (i) f << ", "; f << indices[i]; }
            f << "]\n";
            f << "}\n";
            f.close();

            std::cerr << "select(" << method << ") done. " << indices.size()
                      << " frame(s) selected." << std::endl;
            return 0;
        }

        std::string algorithm = args.get("algorithm");
        std::string inputPath = args.get("input");
        std::string outputPath = args.get("output");
        int nAtoms = args.getInt("natoms", 1);
        int nClusters = args.getInt("nclusters", 10);
        std::string metricStr = args.get("metric", "MSD");
        MD::Metric metric = parseMetric(metricStr);

        std::cerr << "Reading input: " << inputPath << std::endl;
        ArrayXXd data = readCSV(inputPath);
        std::cerr << "Data: " << data.rows() << " frames x " << data.cols() << " features" << std::endl;

        ClusterResult result;
        result.algorithm = algorithm;
        result.nFrames = data.rows();

        if (algorithm == "kmeans") {
            MD::KinitType kinit = parseKinit(args.get("kinit", "StratAll"));
            int percentage = args.getInt("percentage", 10);

            std::cerr << "Running KMeans NANI (k=" << nClusters << ")..." << std::endl;
            KmeansNANI km(data, nClusters, metric, kinit, nAtoms, percentage);
            Veci kLabels = km.getLabels();
            auto scores = km.computeScores();

            // Convert labels
            result.labels.resize(kLabels.size());
            for (int i = 0; i < kLabels.size(); ++i) {
                result.labels[i] = kLabels(i);
            }
            result.nClusters = nClusters;
            result.chScore = scores.first;
            result.dbScore = scores.second;

        } else if (algorithm == "divine") {
#ifndef MDANCE_HAS_DIVINE
            throw std::runtime_error("Algorithm 'divine' is not available in this "
                "build of mdance-cli (built without BUILD_DIVINE).");
#else
            MD::DivineSplit splitType = parseSplit(args.get("split", "WeightedMSD"));
            MD::DivineAnchors anchorType = parseAnchors(args.get("anchors", "NANI"));
            MD::KinitType kinit = parseKinit(args.get("kinit", "StratAll"));
            bool refine = args.hasFlag("refine");
            double threshold = args.getDouble("threshold", 0);
            int percentage = args.getInt("percentage", 10);
            std::string endMode = args.get("end-mode", "k");
            int end = (endMode == "points") ? 1 : 0;

            std::cerr << "Running DIVINE (k=" << nClusters << ")..." << std::endl;
            Divine div(data, splitType, anchorType, kinit, end, nClusters, refine, nAtoms, threshold, percentage);
            result.labels = div.getLabels();

            // Determine actual number of clusters from labels
            std::set<int> uniqueLabels(result.labels.begin(), result.labels.end());
            result.nClusters = uniqueLabels.size();

            // Compute scores
            VectorXi scoreLabels(result.labels.size());
            for (size_t i = 0; i < result.labels.size(); ++i) {
                scoreLabels(i) = result.labels[i];
            }
            result.chScore = calinskiHarabaszScore(data, scoreLabels);
            result.dbScore = daviesBouldinScore(data, scoreLabels);
#endif

        } else if (algorithm == "helm") {
            MD::MergeScheme mergeScheme = parseMergeScheme(args.get("merge-scheme", "Inter"));
            float eps = args.getFloat("eps", -1);
            bool trimStart = args.hasFlag("trim-start");
            float minSamples = args.getFloat("min-samples", 0.01f);
            float trimVal = args.getFloat("trim-val", 0);
            float trimK = args.getFloat("trim-k", 0);

            // Read initial labels
            std::string initialLabelsPath = args.get("initial-labels");
            std::cerr << "Reading initial labels: " << initialLabelsPath << std::endl;
            ArrayXXd labelData = readCSV(initialLabelsPath);

            // Labels can be a single column or second column (frame_id, label format)
            Eigen::ArrayXi initLabels;
            if (labelData.cols() == 1) {
                initLabels = labelData.col(0).cast<int>();
            } else {
                initLabels = labelData.col(labelData.cols() - 1).cast<int>();
            }

            // Build HCTree from labels
            std::cerr << "Building cluster tree..." << std::endl;
            std::vector<HCTree> clusterTree = buildClusterTree(data, initLabels);

            std::cerr << "Running HELM (target clusters=" << nClusters
                      << ", initial clusters=" << clusterTree.size() << ")..." << std::endl;
            Helm helm(clusterTree, nAtoms, metric, mergeScheme, nClusters, eps,
                      trimStart, minSamples, trimVal, trimK);
            std::vector<HCTree> finalClusters = helm.run();

            // Map the final clusters back to one label per frame.
            result.nClusters = finalClusters.size();
            result.labels = labelsFromHelmClusters(finalClusters, initLabels, (int)data.rows());

            // Compute scores
            VectorXi scoreLabels(result.labels.size());
            for (size_t i = 0; i < result.labels.size(); ++i) {
                scoreLabels(i) = result.labels[i];
            }
            result.chScore = calinskiHarabaszScore(data, scoreLabels);
            result.dbScore = daviesBouldinScore(data, scoreLabels);

            // Z-matrix
            Mat zMat = helm.getZMatrix();
            if (zMat.rows() > 0) {
                result.zMatrix.resize(zMat.rows());
                for (int i = 0; i < zMat.rows(); ++i) {
                    result.zMatrix[i].resize(zMat.cols());
                    for (int j = 0; j < zMat.cols(); ++j) {
                        result.zMatrix[i][j] = zMat(i, j);
                    }
                }
            }

        } else if (algorithm == "equal") {
            double threshold = args.getDouble("threshold", -1);
            if (threshold < 0) {
                throw std::runtime_error("eQUAL requires --threshold (radial cutoff).");
            }
            MD::EqualSeed seed = parseEqualSeed(args.get("seed-method", "medoid"));
            double nSeeds = args.getDouble("n-seeds", 1);
            bool checkSim = args.hasFlag("check-sim");
            double simThreshold = args.getDouble("sim-threshold", 0);
            bool rejectLowd = args.hasFlag("reject-lowd");
            double minSamples = args.getDouble("min-samples", 10);
            int percentage = args.getInt("percentage", 10);
            MD::AlignMethod al = parseAlign(args.get("align", "none"));
            if (al != MD::AlignMethod::None) {
                throw std::runtime_error("align_method uni/kron not implemented in this build; use none.");
            }

            std::cerr << "Running eQUAL (threshold=" << threshold << ", metric=" << metricStr << ")..." << std::endl;
            Equal eq(data, metric, threshold, nAtoms, seed, nSeeds, checkSim,
                     simThreshold, rejectLowd, minSamples, percentage, al);
            result.labels = eq.getLabels();

            int actualK = 0;
            for (int l : result.labels) if (l + 1 > actualK) actualK = l + 1;
            result.nClusters = actualK;

            // Scores over clustered frames only (exclude -1 noise)
            std::vector<int> keep;
            for (int j = 0; j < (int)result.labels.size(); ++j)
                if (result.labels[j] >= 0) keep.push_back(j);
            if (actualK >= 2 && (int)keep.size() >= 2) {
                ArrayXXd sub((Index)keep.size(), data.cols());
                VectorXi subLabels((Index)keep.size());
                for (int i = 0; i < (int)keep.size(); ++i) {
                    sub.row(i) = data.row(keep[i]);
                    subLabels(i) = result.labels[keep[i]];
                }
                result.chScore = calinskiHarabaszScore(sub, subLabels);
                result.dbScore = daviesBouldinScore(sub, subLabels);
            } else {
                result.chScore = 0.0;
                result.dbScore = 0.0;
            }
            std::cerr << "eQUAL found " << actualK << " clusters (" << keep.size()
                      << "/" << result.labels.size() << " frames clustered)." << std::endl;

        } else {
            throw std::runtime_error("Unknown algorithm: " + algorithm +
                ". Must be one of: kmeans, "
#ifdef MDANCE_HAS_DIVINE
                "divine, "
#endif
                "helm, equal");
        }

        // Compute cluster sizes and representatives
        result.clusterSizes = computeClusterSizes(result.labels, result.nClusters);
        result.representatives = computeRepresentatives(data, result.labels, result.nClusters, nAtoms, metric);
        result.clusterMSD = computeClusterMSD(data, result.labels, result.nClusters, nAtoms);

        // Write output
        std::cerr << "Writing output: " << outputPath << std::endl;
        writeResultJSON(outputPath, result);

        std::cerr << "Done. " << result.nClusters << " clusters found." << std::endl;
        std::cerr << "  Calinski-Harabasz: " << result.chScore << std::endl;
        std::cerr << "  Davies-Bouldin: " << result.dbScore << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
