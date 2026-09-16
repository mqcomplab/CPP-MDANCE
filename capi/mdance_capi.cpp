#include "mdance_capi.h"

#include <string>
#include <vector>
#include <set>
#include <list>
#include <stdexcept>
#include <algorithm>

#include <Eigen/Dense>

#include "../src/tools/types.h"
#include "../src/tools/bts.h"
#include "../src/tools/scores.h"
#include "../src/tools/parse_enums.h"
#include "../src/tools/result_utils.h"
#include "../src/tools/hc_utils.h"
#include "../src/cluster/KMeansRex/KMeans.h"
#ifdef MDANCE_HAS_DIVINE
#include "../src/cluster/divine.h"
#endif
#include "../src/cluster/helm.h"
#include "../src/cluster/equal.h"
#include "../src/cluster/prime.h"

struct mdance_result {
    int nframes = 0;
    int nclusters = 0;
    std::vector<int> labels;
    std::vector<int> cluster_sizes;
    std::vector<int> representatives;
    std::vector<double> cluster_msd;
    double ch_score = 0.0;
    double db_score = 0.0;
    std::vector<double> zmatrix_flat;
    int zmatrix_rows = 0;
    int zmatrix_cols = 0;
    std::string error;
};

struct mdance_analysis {
    double isim = 0.0;
    int nclusters = 0;
    std::vector<double> cluster_isim;
    std::vector<int> cluster_outliers;
    std::string error;
};

struct mdance_prime {
    PrimeResult res;
    std::string error;
};

struct mdance_select {
    std::vector<int> indices;
    std::string error;
};

static ArrayXXd map_coords(const double* coords, int nframes, int ncols) {
    // Every entry point funnels through here, so this is the one place that has
    // to reject a bad buffer. Without it a null or empty coords turns into a
    // segfault inside Eigen -- which, loaded into VMD, takes the whole session
    // down instead of surfacing as an error string the caller can print.
    if (!coords || nframes <= 0 || ncols <= 0) {
        throw std::runtime_error("coordinate matrix is empty or null (need nframes > 0, ncols > 0).");
    }
    // The caller's buffer is row-major (frame-major: each frame's ncols values
    // are contiguous). ArrayXXd is column-major, so map the buffer as row-major
    // and let the conversion re-lay-out storage. Mapping it directly as a
    // column-major ArrayXXd would silently transpose the data and scramble the
    // per-frame rows the algorithms cluster on.
    using RowArrayXXd =
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    return Eigen::Map<const RowArrayXXd>(coords, nframes, ncols);
}

static void fill_common_results(mdance_result* r, const ArrayXXd& data,
                                 const std::vector<int>& labels, int nclusters,
                                 int natoms, MD::Metric metric) {
    r->nframes = data.rows();
    r->nclusters = nclusters;
    r->labels = labels;
    r->cluster_sizes = computeClusterSizes(labels, nclusters);
    r->representatives = computeRepresentatives(data, labels, nclusters, natoms, metric);
    r->cluster_msd = computeClusterMSD(data, labels, nclusters, natoms);

    VectorXi scoreLabels(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        scoreLabels(i) = labels[i];
    }
    r->ch_score = calinskiHarabaszScore(data, scoreLabels);
    r->db_score = daviesBouldinScore(data, scoreLabels);
}

extern "C" {

mdance_result_t* mdance_kmeans(
    const double* coords, int nframes, int ncols,
    int natoms, int nclusters,
    const char* metric,
    const char* kinit,
    int percentage)
{
    auto* r = new mdance_result;
    try {
        // map_coords returns a fresh array; algorithms may mutate it in place.
        ArrayXXd dataCopy = map_coords(coords, nframes, ncols);

        MD::Metric mt = parseMetric(metric ? metric : "MSD");
        MD::KinitType ki = parseKinit(kinit ? kinit : "StratAll");

        KmeansNANI km(dataCopy, nclusters, mt, ki, natoms, percentage);
        Veci kLabels = km.getLabels();

        std::vector<int> labels(kLabels.size());
        for (int i = 0; i < kLabels.size(); ++i) {
            labels[i] = kLabels(i);
        }

        fill_common_results(r, dataCopy, labels, nclusters, natoms, mt);
    } catch (const std::exception& e) {
        r->error = e.what();
    }
    return r;
}

mdance_result_t* mdance_divine(
    const double* coords, int nframes, int ncols,
    int natoms, int nclusters,
    const char* metric,
    const char* split,
    const char* anchors,
    const char* kinit,
    int refine,
    double threshold,
    const char* end_mode,
    int percentage)
{
    auto* r = new mdance_result;
#ifndef MDANCE_HAS_DIVINE
    (void)coords; (void)nframes; (void)ncols; (void)natoms; (void)nclusters;
    (void)metric; (void)split; (void)anchors; (void)kinit; (void)refine;
    (void)threshold; (void)end_mode; (void)percentage;
    r->error = "DIVINE is not available in this build of libmdance "
               "(built without BUILD_DIVINE).";
    return r;
#else
    try {
        ArrayXXd dataCopy = map_coords(coords, nframes, ncols);

        MD::Metric mt = parseMetric(metric ? metric : "MSD");
        MD::DivineSplit sp = parseSplit(split ? split : "WeightedMSD");
        MD::DivineAnchors an = parseAnchors(anchors ? anchors : "NANI");
        MD::KinitType ki = parseKinit(kinit ? kinit : "StratAll");
        int end = (end_mode && std::string(end_mode) == "points") ? 1 : 0;

        Divine div(dataCopy, sp, an, ki, end, nclusters, refine != 0, natoms, threshold, percentage);
        std::vector<int> labels = div.getLabels();

        std::set<int> uniqueLabels(labels.begin(), labels.end());
        int actualK = uniqueLabels.size();

        fill_common_results(r, dataCopy, labels, actualK, natoms, mt);
    } catch (const std::exception& e) {
        r->error = e.what();
    }
    return r;
#endif
}

mdance_result_t* mdance_helm(
    const double* coords, int nframes, int ncols,
    int natoms, int nclusters,
    const int* initial_labels, int nlabels,
    const char* metric,
    const char* merge_scheme,
    float eps,
    int trim_start,
    float min_samples,
    float trim_val,
    float trim_k)
{
    auto* r = new mdance_result;
    try {
        ArrayXXd dataCopy = map_coords(coords, nframes, ncols);

        MD::Metric mt = parseMetric(metric ? metric : "MSD");
        MD::MergeScheme ms = parseMergeScheme(merge_scheme ? merge_scheme : "Inter");

        // Build initial labels as Eigen array
        if (!initial_labels || nlabels <= 0) {
            throw std::runtime_error("HELM requires per-frame initial cluster labels.");
        }
        Eigen::ArrayXi initLabels(nlabels);
        for (int i = 0; i < nlabels; ++i) {
            initLabels(i) = initial_labels[i];
        }

        std::vector<HCTree> clusterTree = buildClusterTree(dataCopy, initLabels);

        Helm helm(clusterTree, natoms, mt, ms, nclusters, eps,
                  trim_start != 0, min_samples, trim_val, trim_k);
        std::vector<HCTree> finalClusters = helm.run();

        // Map the final clusters back to one label per frame.
        int actualK = finalClusters.size();
        std::vector<int> labels =
            labelsFromHelmClusters(finalClusters, initLabels, (int)dataCopy.rows());

        fill_common_results(r, dataCopy, labels, actualK, natoms, mt);

        // Z-matrix
        Mat zMat = helm.getZMatrix();
        if (zMat.rows() > 0) {
            r->zmatrix_rows = zMat.rows();
            r->zmatrix_cols = zMat.cols();
            r->zmatrix_flat.resize(zMat.rows() * zMat.cols());
            for (int i = 0; i < zMat.rows(); ++i) {
                for (int j = 0; j < zMat.cols(); ++j) {
                    r->zmatrix_flat[i * zMat.cols() + j] = zMat(i, j);
                }
            }
        }
    } catch (const std::exception& e) {
        r->error = e.what();
    }
    return r;
}

mdance_result_t* mdance_equal(
    const double* coords, int nframes, int ncols,
    int natoms,
    const char* metric,
    double threshold,
    const char* seed_method,
    double n_seeds,
    int check_sim,
    double sim_threshold,
    int reject_lowd,
    double min_samples,
    int percentage,
    const char* align)
{
    auto* r = new mdance_result;
    try {
        ArrayXXd dataCopy = map_coords(coords, nframes, ncols);

        MD::Metric mt = parseMetric(metric ? metric : "MSD");
        MD::EqualSeed seed = parseEqualSeed(seed_method ? seed_method : "medoid");
        MD::AlignMethod al = parseAlign(align ? align : "none");
        if (al != MD::AlignMethod::None) {
            throw std::runtime_error("align_method uni/kron not implemented in this build; use none.");
        }

        Equal eq(dataCopy, mt, threshold, natoms, seed, n_seeds,
                 check_sim != 0, sim_threshold, reject_lowd != 0, min_samples,
                 percentage, al);
        std::vector<int> labels = eq.getLabels();

        // eQUAL derives K; -1 marks unclustered frames.
        int actualK = 0;
        for (int l : labels) if (l + 1 > actualK) actualK = l + 1;

        fill_common_results(r, dataCopy, labels, actualK, natoms, mt);

        // Recompute CH/DB over CLUSTERED frames only (exclude -1 noise) so the
        // -1 label is not treated as a spurious cluster (matches upstream).
        std::vector<int> keep;
        for (int j = 0; j < (int)labels.size(); ++j) if (labels[j] >= 0) keep.push_back(j);
        if (actualK >= 2 && (int)keep.size() >= 2) {
            ArrayXXd sub((Index)keep.size(), ncols);
            VectorXi subLabels((Index)keep.size());
            for (int i = 0; i < (int)keep.size(); ++i) {
                sub.row(i) = dataCopy.row(keep[i]);
                subLabels(i) = labels[keep[i]];
            }
            r->ch_score = calinskiHarabaszScore(sub, subLabels);
            r->db_score = daviesBouldinScore(sub, subLabels);
        } else {
            r->ch_score = 0.0;
            r->db_score = 0.0;
        }
    } catch (const std::exception& e) {
        r->error = e.what();
    }
    return r;
}

/* --- Extended-similarity (iSIM) analysis --- */

mdance_analysis_t* mdance_analysis(
    const double* coords, int nframes, int ncols,
    int natoms,
    const char* metric,
    const int* labels, int nlabels)
{
    auto* a = new mdance_analysis_t;
    try {
        ArrayXXd dataCopy = map_coords(coords, nframes, ncols);
        MD::Metric mt = parseMetric(metric ? metric : "MSD");

        // Ensemble iSIM over all frames
        a->isim = (dataCopy.rows() >= 2)
            ? extendedComparison(dataCopy, dataCopy.rows(), natoms, false, mt)
            : 0.0;

        // Per-cluster compactness + outlier (only when labels are provided)
        if (labels && nlabels > 0) {
            int nClusters = 0;
            for (int i = 0; i < nlabels; ++i) {
                if (labels[i] + 1 > nClusters) nClusters = labels[i] + 1;
            }
            a->nclusters = nClusters;
            a->cluster_isim.assign(nClusters, 0.0);
            a->cluster_outliers.assign(nClusters, -1);

            for (int c = 0; c < nClusters; ++c) {
                std::vector<int> members;
                for (int j = 0; j < nlabels && j < (int)dataCopy.rows(); ++j) {
                    if (labels[j] == c) members.push_back(j);
                }
                if (members.empty()) continue;
                if (members.size() == 1) {
                    a->cluster_isim[c] = 0.0;
                    a->cluster_outliers[c] = members[0];
                    continue;
                }
                ArrayXXd sub(members.size(), dataCopy.cols());
                for (size_t i = 0; i < members.size(); ++i) {
                    sub.row(i) = dataCopy.row(members[i]);
                }
                a->cluster_isim[c] = extendedComparison(sub, (Index)members.size(), natoms, false, mt);
                Index ol = calculateOutlier(sub, natoms, mt);
                a->cluster_outliers[c] = members[(int)ol];
            }
        }
    } catch (const std::exception& e) {
        a->error = e.what();
    }
    return a;
}

double mdance_analysis_isim(const mdance_analysis_t* a) {
    return a ? a->isim : 0.0;
}

int mdance_analysis_nclusters(const mdance_analysis_t* a) {
    return a ? a->nclusters : 0;
}

const double* mdance_analysis_cluster_isim(const mdance_analysis_t* a) {
    return (a && !a->cluster_isim.empty()) ? a->cluster_isim.data() : nullptr;
}

const int* mdance_analysis_cluster_outliers(const mdance_analysis_t* a) {
    return (a && !a->cluster_outliers.empty()) ? a->cluster_outliers.data() : nullptr;
}

const char* mdance_analysis_error(const mdance_analysis_t* a) {
    return (a && !a->error.empty()) ? a->error.c_str() : nullptr;
}

void mdance_analysis_free(mdance_analysis_t* a) {
    delete a;
}

/* --- PRIME --- */

mdance_prime_t* mdance_prime(
    const double* coords, int nframes, int ncols,
    int natoms,
    const char* metric,
    const int* labels, int nlabels,
    double trim_frac,
    int weighted_by_frames)
{
    (void)natoms;  // RR/SM are count-based; no per-atom normalization
    auto* p = new mdance_prime_t;
    try {
        if (!labels || nlabels <= 0) {
            throw std::runtime_error("PRIME requires per-frame cluster labels.");
        }
        MD::Metric mt = parseMetric(metric ? metric : "RR");
        if (mt != MD::Metric::RR && mt != MD::Metric::SM) {
            throw std::runtime_error("PRIME metric must be RR or SM.");
        }
        ArrayXXd dataCopy = map_coords(coords, nframes, ncols);

        std::vector<int> lab(labels, labels + std::min(nlabels, nframes));
        lab.resize(nframes, -1);

        Prime prime(dataCopy, lab, mt, trim_frac, weighted_by_frames != 0);
        p->res = prime.getResult();
    } catch (const std::exception& e) {
        p->error = e.what();
    }
    return p;
}

int mdance_prime_pairwise(const mdance_prime_t* p)          { return p ? p->res.pairwise : -1; }
int mdance_prime_union(const mdance_prime_t* p)             { return p ? p->res.uni : -1; }
int mdance_prime_medoid(const mdance_prime_t* p)            { return p ? p->res.medoid : -1; }
int mdance_prime_outlier(const mdance_prime_t* p)           { return p ? p->res.outlier : -1; }
int mdance_prime_medoid_all(const mdance_prime_t* p)        { return p ? p->res.medoid_all : -1; }
int mdance_prime_medoid_c0(const mdance_prime_t* p)         { return p ? p->res.medoid_c0 : -1; }
int mdance_prime_medoid_c0_trimmed(const mdance_prime_t* p) { return p ? p->res.medoid_c0_trimmed : -1; }
int mdance_prime_nclusters(const mdance_prime_t* p)         { return p ? p->res.nclusters : 0; }
const char* mdance_prime_error(const mdance_prime_t* p) {
    return (p && !p->error.empty()) ? p->error.c_str() : nullptr;
}
void mdance_prime_free(mdance_prime_t* p) { delete p; }

/* --- Frame-selection tools --- */

mdance_select_t* mdance_select(
    const double* coords, int nframes, int ncols,
    int natoms,
    const char* metric,
    const char* method,
    double param,
    int nbins)
{
    auto* s = new mdance_select_t;
    try {
        ArrayXXd data = map_coords(coords, nframes, ncols);
        MD::Metric mt = parseMetric(metric ? metric : "MSD");
        std::string m = method ? method : "diversity";
        int N = (int)data.rows();

        if (m == "diversity") {
            int pct = (param > 0 && param < 1) ? (int)(param * 100) : (int)param;
            if (pct < 1) pct = 1;
            if (pct > 100) pct = 100;
            std::vector<Index> idx = diversitySelection(data, pct, mt, natoms, true, MD::StartSeed::Medoid);
            for (Index i : idx) s->indices.push_back((int)i);

        } else if (m == "outliers") {
            int n = (param > 0 && param < 1) ? (int)(param * N) : (int)param;
            if (n < 0) n = 0;
            if (n > N) n = N;
            // bts convention: outlier == lowest complementary similarity
            ArrayXd cs = calculateCompSim(data, natoms, mt);
            std::vector<std::pair<double, int>> v;
            v.reserve(N);
            for (int i = 0; i < cs.size(); ++i) v.emplace_back(cs[i], i);
            std::sort(v.begin(), v.end());   // ascending: biggest outliers first
            for (int i = 0; i < n; ++i) s->indices.push_back(v[i].second);

        } else if (m == "repsample") {
            int n = (param > 0 && param < 1) ? (int)(param * N) : (int)param;
            if (n < 1) n = 1;
            ArrayXi idx = repSample(data, mt, natoms, nbins > 0 ? nbins : 10, n, true);
            for (int i = 0; i < idx.size(); ++i) s->indices.push_back(idx(i));

        } else if (m == "medoid") {
            s->indices.push_back((int)calculateMedoid(data, natoms, mt));

        } else if (m == "outlier") {
            s->indices.push_back((int)calculateOutlier(data, natoms, mt));

        } else {
            throw std::runtime_error("Unknown selection method: " + m +
                " (use diversity|outliers|repsample|medoid|outlier).");
        }
    } catch (const std::exception& e) {
        s->error = e.what();
    }
    return s;
}

int mdance_select_count(const mdance_select_t* s) {
    return s ? (int)s->indices.size() : 0;
}
const int* mdance_select_indices(const mdance_select_t* s) {
    return (s && !s->indices.empty()) ? s->indices.data() : nullptr;
}
const char* mdance_select_error(const mdance_select_t* s) {
    return (s && !s->error.empty()) ? s->error.c_str() : nullptr;
}
void mdance_select_free(mdance_select_t* s) { delete s; }

/* --- Accessors --- */

int mdance_result_nframes(const mdance_result_t* r) {
    return r ? r->nframes : 0;
}

int mdance_result_nclusters(const mdance_result_t* r) {
    return r ? r->nclusters : 0;
}

const int* mdance_result_labels(const mdance_result_t* r) {
    return (r && !r->labels.empty()) ? r->labels.data() : nullptr;
}

const int* mdance_result_cluster_sizes(const mdance_result_t* r) {
    return (r && !r->cluster_sizes.empty()) ? r->cluster_sizes.data() : nullptr;
}

const int* mdance_result_representatives(const mdance_result_t* r) {
    return (r && !r->representatives.empty()) ? r->representatives.data() : nullptr;
}

const double* mdance_result_cluster_msd(const mdance_result_t* r) {
    return (r && !r->cluster_msd.empty()) ? r->cluster_msd.data() : nullptr;
}

double mdance_result_ch_score(const mdance_result_t* r) {
    return r ? r->ch_score : 0.0;
}

double mdance_result_db_score(const mdance_result_t* r) {
    return r ? r->db_score : 0.0;
}

const char* mdance_result_error(const mdance_result_t* r) {
    return (r && !r->error.empty()) ? r->error.c_str() : nullptr;
}

int mdance_result_zmatrix_rows(const mdance_result_t* r) {
    return r ? r->zmatrix_rows : 0;
}

int mdance_result_zmatrix_cols(const mdance_result_t* r) {
    return r ? r->zmatrix_cols : 0;
}

const double* mdance_result_zmatrix(const mdance_result_t* r) {
    return (r && !r->zmatrix_flat.empty()) ? r->zmatrix_flat.data() : nullptr;
}

void mdance_result_free(mdance_result_t* r) {
    delete r;
}

} // extern "C"
