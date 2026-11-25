// libmdance's C interface: see mdance_capi.h.
//
// Every entry point checks its arguments, runs inside a try block that turns
// any exception (std::bad_alloc included) into the call's error, and never
// writes to the console: the library is loaded into other programs (VMD),
// where an abort, an exit or an uncaught exception ends the user's session.
//
// Both interfaces run through the same code: the first one's functions make
// a dataset and options from their arguments and call the second's.

#include "mdance_capi.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "../src/tools/types.h"
#include "../src/tools/control.h"
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

// ---------------------------------------------------------------- objects

struct mdance_result {
    int nframes = 0;
    int nclusters = 0;
    int seed = -1;
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

struct mdance_dataset {
    int nframes = 0;
    int ncols = 0;
    int natoms = 0;
    ArrayXXd X;                 // nframes x ncols, column-major (the algorithms' layout)
    std::vector<char> isSet;
    int nset = 0;
};

struct mdance_options {
    std::map<std::string, std::string> text;
    std::map<std::string, std::vector<int>> lists;
};

struct mdance_control : MD::Control {};

namespace {

thread_local std::string lastError;

int fail(const std::string& msg) {
    lastError = msg;
    return -1;
}

const char* const kOutOfMemory = "out of memory";

// The message of the exception being handled (call inside a catch).
std::string currentMessage() {
    try {
        throw;
    } catch (const MD::Cancelled&) {
        return "cancelled";
    } catch (const std::bad_alloc&) {
        return kOutOfMemory;
    } catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "unknown error";
    }
}

// Run body(*obj) on a new T, its error set from any exception. NULL only when
// the T itself cannot be allocated.
template <class T, class F>
T* guarded(mdance_control_t* ctl, F body) {
    T* obj = new (std::nothrow) T;
    if (!obj) return nullptr;
    try {
        MD::ControlScope scope(ctl);
        body(*obj);
    } catch (...) {
        obj->error = currentMessage();
        if (obj->error.empty()) obj->error = "error";
    }
    return obj;
}

// ---------------------------------------------------------------- numbers

bool parseWhole(const std::string& s, long long& out) {
    if (s.empty() || s.size() > 30) return false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') i = 1;
    if (i >= s.size()) return false;
    for (size_t j = i; j < s.size(); ++j) {
        if (s[j] < '0' || s[j] > '9') return false;
    }
    errno = 0;
    char* end = nullptr;
    long long v = std::strtoll(s.c_str(), &end, 10);   // base 10: "010" is 10
    if (errno == ERANGE || *end) return false;
    out = v;
    return true;
}

bool parseNumber(const std::string& s, double& out) {
    if (s.empty() || s.size() > 64) return false;
    // digits, a sign, a point and an exponent only: strtod would also read
    // "0x1F", "inf", "nan" and leading spaces
    for (char ch : s) {
        if (!((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E'))
            return false;
    }
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (*end || !std::isfinite(v)) return false;
    out = v;
    return true;
}

std::string fmt(double v) {
    char b[40];
    std::snprintf(b, sizeof b, "%.17g", v);
    return b;
}

// ---------------------------------------------------------------- options

// Reads a call's options, checks each, and at the end refuses any it did not
// read: a misspelt option is an error, never a default silently used.
class Reader {
    const mdance_options* o;
    std::string what;
    std::set<std::string> known;

    std::runtime_error bad(const std::string& name, const std::string& why) const {
        return std::runtime_error(what + ": option " + name + ": " + why);
    }
    const std::string* get(const std::string& name) {
        known.insert(name);
        if (!o) return nullptr;
        auto it = o->text.find(name);
        return it == o->text.end() ? nullptr : &it->second;
    }

public:
    Reader(const mdance_options* o, std::string what) : o(o), what(std::move(what)) {}

    std::string choice(const std::string& name, const std::string& def,
                       std::initializer_list<const char*> allowed) {
        const std::string* v = get(name);
        std::string s = v ? *v : def;
        std::string all;
        for (const char* a : allowed) {
            if (s == a) return s;
            all += all.empty() ? "" : " ";
            all += a;
        }
        throw bad(name, "\"" + s + "\" is not one of: " + all);
    }

    long long whole(const std::string& name, long long def, long long lo, long long hi, bool required = false) {
        const std::string* v = get(name);
        if (!v) {
            if (required) throw std::runtime_error(what + ": option " + name + " is required");
            return def;
        }
        long long x;
        if (!parseWhole(*v, x)) throw bad(name, "\"" + *v + "\" is not a whole number");
        if (x < lo || x > hi)
            throw bad(name, *v + " is outside " + std::to_string(lo) + " .. " + std::to_string(hi));
        return x;
    }

    double number(const std::string& name, double def, double lo, double hi, bool required = false) {
        const std::string* v = get(name);
        if (!v) {
            if (required) throw std::runtime_error(what + ": option " + name + " is required");
            return def;
        }
        double x;
        if (!parseNumber(*v, x)) throw bad(name, "\"" + *v + "\" is not a finite number");
        if (x < lo || x > hi) throw bad(name, *v + " is outside " + fmt(lo) + " .. " + fmt(hi));
        return x;
    }

    bool flag(const std::string& name, bool def) {
        const std::string* v = get(name);
        if (!v) return def;
        if (*v == "1" || *v == "true" || *v == "yes" || *v == "on") return true;
        if (*v == "0" || *v == "false" || *v == "no" || *v == "off") return false;
        throw bad(name, "\"" + *v + "\" is not 0 or 1");
    }

    // a list of whole numbers, one per frame when perFrame (else any length)
    const std::vector<int>* list(const std::string& name, int nframes, bool required,
                                 int lo, int hi) {
        known.insert(name);
        const std::vector<int>* l = nullptr;
        if (o) {
            auto it = o->lists.find(name);
            if (it != o->lists.end()) l = &it->second;
            else if (o->text.count(name)) throw bad(name, "must be a list of whole numbers");
        }
        if (!l) {
            if (required) throw std::runtime_error(what + ": option " + name + " is required");
            return nullptr;
        }
        if ((int)l->size() != nframes)
            throw bad(name, std::to_string(l->size()) + " values for " + std::to_string(nframes) + " frames");
        for (size_t i = 0; i < l->size(); ++i) {
            if ((*l)[i] < lo || (*l)[i] > hi)
                throw bad(name, "value " + std::to_string((*l)[i]) + " (frame " + std::to_string(i) +
                                    ") is outside " + std::to_string(lo) + " .. " + std::to_string(hi));
        }
        return l;
    }

    void finish() const {
        if (!o) return;
        std::string unknown;
        for (auto& kv : o->text) if (!known.count(kv.first)) unknown += (unknown.empty() ? "" : ", ") + kv.first;
        for (auto& kv : o->lists) if (!known.count(kv.first)) unknown += (unknown.empty() ? "" : ", ") + kv.first;
        if (!unknown.empty()) {
            std::string all;
            for (auto& k : known) all += (all.empty() ? "" : " ") + k;
            throw std::runtime_error(what + ": unknown option " + unknown + " (the options: " + all + ")");
        }
    }
};

#define METRICS {"MSD", "BUB", "Fai", "Gle", "Ja", "JT", "RT", "RR", "SM", "SS1", "SS2"}
#define KINITS {"StratAll", "StratReduced", "CompSim", "DivSelect", "KmeansPP", "Random", "VanillaKmeansPP"}

// a C float parameter (HELM's): 0, or within float's range
float asFloat(const std::string& what, double v) {
    if (v != 0 && (std::fabs(v) > 3.4e38 || std::fabs(v) < 1.2e-38))
        throw std::runtime_error(what + ": " + fmt(v) + " is not 0 nor within a float's range (1.2e-38 .. 3.4e38)");
    return (float)v;
}

// ---------------------------------------------------------------- datasets

const mdance_dataset& ready(const mdance_dataset_t* d) {
    if (!d) throw std::invalid_argument("no dataset (NULL)");
    if (d->nset < d->nframes) {
        int f = 0;
        while (f < d->nframes && d->isSet[f]) ++f;
        throw std::runtime_error("frame " + std::to_string(f) + " of the dataset was never set (" +
                                 std::to_string(d->nset) + " of " + std::to_string(d->nframes) + " are)");
    }
    return *d;
}

mdance_dataset_t* newDataset(int nframes, int ncols, int natoms) {
    if (nframes <= 0) { fail("nframes must be positive"); return nullptr; }
    if (ncols <= 0) { fail("ncols must be positive"); return nullptr; }
    if (natoms <= 0) { fail("natoms must be positive"); return nullptr; }
    mdance_dataset_t* d = new (std::nothrow) mdance_dataset_t;
    if (!d) { fail(kOutOfMemory); return nullptr; }
    try {
        d->nframes = nframes;
        d->ncols = ncols;
        d->natoms = natoms;
        d->X.resize(nframes, ncols);
        d->isSet.assign(nframes, 0);
    } catch (...) {
        delete d;
        fail(kOutOfMemory + std::string(" (") + std::to_string(nframes) + " x " + std::to_string(ncols) + " values)");
        return nullptr;
    }
    return d;
}

template <class T>
int setFrame(mdance_dataset_t* d, int frame, const T* values, int n) {
    if (!d) return fail("no dataset (NULL)");
    if (!values) return fail("no values (NULL)");
    if (frame < 0 || frame >= d->nframes)
        return fail("frame " + std::to_string(frame) + " is outside 0 .. " + std::to_string(d->nframes - 1));
    if (n != d->ncols)
        return fail(std::to_string(n) + " values for a frame of " + std::to_string(d->ncols));
    for (int j = 0; j < n; ++j) {
        double v = (double)values[j];
        if (!std::isfinite(v))
            return fail("frame " + std::to_string(frame) + ", column " + std::to_string(j) +
                        " (atom " + std::to_string(j / 3) + "): " + fmt(v) + " is not a finite number");
    }
    for (int j = 0; j < n; ++j) d->X(frame, j) = (double)values[j];
    if (!d->isSet[frame]) { d->isSet[frame] = 1; d->nset++; }
    return 0;
}

// ---------------------------------------------------------------- DCD

uint32_t bswap32(uint32_t v) {
    return (v >> 24) | ((v >> 8) & 0xff00) | ((v << 8) & 0xff0000) | (v << 24);
}

// A file opened for reading by a UTF-8 name (Tcl and VMD hand names over in
// UTF-8; on Windows fopen would read them in the ANSI code page).
class File {
    FILE* f = nullptr;
public:
    explicit File(const char* path) {
#ifdef _WIN32
        int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, nullptr, 0);
        if (n > 0) {
            std::wstring w((size_t)n, wchar_t(0));
            MultiByteToWideChar(CP_UTF8, 0, path, -1, &w[0], n);
            f = _wfopen(w.c_str(), L"rb");
        } else {
            f = std::fopen(path, "rb");
        }
#else
        f = std::fopen(path, "rb");
#endif
    }
    ~File() { if (f) std::fclose(f); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    FILE* get() const { return f; }
};

bool seek64(FILE* f, long long off) {
#ifdef _WIN32
    return _fseeki64(f, off, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t)off, SEEK_SET) == 0;
#endif
}

long long size64(FILE* f) {
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return -1;
    return _ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END) != 0) return -1;
    return (long long)ftello(f);
#endif
}

struct DcdReader {
    FILE* f;
    std::string path;
    bool swap = false;
    long long pos = 0;

    std::runtime_error bad(const std::string& why) const {
        return std::runtime_error("DCD file " + path + ": " + why);
    }
    uint32_t u32() {
        unsigned char b[4];
        if (std::fread(b, 1, 4, f) != 4) throw bad("ends early (at byte " + std::to_string(pos) + ")");
        pos += 4;
        uint32_t v;
        std::memcpy(&v, b, 4);
        return swap ? bswap32(v) : v;
    }
    void bytes(void* out, size_t n) {
        if (n && std::fread(out, 1, n, f) != n) throw bad("ends early (at byte " + std::to_string(pos) + ")");
        pos += (long long)n;
    }
    // a Fortran record of exactly n bytes (its length before and after)
    void record(void* out, uint32_t n, const char* what) {
        uint32_t a = u32();
        if (a != n) throw bad(std::string("the ") + what + " record is " + std::to_string(a) +
                              " bytes, not " + std::to_string(n));
        bytes(out, n);
        uint32_t b = u32();
        if (b != n) throw bad(std::string("the ") + what + " record's end marker is wrong");
    }
};

// 3-D superposition by Horn's quaternion: the rotation R and the centres
// with x' = R (x - cm) + cr moving m onto r (least RMSD, equal weights).
void fitRotation(const Eigen::Matrix3Xd& m, const Eigen::Matrix3Xd& r,
                 Eigen::Matrix3d& R, Eigen::Vector3d& cm, Eigen::Vector3d& cr) {
    cm = m.rowwise().mean();
    cr = r.rowwise().mean();
    Eigen::Matrix3d S = (m.colwise() - cm) * (r.colwise() - cr).transpose();   // S(a,b) = sum m_a r_b
    const double xx = S(0, 0), xy = S(0, 1), xz = S(0, 2);
    const double yx = S(1, 0), yy = S(1, 1), yz = S(1, 2);
    const double zx = S(2, 0), zy = S(2, 1), zz = S(2, 2);
    Eigen::Matrix4d K;
    K << xx + yy + zz, yz - zy, zx - xz, xy - yx,
         yz - zy, xx - yy - zz, xy + yx, zx + xz,
         zx - xz, xy + yx, -xx + yy - zz, yz + zy,
         xy - yx, zx + xz, yz + zy, -xx - yy + zz;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(K);
    Eigen::Vector4d q = es.eigenvectors().col(3);   // the largest eigenvalue's
    const double q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    R << q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2),
         2*(q2*q1 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1),
         2*(q3*q1 - q0*q2), 2*(q3*q2 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3;
}

// ---------------------------------------------------------------- results

void fill_common_results(mdance_result* r, const ArrayXXd& data,
                         const std::vector<int>& labels, int nclusters,
                         int natoms, MD::Metric metric) {
    r->nframes = (int)data.rows();
    r->nclusters = nclusters;
    r->labels = labels;
    r->cluster_sizes = computeClusterSizes(labels, nclusters);
    r->representatives = computeRepresentatives(data, labels, nclusters, natoms, metric);
    r->cluster_msd = computeClusterMSD(data, labels, nclusters, natoms);

    VectorXi scoreLabels(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        scoreLabels(i) = labels[i];
    }
    MD::checkCancel();
    r->ch_score = calinskiHarabaszScore(data, scoreLabels);
    r->db_score = daviesBouldinScore(data, scoreLabels);
}

// ---------------------------------------------------------------- algorithms

void runKmeans(mdance_result& r, const mdance_dataset& d, Reader& R) {
    int k = (int)R.whole("k", 0, 1, d.nframes, true);
    MD::Metric mt = parseMetric(R.choice("metric", "MSD", METRICS));
    MD::KinitType ki = parseKinit(R.choice("kinit", "StratAll", KINITS));
    int pct = (int)R.whole("percentage", 10, 0, 100);
    int seed = (int)R.whole("seed", -1, -1, INT_MAX);
    R.finish();

    KmeansNANI km(d.X, k, mt, ki, d.natoms, pct, 16, seed);
    Veci kl = km.getLabels();
    std::vector<int> labels(kl.data(), kl.data() + kl.size());
    fill_common_results(&r, d.X, labels, k, d.natoms, mt);
    r.seed = km.getSeed();
}

void runDivine(mdance_result& r, const mdance_dataset& d, Reader& R) {
    int k = (int)R.whole("k", 0, 0, d.nframes, true);
    MD::Metric mt = parseMetric(R.choice("metric", "MSD", METRICS));
    MD::DivineSplit sp = parseSplit(R.choice("split", "WeightedMSD", {"MSD", "Radius", "WeightedMSD"}));
    MD::DivineAnchors an = parseAnchors(R.choice("anchors", "NANI", {"NANI", "OutlierPair", "SplinterPair"}));
    MD::KinitType ki = parseKinit(R.choice("kinit", "StratAll", KINITS));
    bool refine = R.flag("refine", false);
    double threshold = R.number("threshold", 0.0, 0.0, 1.0);
    int end = R.choice("end_mode", "k", {"k", "points"}) == "points" ? 1 : 0;
    int pct = (int)R.whole("percentage", 10, 0, 100);
    R.finish();
#ifndef MDANCE_HAS_DIVINE
    (void)r; (void)k; (void)mt; (void)sp; (void)an; (void)ki; (void)refine; (void)threshold; (void)end; (void)pct;
    throw std::runtime_error("DIVINE is not available in this build of libmdance "
                             "(built without BUILD_DIVINE).");
#else
    Divine div(d.X, sp, an, ki, end, k, refine, d.natoms, threshold, pct);
    std::vector<int> labels = div.getLabels();
    std::set<int> uniqueLabels(labels.begin(), labels.end());
    fill_common_results(&r, d.X, labels, (int)uniqueLabels.size(), d.natoms, mt);
#endif
}

// HELM's distance matrix holds a double for every pair of starting clusters,
// and each merge copies it: refuse a number of them that would take gigabytes.
const int kHelmMaxStart = 5000;

void runHelm(mdance_result& r, const mdance_dataset& d, Reader& R) {
    const std::vector<int>* init = R.list("labels", d.nframes, true, INT_MIN, INT_MAX);
    int k = (int)R.whole("k", 0, 0, d.nframes);
    double eps = R.number("eps", -1.0, -1.0, 3.4e38);
    MD::Metric mt = parseMetric(R.choice("metric", "MSD", METRICS));
    MD::MergeScheme ms = parseMergeScheme(R.choice("merge_scheme", "Inter", {"Inter", "Intra", "Half"}));
    bool trimStart = R.flag("trim_start", false);
    double minSamples = R.number("min_samples", 0.01, 0.0, 3.4e38);
    double trimVal = R.number("trim_val", 0.0, 0.0, 3.4e38);
    double trimK = R.number("trim_k", 0.0, 0.0, 2147483647.0);
    R.finish();
    // (judged as the float HELM keeps: -0.99999999 is -1 there, as in mdance-cli)
    const float epsf = asFloat("helm: eps", eps);
    if (epsf != -1.0f && epsf <= 0) throw std::runtime_error("helm: option eps: -1 (none) or a number above 0");

    std::set<int> distinct(init->begin(), init->end());
    if ((int)distinct.size() > kHelmMaxStart)
        throw std::runtime_error("helm: " + std::to_string(distinct.size()) + " starting clusters; at most " +
                                 std::to_string(kHelmMaxStart) + " (each pair's distance is kept)");

    Eigen::ArrayXi initLabels = Eigen::Map<const Eigen::ArrayXi>(init->data(), (Index)init->size());
    std::vector<HCTree> clusterTree = buildClusterTree(d.X, initLabels);

    Helm helm(clusterTree, d.natoms, mt, ms, k, epsf,
              trimStart, asFloat("helm: min_samples", minSamples),
              asFloat("helm: trim_val", trimVal), asFloat("helm: trim_k", trimK));
    std::vector<HCTree> finalClusters = helm.run();

    int actualK = (int)finalClusters.size();
    std::vector<int> labels = labelsFromHelmClusters(finalClusters, initLabels, d.nframes);
    fill_common_results(&r, d.X, labels, actualK, d.natoms, mt);

    Mat zMat = helm.getZMatrix();
    if (zMat.rows() > 0) {
        r.zmatrix_rows = (int)zMat.rows();
        r.zmatrix_cols = (int)zMat.cols();
        r.zmatrix_flat.resize((size_t)zMat.rows() * zMat.cols());
        for (Index i = 0; i < zMat.rows(); ++i)
            for (Index j = 0; j < zMat.cols(); ++j)
                r.zmatrix_flat[(size_t)i * zMat.cols() + j] = zMat(i, j);
    }
}

void runEqual(mdance_result& r, const mdance_dataset& d, Reader& R) {
    double threshold = R.number("threshold", 0, 0, 1e300, true);
    MD::Metric mt = parseMetric(R.choice("metric", "MSD", METRICS));
    MD::EqualSeed seed = parseEqualSeed(R.choice("seed_method", "medoid", {"medoid", "comp_sim", "Medoid", "CompSim"}));
    double nSeeds = R.number("n_seeds", 1, 0, 1e300);
    bool checkSim = R.flag("check_sim", false);
    double simThreshold = R.number("sim_threshold", 0, -1e300, 1e300);
    bool rejectLowd = R.flag("reject_lowd", false);
    double minSamples = R.number("min_samples", 10, 0, 1e300);
    int pct = (int)R.whole("percentage", 10, 0, 100);
    MD::AlignMethod al = parseAlign(R.choice("align", "none", {"none", "None"}));
    R.finish();

    Equal eq(d.X, mt, threshold, d.natoms, seed, nSeeds, checkSim, simThreshold, rejectLowd,
             minSamples, pct, al);
    std::vector<int> labels = eq.getLabels();

    // eQUAL derives K; -1 marks unclustered frames.
    int actualK = 0;
    for (int l : labels) if (l + 1 > actualK) actualK = l + 1;
    fill_common_results(&r, d.X, labels, actualK, d.natoms, mt);

    // CH/DB over the clustered frames only (the -1 noise is no cluster).
    std::vector<int> keep;
    for (int j = 0; j < (int)labels.size(); ++j) if (labels[j] >= 0) keep.push_back(j);
    if (actualK >= 2 && (int)keep.size() >= 2) {
        ArrayXXd sub((Index)keep.size(), d.ncols);
        VectorXi subLabels((Index)keep.size());
        for (int i = 0; i < (int)keep.size(); ++i) {
            sub.row(i) = d.X.row(keep[i]);
            subLabels(i) = labels[keep[i]];
        }
        r.ch_score = calinskiHarabaszScore(sub, subLabels);
        r.db_score = daviesBouldinScore(sub, subLabels);
    } else {
        r.ch_score = 0.0;
        r.db_score = 0.0;
    }
}

void runAnalysis(mdance_analysis_t& a, const mdance_dataset& d, Reader& R) {
    MD::Metric mt = parseMetric(R.choice("metric", "MSD", METRICS));
    const std::vector<int>* labels = R.list("labels", d.nframes, false, -1, d.nframes - 1);
    R.finish();
    const ArrayXXd& X = d.X;

    a.isim = (X.rows() >= 2) ? extendedComparison(X, X.rows(), d.natoms, false, mt) : 0.0;
    if (!labels) return;

    int nClusters = 0;
    for (int l : *labels) if (l + 1 > nClusters) nClusters = l + 1;
    a.nclusters = nClusters;
    a.cluster_isim.assign(nClusters, 0.0);
    a.cluster_outliers.assign(nClusters, -1);
    std::vector<std::vector<int>> members(nClusters);
    for (int j = 0; j < d.nframes; ++j) if ((*labels)[j] >= 0) members[(*labels)[j]].push_back(j);
    for (int c = 0; c < nClusters; ++c) {
        MD::checkCancel();
        const std::vector<int>& m = members[c];
        if (m.empty()) continue;
        if (m.size() == 1) { a.cluster_outliers[c] = m[0]; continue; }
        ArrayXXd sub((Index)m.size(), X.cols());
        for (size_t i = 0; i < m.size(); ++i) sub.row(i) = X.row(m[i]);
        a.cluster_isim[c] = extendedComparison(sub, (Index)m.size(), d.natoms, false, mt);
        a.cluster_outliers[c] = m[(int)calculateOutlier(sub, d.natoms, mt)];
    }
}

void runPrime(mdance_prime_t& p, const mdance_dataset& d, Reader& R) {
    const std::vector<int>* labels = R.list("labels", d.nframes, true, -1, d.nframes - 1);
    MD::Metric mt = parseMetric(R.choice("metric", "RR", {"RR", "SM"}));
    double trim = R.number("trim_frac", 0.0, 0.0, 0.999999999);
    bool weighted = R.flag("weighted", false);
    R.finish();
    Prime prime(d.X, *labels, mt, trim, weighted);
    p.res = prime.getResult();
}

void runSelect(mdance_select_t& s, const mdance_dataset& d, Reader& R) {
    std::string m = R.choice("method", "diversity", {"diversity", "outliers", "repsample", "medoid", "outlier"});
    MD::Metric mt = parseMetric(R.choice("metric", "MSD", METRICS));
    double param = R.number("param", 10.0, 0.0, 1e15);
    int nbins = (int)R.whole("nbins", 10, 1, 1000000);
    R.finish();
    const ArrayXXd& X = d.X;
    int N = d.nframes;
    // a count, or a fraction of the frames in (0, 1); clamped to the frames
    auto count = [&](double v) {
        double c = (v > 0 && v < 1) ? std::floor(v * N) : std::floor(v);
        return (int)std::min<double>(c, N);
    };

    if (m == "diversity") {
        double pv = (param > 0 && param < 1) ? param * 100 : param;
        int pct = (int)std::min(100.0, std::max(1.0, std::floor(pv)));
        for (Index i : diversitySelection(X, pct, mt, d.natoms, true, MD::StartSeed::Medoid))
            s.indices.push_back((int)i);
    } else if (m == "outliers") {
        int n = count(param);
        ArrayXd cs = calculateCompSim(X, d.natoms, mt);
        std::vector<std::pair<double, int>> v;
        v.reserve(N);
        for (int i = 0; i < (int)cs.size(); ++i) v.emplace_back(cs[i], i);
        std::sort(v.begin(), v.end());   // ascending: the biggest outliers first
        for (int i = 0; i < n; ++i) s.indices.push_back(v[i].second);
    } else if (m == "repsample") {
        int n = std::max(1, count(param));
        ArrayXi idx = repSample(X, mt, d.natoms, nbins, n, true);
        for (int i = 0; i < idx.size(); ++i) s.indices.push_back(idx(i));
    } else if (m == "medoid") {
        s.indices.push_back((int)calculateMedoid(X, d.natoms, mt));
    } else {
        s.indices.push_back((int)calculateOutlier(X, d.natoms, mt));
    }
}

// ------------------------------------------- the first interface's helpers

// A dataset copied from the first interface's buffer: NULL and the reason in
// err when the buffer is not usable.
mdance_dataset_t* legacyData(const double* coords, int nframes, int ncols, int natoms, std::string& err) {
    if (!coords || nframes <= 0 || ncols <= 0) {
        err = "coordinate matrix is empty or null (need nframes > 0, ncols > 0).";
        return nullptr;
    }
    mdance_dataset_t* d = mdance_dataset_from_rows(coords, nframes, ncols, natoms);
    if (!d) err = lastError;
    return d;
}

void put(mdance_options& o, const char* k, const char* v) { if (v) o.text[k] = v; }
void put(mdance_options& o, const char* k, long long v) { o.text[k] = std::to_string(v); }
void putd(mdance_options& o, const char* k, double v) {
    // a value that is no number goes in as its text, which the reader refuses
    o.text[k] = std::isfinite(v) ? fmt(v) : (std::isnan(v) ? "nan" : "inf");
}

// the labels of the first interface: as many as given, cut or padded with -1
// to the frames (PRIME and the analysis took them so)
std::vector<int> fitted(const int* labels, int nlabels, int nframes) {
    std::vector<int> v;
    if (labels && nlabels > 0) v.assign(labels, labels + std::min(nlabels, nframes));
    v.resize(nframes, -1);
    return v;
}

template <class T, class Run>
T* legacyCall(const double* coords, int nframes, int ncols, int natoms, mdance_options& o, Run run) {
    std::string err;
    mdance_dataset_t* d = legacyData(coords, nframes, ncols, natoms, err);
    if (!d) {
        T* r = new (std::nothrow) T;
        if (r) r->error = err;
        return r;
    }
    T* r = run(d, &o);
    mdance_dataset_free(d);
    return r;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------- general

const char* mdance_version(void) { return "2.0.0"; }
int mdance_api_version(void) { return MDANCE_API_VERSION; }
int mdance_has_divine(void) {
#ifdef MDANCE_HAS_DIVINE
    return 1;
#else
    return 0;
#endif
}
const char* mdance_last_error(void) { return lastError.c_str(); }

// ---------------------------------------------------------------- datasets

mdance_dataset_t* mdance_dataset_new(int nframes, int natoms) {
    if (natoms <= 0 || natoms > INT_MAX / 3) { fail("natoms must be in 1 .. " + std::to_string(INT_MAX / 3)); return nullptr; }
    return newDataset(nframes, 3 * natoms, natoms);
}

mdance_dataset_t* mdance_dataset_new_cols(int nframes, int ncols, int natoms) {
    return newDataset(nframes, ncols, natoms);
}

mdance_dataset_t* mdance_dataset_from_rows(const double* rows, int nframes, int ncols, int natoms) {
    if (!rows) { fail("no values (NULL)"); return nullptr; }
    mdance_dataset_t* d = newDataset(nframes, ncols, natoms);
    if (!d) return nullptr;
    for (int f = 0; f < nframes; ++f) {
        if (setFrame(d, f, rows + (size_t)f * ncols, ncols) != 0) {
            mdance_dataset_free(d);
            return nullptr;
        }
    }
    return d;
}

int mdance_dataset_set_frame(mdance_dataset_t* d, int frame, const double* values, int n) {
    return setFrame(d, frame, values, n);
}

int mdance_dataset_set_frame_float(mdance_dataset_t* d, int frame, const float* values, int n) {
    return setFrame(d, frame, values, n);
}

int mdance_dataset_set_frame_atoms_float(mdance_dataset_t* d, int frame, const float* pos,
                                         int natoms_in_pos, const int* atoms, int n) {
    if (!d) return fail("no dataset (NULL)");
    if (!pos || !atoms) return fail("no coordinates or atoms (NULL)");
    if (d->ncols != 3 * d->natoms) return fail("the dataset does not have 3 columns an atom");
    if (n != d->natoms) return fail(std::to_string(n) + " atoms for a dataset of " + std::to_string(d->natoms));
    std::vector<double> v((size_t)3 * n);
    for (int i = 0; i < n; ++i) {
        int a = atoms[i];
        if (a < 0 || a >= natoms_in_pos)
            return fail("atom index " + std::to_string(a) + " is outside 0 .. " + std::to_string(natoms_in_pos - 1));
        v[3 * i] = pos[3 * (size_t)a];
        v[3 * i + 1] = pos[3 * (size_t)a + 1];
        v[3 * i + 2] = pos[3 * (size_t)a + 2];
    }
    return setFrame(d, frame, v.data(), 3 * n);
}

mdance_dataset_t* mdance_dataset_read_dcd(const char* path, const int* atoms, int n,
                                          int first, int last, int stride) {
    if (!path) { fail("no file name (NULL)"); return nullptr; }
    mdance_dataset_t* d = nullptr;
    try {
        File file(path);
        if (!file.get()) throw std::runtime_error(std::string("DCD file ") + path + " cannot be opened: " + std::strerror(errno));
        DcdReader r{file.get(), path};
        long long fsize = size64(r.f);
        if (fsize < 0 || !seek64(r.f, 0)) throw r.bad("cannot be read");

        // byte order, from the first record's length (84)
        uint32_t m = r.u32();
        if (m == 84) r.swap = false;
        else if (bswap32(m) == 84) r.swap = true;
        else throw r.bad("is not a DCD file of 32-bit records (first record of " + std::to_string(m) + " bytes, not 84)");
        char cord[4];
        r.bytes(cord, 4);
        if (std::memcmp(cord, "CORD", 4) != 0) throw r.bad("has no CORD header");
        uint32_t icntrl[20];
        for (int i = 0; i < 20; ++i) icntrl[i] = r.u32();
        if (r.u32() != 84) throw r.bad("its header's end marker is wrong");
        const bool charmm = icntrl[19] != 0;
        const bool extra = charmm && icntrl[10] != 0;     // unit cell before each frame
        const bool fourD = charmm && icntrl[11] != 0;
        const uint32_t nfixed = icntrl[8];
        if (nfixed != 0) throw r.bad("has fixed atoms, which this reader does not read");

        // titles
        uint32_t tl = r.u32();
        if (tl < 4 || (tl - 4) % 80 != 0) throw r.bad("its title record is " + std::to_string(tl) + " bytes");
        if (!seek64(r.f, r.pos + tl)) throw r.bad("ends early");
        r.pos += tl;
        if (r.u32() != tl) throw r.bad("its title record's end marker is wrong");
        // atoms
        int32_t natomsFile;
        r.record(&natomsFile, 4, "atom count");
        if (r.swap) natomsFile = (int32_t)bswap32((uint32_t)natomsFile);
        if (natomsFile <= 0) throw r.bad("holds " + std::to_string(natomsFile) + " atoms");
        if ((long long)natomsFile * 4 > 0x7fffffffLL) throw r.bad("holds too many atoms");

        const long long header = r.pos;
        const long long block = 8 + 4LL * natomsFile;
        const long long frameBytes = (extra ? 8 + 48 : 0) + 3 * block + (fourD ? block : 0);
        const long long body = fsize - header;
        if (body < 0 || body % frameBytes != 0)
            throw r.bad("is cut short or damaged: " + std::to_string(body) + " bytes of frames are not a whole number of " +
                        std::to_string(frameBytes) + "-byte frames");
        const long long nf = body / frameBytes;
        if (nf == 0) throw r.bad("holds no frames");
        if (nf > INT_MAX) throw r.bad("holds too many frames");

        if (stride <= 0) throw std::runtime_error("stride must be positive");
        if (last < 0) last = (int)nf - 1;
        if (first < 0 || first >= nf || last >= nf || last < first)
            throw std::runtime_error("frames " + std::to_string(first) + " .. " + std::to_string(last) +
                                     " are not within the file's 0 .. " + std::to_string(nf - 1));
        std::vector<int> sel;
        if (atoms && n > 0) {
            sel.assign(atoms, atoms + n);
            for (int a : sel)
                if (a < 0 || a >= natomsFile)
                    throw std::runtime_error("atom index " + std::to_string(a) + " is outside the file's 0 .. " +
                                             std::to_string(natomsFile - 1));
        } else {
            sel.resize(natomsFile);
            for (int i = 0; i < natomsFile; ++i) sel[i] = i;
        }
        const int nsel = (int)sel.size();
        const int nout = (last - first) / stride + 1;
        d = mdance_dataset_new(nout, nsel);
        if (!d) throw std::runtime_error(lastError);

        std::vector<float> xyz[3];
        for (auto& v : xyz) v.resize(natomsFile);
        std::vector<double> row((size_t)3 * nsel);
        const char* names[3] = {"x", "y", "z"};
        for (int o = 0; o < nout; ++o) {
            long long f = first + (long long)o * stride;
            r.pos = header + f * frameBytes + (extra ? 56 : 0);
            if (!seek64(r.f, r.pos)) throw r.bad("cannot be read at frame " + std::to_string(f));
            for (int c = 0; c < 3; ++c) {
                r.record(xyz[c].data(), 4u * (uint32_t)natomsFile, names[c]);
                if (r.swap) {
                    for (float& v : xyz[c]) {
                        uint32_t u;
                        std::memcpy(&u, &v, 4);
                        u = bswap32(u);
                        std::memcpy(&v, &u, 4);
                    }
                }
            }
            for (int i = 0; i < nsel; ++i) {
                row[3 * i] = xyz[0][sel[i]];
                row[3 * i + 1] = xyz[1][sel[i]];
                row[3 * i + 2] = xyz[2][sel[i]];
            }
            if (setFrame(d, o, row.data(), 3 * nsel) != 0)
                throw std::runtime_error("DCD file " + std::string(path) + ", frame " + std::to_string(f) + ": " + lastError);
        }
        return d;
    } catch (...) {
        mdance_dataset_free(d);
        fail(currentMessage());
        return nullptr;
    }
}

int mdance_dataset_superpose(mdance_dataset_t* d, const int* fit, int nfit, int ref_frame) {
    try {
        const mdance_dataset& D = ready(d);
        if (D.ncols != 3 * D.natoms) return fail("superpose: the dataset does not have 3 columns an atom");
        if (ref_frame < 0 || ref_frame >= D.nframes)
            return fail("superpose: reference frame " + std::to_string(ref_frame) + " is outside 0 .. " +
                        std::to_string(D.nframes - 1));
        std::vector<int> idx;
        if (fit && nfit > 0) idx.assign(fit, fit + nfit);
        else { idx.resize(D.natoms); for (int i = 0; i < D.natoms; ++i) idx[i] = i; }
        for (int a : idx)
            if (a < 0 || a >= D.natoms)
                return fail("superpose: atom " + std::to_string(a) + " is outside 0 .. " + std::to_string(D.natoms - 1));
        const int nf = (int)idx.size();
        Eigen::Matrix3Xd ref(3, nf), mov(3, nf);
        for (int i = 0; i < nf; ++i)
            for (int c = 0; c < 3; ++c) ref(c, i) = D.X(ref_frame, 3 * idx[i] + c);
        Eigen::Matrix3d R;
        Eigen::Vector3d cm, cr;
        for (int f = 0; f < D.nframes; ++f) {
            for (int i = 0; i < nf; ++i)
                for (int c = 0; c < 3; ++c) mov(c, i) = D.X(f, 3 * idx[i] + c);
            fitRotation(mov, ref, R, cm, cr);
            for (int a = 0; a < D.natoms; ++a) {
                Eigen::Vector3d x(D.X(f, 3 * a), D.X(f, 3 * a + 1), D.X(f, 3 * a + 2));
                Eigen::Vector3d y = R * (x - cm) + cr;
                d->X(f, 3 * a) = y[0];
                d->X(f, 3 * a + 1) = y[1];
                d->X(f, 3 * a + 2) = y[2];
            }
        }
        return 0;
    } catch (...) {
        return fail(currentMessage());
    }
}

int mdance_dataset_nframes(const mdance_dataset_t* d) { return d ? d->nframes : 0; }
int mdance_dataset_ncols(const mdance_dataset_t* d) { return d ? d->ncols : 0; }
int mdance_dataset_natoms(const mdance_dataset_t* d) { return d ? d->natoms : 0; }
int mdance_dataset_nset(const mdance_dataset_t* d) { return d ? d->nset : 0; }

int mdance_dataset_get_frame(const mdance_dataset_t* d, int frame, double* out, int n) {
    if (!d || !out) return fail("no dataset or output (NULL)");
    if (frame < 0 || frame >= d->nframes) return fail("frame " + std::to_string(frame) + " is outside the dataset");
    if (n != d->ncols) return fail(std::to_string(n) + " values for a frame of " + std::to_string(d->ncols));
    for (int j = 0; j < n; ++j) out[j] = d->X(frame, j);
    return 0;
}

void mdance_dataset_free(mdance_dataset_t* d) { delete d; }

// ---------------------------------------------------------------- options

mdance_options_t* mdance_options_new(void) {
    mdance_options_t* o = new (std::nothrow) mdance_options_t;
    if (!o) fail(kOutOfMemory);
    return o;
}

int mdance_options_set(mdance_options_t* o, const char* name, const char* value) {
    if (!o || !name || !value) return fail("options: NULL argument");
    try {
        o->lists.erase(name);
        o->text[name] = value;
        return 0;
    } catch (...) {
        return fail(currentMessage());
    }
}

int mdance_options_set_int(mdance_options_t* o, const char* name, long long value) {
    return mdance_options_set(o, name, std::to_string(value).c_str());
}

int mdance_options_set_double(mdance_options_t* o, const char* name, double value) {
    if (!std::isfinite(value)) return fail(std::string("options: ") + (name ? name : "?") + ": not a finite number");
    return mdance_options_set(o, name, fmt(value).c_str());
}

int mdance_options_set_ints(mdance_options_t* o, const char* name, const int* values, int n) {
    if (!o || !name || (!values && n > 0) || n < 0) return fail("options: NULL argument or negative count");
    try {
        o->text.erase(name);
        o->lists[name].assign(values, values + n);
        return 0;
    } catch (...) {
        return fail(currentMessage());
    }
}

int mdance_options_unset(mdance_options_t* o, const char* name) {
    if (!o || !name) return fail("options: NULL argument");
    o->text.erase(name);
    o->lists.erase(name);
    return 0;
}

void mdance_options_free(mdance_options_t* o) { delete o; }

// ---------------------------------------------------------------- controls

mdance_control_t* mdance_control_new(void) {
    mdance_control_t* c = new (std::nothrow) mdance_control_t;
    if (!c) fail(kOutOfMemory);
    return c;
}
void mdance_control_cancel(mdance_control_t* c) { if (c) c->cancel.store(1); }
int mdance_control_cancelled(const mdance_control_t* c) { return c ? c->cancel.load() : 0; }
double mdance_control_progress(const mdance_control_t* c) { return c ? c->fraction.load() : 0.0; }
const char* mdance_control_stage(const mdance_control_t* c) { return c ? c->stage.load() : ""; }
void mdance_control_reset(mdance_control_t* c) {
    if (!c) return;
    c->cancel.store(0);
    c->fraction.store(0.0);
    c->stage.store("");
}
void mdance_control_free(mdance_control_t* c) { delete c; }

// ---------------------------------------------------------------- calls

mdance_result_t* mdance_cluster(const mdance_dataset_t* d, const char* algorithm,
                                const mdance_options_t* o, mdance_control_t* c) {
    return guarded<mdance_result>(c, [&](mdance_result& r) {
        std::string a = algorithm ? algorithm : "";
        const mdance_dataset& D = ready(d);
        Reader R(o, a);
        if (a == "kmeans") runKmeans(r, D, R);
        else if (a == "divine") runDivine(r, D, R);
        else if (a == "helm") runHelm(r, D, R);
        else if (a == "equal") runEqual(r, D, R);
        else throw std::runtime_error("unknown algorithm \"" + a + "\" (kmeans divine helm equal)");
    });
}

mdance_analysis_t* mdance_analyze(const mdance_dataset_t* d, const mdance_options_t* o, mdance_control_t* c) {
    return guarded<mdance_analysis_t>(c, [&](mdance_analysis_t& a) {
        Reader R(o, "analysis");
        runAnalysis(a, ready(d), R);
    });
}

mdance_prime_t* mdance_prime_run(const mdance_dataset_t* d, const mdance_options_t* o, mdance_control_t* c) {
    return guarded<mdance_prime_t>(c, [&](mdance_prime_t& p) {
        Reader R(o, "prime");
        runPrime(p, ready(d), R);
    });
}

mdance_select_t* mdance_select_run(const mdance_dataset_t* d, const mdance_options_t* o, mdance_control_t* c) {
    return guarded<mdance_select_t>(c, [&](mdance_select_t& s) {
        Reader R(o, "select");
        runSelect(s, ready(d), R);
    });
}

// ---------------------------------------------------------------- the first interface

mdance_result_t* mdance_kmeans(const double* coords, int nframes, int ncols, int natoms, int nclusters,
                               const char* metric, const char* kinit, int percentage) {
    mdance_options o;
    put(o, "k", nclusters);
    put(o, "metric", metric);
    put(o, "kinit", kinit);
    put(o, "percentage", percentage);
    return legacyCall<mdance_result>(coords, nframes, ncols, natoms, o, [](mdance_dataset_t* d, mdance_options_t* op) {
        return mdance_cluster(d, "kmeans", op, nullptr);
    });
}

mdance_result_t* mdance_divine(const double* coords, int nframes, int ncols, int natoms, int nclusters,
                               const char* metric, const char* split, const char* anchors, const char* kinit,
                               int refine, double threshold, const char* end_mode, int percentage) {
    mdance_options o;
    put(o, "k", nclusters);
    put(o, "metric", metric);
    put(o, "split", split);
    put(o, "anchors", anchors);
    put(o, "kinit", kinit);
    put(o, "refine", refine ? 1 : 0);
    putd(o, "threshold", threshold);
    put(o, "end_mode", end_mode);
    put(o, "percentage", percentage);
    return legacyCall<mdance_result>(coords, nframes, ncols, natoms, o, [](mdance_dataset_t* d, mdance_options_t* op) {
        return mdance_cluster(d, "divine", op, nullptr);
    });
}

mdance_result_t* mdance_helm(const double* coords, int nframes, int ncols, int natoms, int nclusters,
                             const int* initial_labels, int nlabels, const char* metric,
                             const char* merge_scheme, float eps, int trim_start, float min_samples,
                             float trim_val, float trim_k) {
    mdance_options o;
    if (!initial_labels || nlabels <= 0) {
        mdance_result_t* r = new (std::nothrow) mdance_result_t;
        if (r) r->error = "HELM requires per-frame initial cluster labels.";
        return r;
    }
    o.lists["labels"].assign(initial_labels, initial_labels + nlabels);
    put(o, "k", nclusters);
    put(o, "metric", metric);
    put(o, "merge_scheme", merge_scheme);
    putd(o, "eps", eps);
    put(o, "trim_start", trim_start ? 1 : 0);
    putd(o, "min_samples", min_samples);
    putd(o, "trim_val", trim_val);
    putd(o, "trim_k", trim_k);
    return legacyCall<mdance_result>(coords, nframes, ncols, natoms, o, [](mdance_dataset_t* d, mdance_options_t* op) {
        return mdance_cluster(d, "helm", op, nullptr);
    });
}

mdance_result_t* mdance_equal(const double* coords, int nframes, int ncols, int natoms, const char* metric,
                              double threshold, const char* seed_method, double n_seeds, int check_sim,
                              double sim_threshold, int reject_lowd, double min_samples, int percentage,
                              const char* align) {
    mdance_options o;
    put(o, "metric", metric);
    putd(o, "threshold", threshold);
    put(o, "seed_method", seed_method);
    putd(o, "n_seeds", n_seeds);
    put(o, "check_sim", check_sim ? 1 : 0);
    putd(o, "sim_threshold", sim_threshold);
    put(o, "reject_lowd", reject_lowd ? 1 : 0);
    putd(o, "min_samples", min_samples);
    put(o, "percentage", percentage);
    put(o, "align", align);
    return legacyCall<mdance_result>(coords, nframes, ncols, natoms, o, [](mdance_dataset_t* d, mdance_options_t* op) {
        return mdance_cluster(d, "equal", op, nullptr);
    });
}

mdance_analysis_t* mdance_analysis(const double* coords, int nframes, int ncols, int natoms,
                                   const char* metric, const int* labels, int nlabels) {
    mdance_options o;
    put(o, "metric", metric);
    if (labels && nlabels > 0 && nframes > 0) o.lists["labels"] = fitted(labels, nlabels, nframes);
    return legacyCall<mdance_analysis_t>(coords, nframes, ncols, natoms, o, [](mdance_dataset_t* d, mdance_options_t* op) {
        return mdance_analyze(d, op, nullptr);
    });
}

mdance_prime_t* mdance_prime(const double* coords, int nframes, int ncols, int natoms, const char* metric,
                             const int* labels, int nlabels, double trim_frac, int weighted_by_frames) {
    if (!labels || nlabels <= 0) {
        mdance_prime_t* p = new (std::nothrow) mdance_prime_t;
        if (p) p->error = "PRIME requires per-frame cluster labels.";
        return p;
    }
    mdance_options o;
    put(o, "metric", metric);
    if (nframes > 0) o.lists["labels"] = fitted(labels, nlabels, nframes);
    putd(o, "trim_frac", trim_frac);
    put(o, "weighted", weighted_by_frames ? 1 : 0);
    return legacyCall<mdance_prime_t>(coords, nframes, ncols, natoms, o, [](mdance_dataset_t* d, mdance_options_t* op) {
        return mdance_prime_run(d, op, nullptr);
    });
}

mdance_select_t* mdance_select(const double* coords, int nframes, int ncols, int natoms, const char* metric,
                               const char* method, double param, int nbins) {
    mdance_options o;
    put(o, "metric", metric);
    put(o, "method", method);
    putd(o, "param", param);
    put(o, "nbins", nbins > 0 ? nbins : 10);
    return legacyCall<mdance_select_t>(coords, nframes, ncols, natoms, o, [](mdance_dataset_t* d, mdance_options_t* op) {
        return mdance_select_run(d, op, nullptr);
    });
}

// ---------------------------------------------------------------- accessors

// (an object that could not be allocated, NULL, reads as failed)
#define ERR_OF(x) ((x) ? ((x)->error.empty() ? nullptr : (x)->error.c_str()) : kOutOfMemory)

int mdance_result_nframes(const mdance_result_t* r) { return r ? r->nframes : 0; }
int mdance_result_nclusters(const mdance_result_t* r) { return r ? r->nclusters : 0; }
int mdance_result_seed(const mdance_result_t* r) { return r ? r->seed : -1; }
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
double mdance_result_ch_score(const mdance_result_t* r) { return r ? r->ch_score : 0.0; }
double mdance_result_db_score(const mdance_result_t* r) { return r ? r->db_score : 0.0; }
const char* mdance_result_error(const mdance_result_t* r) { return ERR_OF(r); }
int mdance_result_zmatrix_rows(const mdance_result_t* r) { return r ? r->zmatrix_rows : 0; }
int mdance_result_zmatrix_cols(const mdance_result_t* r) { return r ? r->zmatrix_cols : 0; }
const double* mdance_result_zmatrix(const mdance_result_t* r) {
    return (r && !r->zmatrix_flat.empty()) ? r->zmatrix_flat.data() : nullptr;
}
void mdance_result_free(mdance_result_t* r) { delete r; }

double mdance_analysis_isim(const mdance_analysis_t* a) { return a ? a->isim : 0.0; }
int mdance_analysis_nclusters(const mdance_analysis_t* a) { return a ? a->nclusters : 0; }
const double* mdance_analysis_cluster_isim(const mdance_analysis_t* a) {
    return (a && !a->cluster_isim.empty()) ? a->cluster_isim.data() : nullptr;
}
const int* mdance_analysis_cluster_outliers(const mdance_analysis_t* a) {
    return (a && !a->cluster_outliers.empty()) ? a->cluster_outliers.data() : nullptr;
}
const char* mdance_analysis_error(const mdance_analysis_t* a) { return ERR_OF(a); }
void mdance_analysis_free(mdance_analysis_t* a) { delete a; }

int mdance_prime_pairwise(const mdance_prime_t* p)          { return p ? p->res.pairwise : -1; }
int mdance_prime_union(const mdance_prime_t* p)             { return p ? p->res.uni : -1; }
int mdance_prime_medoid(const mdance_prime_t* p)            { return p ? p->res.medoid : -1; }
int mdance_prime_outlier(const mdance_prime_t* p)           { return p ? p->res.outlier : -1; }
int mdance_prime_medoid_all(const mdance_prime_t* p)        { return p ? p->res.medoid_all : -1; }
int mdance_prime_medoid_c0(const mdance_prime_t* p)         { return p ? p->res.medoid_c0 : -1; }
int mdance_prime_medoid_c0_trimmed(const mdance_prime_t* p) { return p ? p->res.medoid_c0_trimmed : -1; }
int mdance_prime_nclusters(const mdance_prime_t* p)         { return p ? p->res.nclusters : 0; }
const char* mdance_prime_error(const mdance_prime_t* p) { return ERR_OF(p); }
void mdance_prime_free(mdance_prime_t* p) { delete p; }

int mdance_select_count(const mdance_select_t* s) { return s ? (int)s->indices.size() : 0; }
const int* mdance_select_indices(const mdance_select_t* s) {
    return (s && !s->indices.empty()) ? s->indices.data() : nullptr;
}
const char* mdance_select_error(const mdance_select_t* s) { return ERR_OF(s); }
void mdance_select_free(mdance_select_t* s) { delete s; }

}  // extern "C"
