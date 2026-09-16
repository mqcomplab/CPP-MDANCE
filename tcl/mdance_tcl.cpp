// Build as a stubs-enabled loadable extension: USE_TCL_STUBS turns the Tcl_*
// API into calls through the host interpreter's stubs table (wired up by
// Tcl_InitStubs at load time) instead of direct references to a linked Tcl
// runtime. Must be defined before <tcl.h>. Paired with linking libtclstub
// (not the full Tcl library) in CMakeLists.txt — otherwise a second Tcl
// runtime is pulled into the host process and it aborts ("alloc: invalid
// block") when loaded into an embedder such as VMD.
#define USE_TCL_STUBS
#include <tcl.h>
#include <string>
#include <vector>
#include <cstring>

#include "../capi/mdance_capi.h"

// Helper: extract a flat double array from a Tcl list object
static int GetDoubleList(Tcl_Interp* interp, Tcl_Obj* listObj,
                          std::vector<double>& out) {
    int objc;
    Tcl_Obj** objv;
    if (Tcl_ListObjGetElements(interp, listObj, &objc, &objv) != TCL_OK) {
        return TCL_ERROR;
    }
    out.resize(objc);
    for (int i = 0; i < objc; ++i) {
        if (Tcl_GetDoubleFromObj(interp, objv[i], &out[i]) != TCL_OK) {
            return TCL_ERROR;
        }
    }
    return TCL_OK;
}

// Helper: extract a flat int array from a Tcl list object
static int GetIntList(Tcl_Interp* interp, Tcl_Obj* listObj,
                       std::vector<int>& out) {
    int objc;
    Tcl_Obj** objv;
    if (Tcl_ListObjGetElements(interp, listObj, &objc, &objv) != TCL_OK) {
        return TCL_ERROR;
    }
    out.resize(objc);
    for (int i = 0; i < objc; ++i) {
        if (Tcl_GetIntFromObj(interp, objv[i], &out[i]) != TCL_OK) {
            return TCL_ERROR;
        }
    }
    return TCL_OK;
}

// Helper: get optional string argument from -flag value pairs
static const char* GetOpt(int objc, Tcl_Obj* const objv[], const char* flag,
                           const char* defaultVal) {
    for (int i = 0; i < objc - 1; ++i) {
        if (strcmp(Tcl_GetString(objv[i]), flag) == 0) {
            return Tcl_GetString(objv[i + 1]);
        }
    }
    return defaultVal;
}

static int GetOptInt(int objc, Tcl_Obj* const objv[], const char* flag, int defaultVal) {
    for (int i = 0; i < objc - 1; ++i) {
        if (strcmp(Tcl_GetString(objv[i]), flag) == 0) {
            int val;
            if (Tcl_GetIntFromObj(nullptr, objv[i + 1], &val) == TCL_OK) {
                return val;
            }
        }
    }
    return defaultVal;
}

static double GetOptDouble(int objc, Tcl_Obj* const objv[], const char* flag, double defaultVal) {
    for (int i = 0; i < objc - 1; ++i) {
        if (strcmp(Tcl_GetString(objv[i]), flag) == 0) {
            double val;
            if (Tcl_GetDoubleFromObj(nullptr, objv[i + 1], &val) == TCL_OK) {
                return val;
            }
        }
    }
    return defaultVal;
}

static int HasFlag(int objc, Tcl_Obj* const objv[], const char* flag) {
    for (int i = 0; i < objc; ++i) {
        if (strcmp(Tcl_GetString(objv[i]), flag) == 0) {
            return 1;
        }
    }
    return 0;
}

// Helper: build result dict from mdance_result_t
static Tcl_Obj* BuildResultDict(Tcl_Interp* interp, const mdance_result_t* res,
                                  const char* algorithm) {
    Tcl_Obj* dict = Tcl_NewDictObj();

    Tcl_DictObjPut(interp, dict,
        Tcl_NewStringObj("algorithm", -1),
        Tcl_NewStringObj(algorithm, -1));

    int nframes = mdance_result_nframes(res);
    int nclusters = mdance_result_nclusters(res);

    Tcl_DictObjPut(interp, dict,
        Tcl_NewStringObj("nFrames", -1),
        Tcl_NewIntObj(nframes));
    Tcl_DictObjPut(interp, dict,
        Tcl_NewStringObj("nClusters", -1),
        Tcl_NewIntObj(nclusters));

    // Labels
    const int* labels = mdance_result_labels(res);
    if (labels) {
        Tcl_Obj* labelList = Tcl_NewListObj(0, nullptr);
        for (int i = 0; i < nframes; ++i) {
            Tcl_ListObjAppendElement(interp, labelList, Tcl_NewIntObj(labels[i]));
        }
        Tcl_DictObjPut(interp, dict,
            Tcl_NewStringObj("labels", -1), labelList);
    }

    // Cluster sizes
    const int* sizes = mdance_result_cluster_sizes(res);
    if (sizes) {
        Tcl_Obj* sizeList = Tcl_NewListObj(0, nullptr);
        for (int i = 0; i < nclusters; ++i) {
            Tcl_ListObjAppendElement(interp, sizeList, Tcl_NewIntObj(sizes[i]));
        }
        Tcl_DictObjPut(interp, dict,
            Tcl_NewStringObj("clusterSizes", -1), sizeList);
    }

    // Representatives
    const int* reps = mdance_result_representatives(res);
    if (reps) {
        Tcl_Obj* repList = Tcl_NewListObj(0, nullptr);
        for (int i = 0; i < nclusters; ++i) {
            Tcl_ListObjAppendElement(interp, repList, Tcl_NewIntObj(reps[i]));
        }
        Tcl_DictObjPut(interp, dict,
            Tcl_NewStringObj("representatives", -1), repList);
    }

    // Cluster MSD
    const double* msd = mdance_result_cluster_msd(res);
    if (msd) {
        Tcl_Obj* msdList = Tcl_NewListObj(0, nullptr);
        for (int i = 0; i < nclusters; ++i) {
            Tcl_ListObjAppendElement(interp, msdList, Tcl_NewDoubleObj(msd[i]));
        }
        Tcl_DictObjPut(interp, dict,
            Tcl_NewStringObj("clusterMSD", -1), msdList);
    }

    // Scores
    Tcl_DictObjPut(interp, dict,
        Tcl_NewStringObj("score_calinskiHarabasz", -1),
        Tcl_NewDoubleObj(mdance_result_ch_score(res)));
    Tcl_DictObjPut(interp, dict,
        Tcl_NewStringObj("score_daviesBouldin", -1),
        Tcl_NewDoubleObj(mdance_result_db_score(res)));

    // Z-matrix (HELM only)
    int zrows = mdance_result_zmatrix_rows(res);
    int zcols = mdance_result_zmatrix_cols(res);
    const double* zdata = mdance_result_zmatrix(res);
    if (zdata && zrows > 0) {
        Tcl_Obj* zMatrix = Tcl_NewListObj(0, nullptr);
        for (int i = 0; i < zrows; ++i) {
            Tcl_Obj* row = Tcl_NewListObj(0, nullptr);
            for (int j = 0; j < zcols; ++j) {
                Tcl_ListObjAppendElement(interp, row,
                    Tcl_NewDoubleObj(zdata[i * zcols + j]));
            }
            Tcl_ListObjAppendElement(interp, zMatrix, row);
        }
        Tcl_DictObjPut(interp, dict,
            Tcl_NewStringObj("zMatrix", -1), zMatrix);
    }

    return dict;
}

// --- Tcl commands ---

// mdance::kmeans coords nframes natoms nclusters ?-metric MSD? ?-kinit StratAll? ?-percentage 10?
static int KmeansCmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 5) {
        Tcl_WrongNumArgs(interp, 1, objv,
            "coords nframes natoms nclusters ?-metric MSD? ?-kinit StratAll? ?-percentage 10?");
        return TCL_ERROR;
    }

    std::vector<double> coords;
    if (GetDoubleList(interp, objv[1], coords) != TCL_OK) {
        return TCL_ERROR;
    }

    int nframes, natoms, nclusters;
    if (Tcl_GetIntFromObj(interp, objv[2], &nframes) != TCL_OK ||
        Tcl_GetIntFromObj(interp, objv[3], &natoms) != TCL_OK ||
        Tcl_GetIntFromObj(interp, objv[4], &nclusters) != TCL_OK) {
        return TCL_ERROR;
    }
    if (nframes <= 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("nframes must be positive", -1));
        return TCL_ERROR;
    }

    int ncols = coords.size() / nframes;

    const char* metric = GetOpt(objc, objv, "-metric", "MSD");
    const char* kinit = GetOpt(objc, objv, "-kinit", "StratAll");
    int percentage = GetOptInt(objc, objv, "-percentage", 10);

    mdance_result_t* res = mdance_kmeans(
        coords.data(), nframes, ncols, natoms, nclusters,
        metric, kinit, percentage);

    const char* err = mdance_result_error(res);
    if (err) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(err, -1));
        mdance_result_free(res);
        return TCL_ERROR;
    }

    Tcl_SetObjResult(interp, BuildResultDict(interp, res, "kmeans"));
    mdance_result_free(res);
    return TCL_OK;
}

// mdance::divine coords nframes natoms nclusters ?options...?
static int DivineCmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 5) {
        Tcl_WrongNumArgs(interp, 1, objv,
            "coords nframes natoms nclusters ?-metric MSD? ?-split WeightedMSD? "
            "?-anchors NANI? ?-kinit StratAll? ?-refine? ?-threshold 0? "
            "?-end-mode k? ?-percentage 10?");
        return TCL_ERROR;
    }

    std::vector<double> coords;
    if (GetDoubleList(interp, objv[1], coords) != TCL_OK) {
        return TCL_ERROR;
    }

    int nframes, natoms, nclusters;
    if (Tcl_GetIntFromObj(interp, objv[2], &nframes) != TCL_OK ||
        Tcl_GetIntFromObj(interp, objv[3], &natoms) != TCL_OK ||
        Tcl_GetIntFromObj(interp, objv[4], &nclusters) != TCL_OK) {
        return TCL_ERROR;
    }
    if (nframes <= 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("nframes must be positive", -1));
        return TCL_ERROR;
    }

    int ncols = coords.size() / nframes;

    const char* metric = GetOpt(objc, objv, "-metric", "MSD");
    const char* split = GetOpt(objc, objv, "-split", "WeightedMSD");
    const char* anchors = GetOpt(objc, objv, "-anchors", "NANI");
    const char* kinit = GetOpt(objc, objv, "-kinit", "StratAll");
    int refine = HasFlag(objc, objv, "-refine");
    double threshold = GetOptDouble(objc, objv, "-threshold", 0.0);
    const char* end_mode = GetOpt(objc, objv, "-end-mode", "k");
    int percentage = GetOptInt(objc, objv, "-percentage", 10);

    mdance_result_t* res = mdance_divine(
        coords.data(), nframes, ncols, natoms, nclusters,
        metric, split, anchors, kinit, refine, threshold, end_mode, percentage);

    const char* err = mdance_result_error(res);
    if (err) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(err, -1));
        mdance_result_free(res);
        return TCL_ERROR;
    }

    Tcl_SetObjResult(interp, BuildResultDict(interp, res, "divine"));
    mdance_result_free(res);
    return TCL_OK;
}

// mdance::helm coords nframes natoms nclusters initial_labels ?options...?
static int HelmCmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 6) {
        Tcl_WrongNumArgs(interp, 1, objv,
            "coords nframes natoms nclusters initial_labels "
            "?-metric MSD? ?-merge-scheme Inter? ?-eps -1? "
            "?-trim-start? ?-min-samples 0.01? ?-trim-val 0? ?-trim-k 0?");
        return TCL_ERROR;
    }

    std::vector<double> coords;
    if (GetDoubleList(interp, objv[1], coords) != TCL_OK) {
        return TCL_ERROR;
    }

    int nframes, natoms, nclusters;
    if (Tcl_GetIntFromObj(interp, objv[2], &nframes) != TCL_OK ||
        Tcl_GetIntFromObj(interp, objv[3], &natoms) != TCL_OK ||
        Tcl_GetIntFromObj(interp, objv[4], &nclusters) != TCL_OK) {
        return TCL_ERROR;
    }
    if (nframes <= 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("nframes must be positive", -1));
        return TCL_ERROR;
    }

    int ncols = coords.size() / nframes;

    std::vector<int> initial_labels;
    if (GetIntList(interp, objv[5], initial_labels) != TCL_OK) {
        return TCL_ERROR;
    }

    const char* metric = GetOpt(objc, objv, "-metric", "MSD");
    const char* merge_scheme = GetOpt(objc, objv, "-merge-scheme", "Inter");
    float eps = (float)GetOptDouble(objc, objv, "-eps", -1.0);
    int trim_start = HasFlag(objc, objv, "-trim-start");
    float min_samples = (float)GetOptDouble(objc, objv, "-min-samples", 0.01);
    float trim_val = (float)GetOptDouble(objc, objv, "-trim-val", 0.0);
    float trim_k = (float)GetOptDouble(objc, objv, "-trim-k", 0.0);

    mdance_result_t* res = mdance_helm(
        coords.data(), nframes, ncols, natoms, nclusters,
        initial_labels.data(), initial_labels.size(),
        metric, merge_scheme, eps, trim_start, min_samples, trim_val, trim_k);

    const char* err = mdance_result_error(res);
    if (err) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(err, -1));
        mdance_result_free(res);
        return TCL_ERROR;
    }

    Tcl_SetObjResult(interp, BuildResultDict(interp, res, "helm"));
    mdance_result_free(res);
    return TCL_OK;
}

// mdance::equal coords nframes natoms ?-metric MSD? -threshold <f> ?options...?
static int EqualCmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv,
            "coords nframes natoms -threshold <f> ?-metric MSD? ?-seed-method medoid? "
            "?-n-seeds 1? ?-percentage 10? ?-min-samples 10? ?-check-sim? "
            "?-sim-threshold 0? ?-reject-lowd? ?-align none?");
        return TCL_ERROR;
    }

    std::vector<double> coords;
    if (GetDoubleList(interp, objv[1], coords) != TCL_OK) {
        return TCL_ERROR;
    }
    int nframes, natoms;
    if (Tcl_GetIntFromObj(interp, objv[2], &nframes) != TCL_OK ||
        Tcl_GetIntFromObj(interp, objv[3], &natoms) != TCL_OK) {
        return TCL_ERROR;
    }
    if (nframes <= 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("nframes must be positive", -1));
        return TCL_ERROR;
    }
    int ncols = coords.size() / nframes;

    const char* metric = GetOpt(objc, objv, "-metric", "MSD");
    double threshold = GetOptDouble(objc, objv, "-threshold", -1.0);
    if (threshold < 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("eQUAL requires -threshold >= 0", -1));
        return TCL_ERROR;
    }
    const char* seed = GetOpt(objc, objv, "-seed-method", "medoid");
    double nSeeds = GetOptDouble(objc, objv, "-n-seeds", 1.0);
    int checkSim = HasFlag(objc, objv, "-check-sim");
    double simThreshold = GetOptDouble(objc, objv, "-sim-threshold", 0.0);
    int rejectLowd = HasFlag(objc, objv, "-reject-lowd");
    double minSamples = GetOptDouble(objc, objv, "-min-samples", 10.0);
    int percentage = GetOptInt(objc, objv, "-percentage", 10);
    const char* align = GetOpt(objc, objv, "-align", "none");

    mdance_result_t* res = mdance_equal(
        coords.data(), nframes, ncols, natoms, metric, threshold, seed,
        nSeeds, checkSim, simThreshold, rejectLowd, minSamples, percentage, align);

    const char* err = mdance_result_error(res);
    if (err) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(err, -1));
        mdance_result_free(res);
        return TCL_ERROR;
    }
    Tcl_SetObjResult(interp, BuildResultDict(interp, res, "equal"));
    mdance_result_free(res);
    return TCL_OK;
}

// mdance::analysis coords nframes natoms ?-metric MSD? ?-labels {...}?
static int AnalysisCmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv,
            "coords nframes natoms ?-metric MSD? ?-labels {labels}?");
        return TCL_ERROR;
    }

    std::vector<double> coords;
    if (GetDoubleList(interp, objv[1], coords) != TCL_OK) {
        return TCL_ERROR;
    }

    int nframes, natoms;
    if (Tcl_GetIntFromObj(interp, objv[2], &nframes) != TCL_OK ||
        Tcl_GetIntFromObj(interp, objv[3], &natoms) != TCL_OK) {
        return TCL_ERROR;
    }
    if (nframes <= 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("nframes must be positive", -1));
        return TCL_ERROR;
    }
    int ncols = coords.size() / nframes;

    const char* metric = GetOpt(objc, objv, "-metric", "MSD");

    std::vector<int> labels;
    for (int i = 0; i < objc - 1; ++i) {
        if (strcmp(Tcl_GetString(objv[i]), "-labels") == 0) {
            if (GetIntList(interp, objv[i + 1], labels) != TCL_OK) {
                return TCL_ERROR;
            }
            break;
        }
    }

    mdance_analysis_t* a = mdance_analysis(
        coords.data(), nframes, ncols, natoms, metric,
        labels.empty() ? nullptr : labels.data(), (int)labels.size());

    const char* err = mdance_analysis_error(a);
    if (err) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(err, -1));
        mdance_analysis_free(a);
        return TCL_ERROR;
    }

    Tcl_Obj* dict = Tcl_NewDictObj();
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("isim", -1),
        Tcl_NewDoubleObj(mdance_analysis_isim(a)));
    int nc = mdance_analysis_nclusters(a);
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("nClusters", -1), Tcl_NewIntObj(nc));

    const double* ci = mdance_analysis_cluster_isim(a);
    if (ci) {
        Tcl_Obj* l = Tcl_NewListObj(0, nullptr);
        for (int i = 0; i < nc; ++i) Tcl_ListObjAppendElement(interp, l, Tcl_NewDoubleObj(ci[i]));
        Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("clusterISIM", -1), l);
    }
    const int* co = mdance_analysis_cluster_outliers(a);
    if (co) {
        Tcl_Obj* l = Tcl_NewListObj(0, nullptr);
        for (int i = 0; i < nc; ++i) Tcl_ListObjAppendElement(interp, l, Tcl_NewIntObj(co[i]));
        Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("clusterOutliers", -1), l);
    }

    Tcl_SetObjResult(interp, dict);
    mdance_analysis_free(a);
    return TCL_OK;
}

// mdance::prime coords nframes natoms ?-metric RR? -labels {...} ?-trim-frac 0? ?-weighted?
static int PrimeCmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv,
            "coords nframes natoms -labels {labels} ?-metric RR? ?-trim-frac 0? ?-weighted?");
        return TCL_ERROR;
    }

    std::vector<double> coords;
    if (GetDoubleList(interp, objv[1], coords) != TCL_OK) {
        return TCL_ERROR;
    }
    int nframes, natoms;
    if (Tcl_GetIntFromObj(interp, objv[2], &nframes) != TCL_OK ||
        Tcl_GetIntFromObj(interp, objv[3], &natoms) != TCL_OK) {
        return TCL_ERROR;
    }
    if (nframes <= 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("nframes must be positive", -1));
        return TCL_ERROR;
    }
    int ncols = coords.size() / nframes;

    const char* metric = GetOpt(objc, objv, "-metric", "RR");
    double trimFrac = GetOptDouble(objc, objv, "-trim-frac", 0.0);
    int weighted = HasFlag(objc, objv, "-weighted");

    std::vector<int> labels;
    for (int i = 0; i < objc - 1; ++i) {
        if (strcmp(Tcl_GetString(objv[i]), "-labels") == 0) {
            if (GetIntList(interp, objv[i + 1], labels) != TCL_OK) return TCL_ERROR;
            break;
        }
    }
    if (labels.empty()) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("PRIME requires -labels", -1));
        return TCL_ERROR;
    }

    mdance_prime_t* p = mdance_prime(
        coords.data(), nframes, ncols, natoms, metric,
        labels.data(), (int)labels.size(), trimFrac, weighted);

    const char* err = mdance_prime_error(p);
    if (err) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(err, -1));
        mdance_prime_free(p);
        return TCL_ERROR;
    }

    Tcl_Obj* dict = Tcl_NewDictObj();
    auto put = [&](const char* k, int v) {
        Tcl_DictObjPut(interp, dict, Tcl_NewStringObj(k, -1), Tcl_NewIntObj(v));
    };
    put("pairwise", mdance_prime_pairwise(p));
    put("union", mdance_prime_union(p));
    put("medoid", mdance_prime_medoid(p));
    put("outlier", mdance_prime_outlier(p));
    put("medoidAll", mdance_prime_medoid_all(p));
    put("medoidC0", mdance_prime_medoid_c0(p));
    put("medoidC0Trimmed", mdance_prime_medoid_c0_trimmed(p));
    put("nClusters", mdance_prime_nclusters(p));
    Tcl_SetObjResult(interp, dict);
    mdance_prime_free(p);
    return TCL_OK;
}

// mdance::select coords nframes natoms ?-metric MSD? ?-method diversity? ?-param 10? ?-nbins 10?
static int SelectCmd(ClientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv,
            "coords nframes natoms ?-metric MSD? ?-method diversity? ?-param 10? ?-nbins 10?");
        return TCL_ERROR;
    }
    std::vector<double> coords;
    if (GetDoubleList(interp, objv[1], coords) != TCL_OK) return TCL_ERROR;
    int nframes, natoms;
    if (Tcl_GetIntFromObj(interp, objv[2], &nframes) != TCL_OK ||
        Tcl_GetIntFromObj(interp, objv[3], &natoms) != TCL_OK) {
        return TCL_ERROR;
    }
    if (nframes <= 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("nframes must be positive", -1));
        return TCL_ERROR;
    }
    int ncols = coords.size() / nframes;

    const char* metric = GetOpt(objc, objv, "-metric", "MSD");
    const char* method = GetOpt(objc, objv, "-method", "diversity");
    double param = GetOptDouble(objc, objv, "-param", 10.0);
    int nbins = GetOptInt(objc, objv, "-nbins", 10);

    mdance_select_t* s = mdance_select(
        coords.data(), nframes, ncols, natoms, metric, method, param, nbins);

    const char* err = mdance_select_error(s);
    if (err) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(err, -1));
        mdance_select_free(s);
        return TCL_ERROR;
    }
    int n = mdance_select_count(s);
    const int* idx = mdance_select_indices(s);
    Tcl_Obj* list = Tcl_NewListObj(0, nullptr);
    for (int i = 0; i < n; ++i) {
        Tcl_ListObjAppendElement(interp, list, Tcl_NewIntObj(idx[i]));
    }
    Tcl_SetObjResult(interp, list);
    mdance_select_free(s);
    return TCL_OK;
}

// mdance::version
static int VersionCmd(ClientData, Tcl_Interp* interp, int, Tcl_Obj* const[]) {
    Tcl_SetObjResult(interp, Tcl_NewStringObj("1.0", -1));
    return TCL_OK;
}

// Entry point: called by Tcl's "load" command
extern "C" int Mdance_Init(Tcl_Interp* interp) {
    if (Tcl_InitStubs(interp, "8.5", 0) == nullptr) {
        return TCL_ERROR;
    }

    Tcl_CreateNamespace(interp, "::mdance", nullptr, nullptr);

    Tcl_CreateObjCommand(interp, "::mdance::kmeans", KmeansCmd, nullptr, nullptr);
    Tcl_CreateObjCommand(interp, "::mdance::divine", DivineCmd, nullptr, nullptr);
    Tcl_CreateObjCommand(interp, "::mdance::helm", HelmCmd, nullptr, nullptr);
    Tcl_CreateObjCommand(interp, "::mdance::equal", EqualCmd, nullptr, nullptr);
    Tcl_CreateObjCommand(interp, "::mdance::analysis", AnalysisCmd, nullptr, nullptr);
    Tcl_CreateObjCommand(interp, "::mdance::prime", PrimeCmd, nullptr, nullptr);
    Tcl_CreateObjCommand(interp, "::mdance::select", SelectCmd, nullptr, nullptr);
    Tcl_CreateObjCommand(interp, "::mdance::version", VersionCmd, nullptr, nullptr);

    Tcl_PkgProvide(interp, "mdance_native", "1.0");
    return TCL_OK;
}
