#ifndef MDANCE_CAPI_H
#define MDANCE_CAPI_H

/* Export macro. The shared library is built with hidden default visibility
 * (see capi/CMakeLists.txt), so every public C-API symbol must be explicitly
 * marked default-visible or it will not be linkable from the Tcl extension or
 * any other C-FFI consumer. */
#if defined(_WIN32) || defined(__CYGWIN__)
  #define MDANCE_API __declspec(dllexport)
#else
  #define MDANCE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mdance_result mdance_result_t;
typedef struct mdance_analysis mdance_analysis_t;

/* --- Run algorithms --- */

MDANCE_API mdance_result_t* mdance_kmeans(
    const double* coords, int nframes, int ncols,
    int natoms, int nclusters,
    const char* metric,
    const char* kinit,
    int percentage
);

MDANCE_API mdance_result_t* mdance_divine(
    const double* coords, int nframes, int ncols,
    int natoms, int nclusters,
    const char* metric,
    const char* split,
    const char* anchors,
    const char* kinit,
    int refine,
    double threshold,
    const char* end_mode,
    int percentage
);

MDANCE_API mdance_result_t* mdance_helm(
    const double* coords, int nframes, int ncols,
    int natoms, int nclusters,
    const int* initial_labels, int nlabels,
    const char* metric,
    const char* merge_scheme,
    float eps,
    int trim_start,
    float min_samples,
    float trim_val,
    float trim_k
);

/* eQUAL (Extended QUALity) radial/threshold clustering. The cluster count
 * emerges from `threshold` (there is no nclusters). seed_method is "comp_sim"
 * or "medoid" (sklearn methods unsupported). n_seeds / min_samples use the dual
 * typing convention: a value in (0,1) is a fraction of nframes, >=1 is a literal
 * count. threshold is REQUIRED. align is "none" (uni/kron unimplemented).
 * Results use the EXISTING mdance_result_t accessors; unclustered frames are -1. */
MDANCE_API mdance_result_t* mdance_equal(
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
    const char* align
);

/* --- Access results --- */

MDANCE_API int           mdance_result_nframes(const mdance_result_t* r);
MDANCE_API int           mdance_result_nclusters(const mdance_result_t* r);
MDANCE_API const int*    mdance_result_labels(const mdance_result_t* r);
MDANCE_API const int*    mdance_result_cluster_sizes(const mdance_result_t* r);
MDANCE_API const int*    mdance_result_representatives(const mdance_result_t* r);
MDANCE_API const double* mdance_result_cluster_msd(const mdance_result_t* r);
MDANCE_API double        mdance_result_ch_score(const mdance_result_t* r);
MDANCE_API double        mdance_result_db_score(const mdance_result_t* r);
MDANCE_API const char*   mdance_result_error(const mdance_result_t* r);

/* HELM z-matrix */
MDANCE_API int           mdance_result_zmatrix_rows(const mdance_result_t* r);
MDANCE_API int           mdance_result_zmatrix_cols(const mdance_result_t* r);
MDANCE_API const double* mdance_result_zmatrix(const mdance_result_t* r);

MDANCE_API void          mdance_result_free(mdance_result_t* r);

/* --- Extended-similarity (iSIM) analysis ---
 * Computes the ensemble extended comparison (iSIM) over all frames and, when
 * per-frame cluster labels are supplied (labels may be NULL / nlabels==0),
 * the per-cluster extended comparison ("compactness" in the chosen metric) and
 * the per-cluster outlier (least-similar member) frame index.
 * Per-cluster arrays are indexed 0..nclusters-1 (nclusters = max label + 1);
 * empty clusters get isim 0 and outlier -1. */
MDANCE_API mdance_analysis_t* mdance_analysis(
    const double* coords, int nframes, int ncols,
    int natoms,
    const char* metric,
    const int* labels, int nlabels
);

MDANCE_API double        mdance_analysis_isim(const mdance_analysis_t* a);
MDANCE_API int           mdance_analysis_nclusters(const mdance_analysis_t* a);
MDANCE_API const double* mdance_analysis_cluster_isim(const mdance_analysis_t* a);
MDANCE_API const int*    mdance_analysis_cluster_outliers(const mdance_analysis_t* a);
MDANCE_API const char*   mdance_analysis_error(const mdance_analysis_t* a);
MDANCE_API void          mdance_analysis_free(mdance_analysis_t* a);

/* --- PRIME: representative/"native"-like frame prediction from a clustered
 * ensemble. Consumes the coordinate matrix + per-frame cluster labels (REQUIRED;
 * nclusters inferred as max label + 1). metric is "RR" or "SM" only. trim_frac
 * drops the most-redundant fraction of the most-populated cluster before scoring
 * (0 = none). weighted_by_frames scales each cluster's score by its population.
 * All outputs are frame indices (rows of the input coords) or -1 if undefined. */
typedef struct mdance_prime mdance_prime_t;

MDANCE_API mdance_prime_t* mdance_prime(
    const double* coords, int nframes, int ncols,
    int natoms,
    const char* metric,
    const int* labels, int nlabels,
    double trim_frac,
    int weighted_by_frames
);

MDANCE_API int         mdance_prime_pairwise(const mdance_prime_t* p);
MDANCE_API int         mdance_prime_union(const mdance_prime_t* p);
MDANCE_API int         mdance_prime_medoid(const mdance_prime_t* p);
MDANCE_API int         mdance_prime_outlier(const mdance_prime_t* p);
MDANCE_API int         mdance_prime_medoid_all(const mdance_prime_t* p);
MDANCE_API int         mdance_prime_medoid_c0(const mdance_prime_t* p);
MDANCE_API int         mdance_prime_medoid_c0_trimmed(const mdance_prime_t* p);
MDANCE_API int         mdance_prime_nclusters(const mdance_prime_t* p);
MDANCE_API const char* mdance_prime_error(const mdance_prime_t* p);
MDANCE_API void        mdance_prime_free(mdance_prime_t* p);

/* --- Frame-selection tools (whole-trajectory, no clustering required).
 * method:
 *   "diversity" -> most diverse subset (param = percentage 1..100)
 *   "outliers"  -> the param most-dissimilar frames (param: >=1 count, (0,1) fraction)
 *   "repsample" -> representative sampling across comp-sim density bins
 *                  (param: count/fraction; nbins = number of bins)
 *   "medoid"    -> single most-central frame
 *   "outlier"   -> single most-dissimilar frame
 * Returns frame indices (rows of the input coords). */
typedef struct mdance_select mdance_select_t;

MDANCE_API mdance_select_t* mdance_select(
    const double* coords, int nframes, int ncols,
    int natoms,
    const char* metric,
    const char* method,
    double param,
    int nbins
);

MDANCE_API int         mdance_select_count(const mdance_select_t* s);
MDANCE_API const int*  mdance_select_indices(const mdance_select_t* s);
MDANCE_API const char* mdance_select_error(const mdance_select_t* s);
MDANCE_API void        mdance_select_free(mdance_select_t* s);

#ifdef __cplusplus
}
#endif

#endif
