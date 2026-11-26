#ifndef MDANCE_CAPI_H
#define MDANCE_CAPI_H

/* The C interface of libmdance: MDANCE's clustering, frame selection and
 * similarity analysis for any program that can call C (VMD's Tcl extension,
 * Python's ctypes, a C or C++ application).
 *
 * Two layers:
 *
 *   1. Datasets, options and controls (API version 2). The coordinates are
 *      loaded once into an mdance_dataset_t (from rows, frame by frame from
 *      a program's own float arrays, or from a DCD file), checked as they
 *      come in, optionally superposed, and then clustered as often as
 *      wanted: a scan over k reads them once. The parameters are named
 *      options (mdance_options_t), each checked before anything runs; an
 *      option the algorithm does not take is an error, not ignored. An
 *      mdance_control_t lets another thread cancel a call and read how far
 *      it is.
 *
 *   2. The first interface (mdance_kmeans, mdance_helm, ...): one call per
 *      algorithm on a row-major buffer of doubles, positional arguments. It
 *      is kept as it was, and now runs through the same checks.
 *
 * Errors. Nothing here aborts, exits or prints. A call that returns an object
 * (a result, an analysis, ...) returns one even when it fails: its *_error()
 * is then the message, else NULL. A call that returns a pointer it created
 * (mdance_dataset_*, mdance_options_new, mdance_control_new) returns NULL on
 * failure; a call that returns an int returns 0 on success and -1 on
 * failure. The message of the last such failure on this thread is
 * mdance_last_error(). Out of memory is an error like any other.
 *
 * Threads. A call uses only its arguments: calls on different threads run
 * concurrently. A dataset may be read by several calls at once; it must not
 * be changed (set_frame, superpose, free) while a call reads it. Results are
 * owned by the caller and freed with their *_free(), which accept NULL.
 *
 * Frames and indices. A dataset's rows are frames 0..nframes-1, in the order
 * they were given; every index a result holds is such a row (or -1: none).
 */

#include <stddef.h>

#if defined(MDANCE_STATIC)
  #define MDANCE_API
#elif defined(_WIN32) || defined(__CYGWIN__)
  #if defined(MDANCE_BUILDING)
    #define MDANCE_API __declspec(dllexport)
  #else
    #define MDANCE_API __declspec(dllimport)
  #endif
#else
  #define MDANCE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MDANCE_API_VERSION 2

/* The library's version ("2.0.0") and API version (MDANCE_API_VERSION of the
 * header it was built with): a program built against a later header checks
 * this before using what that header adds. */
MDANCE_API const char*   mdance_version(void);
MDANCE_API int           mdance_api_version(void);
/* 1 if DIVINE is in this build (BUILD_DIVINE), else 0. */
MDANCE_API int           mdance_has_divine(void);
/* The message of the last failure of a call returning NULL or -1 on this
 * thread ("" if none). Valid until the next such call on the thread. */
MDANCE_API const char*   mdance_last_error(void);

typedef struct mdance_result mdance_result_t;
typedef struct mdance_analysis mdance_analysis_t;
typedef struct mdance_prime mdance_prime_t;
typedef struct mdance_select mdance_select_t;

/* ======================================================================
 * Datasets
 * ====================================================================== */

typedef struct mdance_dataset mdance_dataset_t;

/* A dataset of nframes frames of natoms atoms (3 * natoms columns: x y z of
 * each atom), every frame still to be set. */
MDANCE_API mdance_dataset_t* mdance_dataset_new(int nframes, int natoms);
/* The same with ncols columns of any kind (natoms divides the MSD: 1 for data
 * that are not atoms). */
MDANCE_API mdance_dataset_t* mdance_dataset_new_cols(int nframes, int ncols, int natoms);
/* A dataset copied from a row-major buffer (frame after frame, ncols values
 * each). */
MDANCE_API mdance_dataset_t* mdance_dataset_from_rows(const double* rows, int nframes, int ncols, int natoms);

/* Set a frame's ncols values (n must be ncols). Values that are not finite
 * are refused, naming the frame and column. */
MDANCE_API int mdance_dataset_set_frame(mdance_dataset_t* d, int frame, const double* values, int n);
MDANCE_API int mdance_dataset_set_frame_float(mdance_dataset_t* d, int frame, const float* values, int n);
/* Set a frame from a program's array of every atom's x y z (floats, as VMD
 * keeps a timestep: pos[3*i], pos[3*i+1], pos[3*i+2]), taking the atoms
 * atoms[0..n-1] in that order (n must be the dataset's natoms; each index in
 * 0..natoms_in_pos-1). */
MDANCE_API int mdance_dataset_set_frame_atoms_float(mdance_dataset_t* d, int frame,
                                                    const float* pos, int natoms_in_pos,
                                                    const int* atoms, int n);

/* Read frames of a DCD file (CHARMM or X-PLOR format, either byte order, as
 * VMD's `animate write dcd` writes it): the frames first, first+stride, ...
 * up to last (last -1: the last frame), and of each the atoms atoms[0..n-1]
 * (atoms NULL or n 0: every atom). The file is checked record by record: a
 * file cut short, a record of the wrong length or a frame count it does not
 * hold is an error, never a read past its end. */
MDANCE_API mdance_dataset_t* mdance_dataset_read_dcd(const char* path, const int* atoms, int n,
                                                     int first, int last, int stride);

/* Superpose every frame on frame ref_frame by the atoms fit[0..nfit-1]
 * (indices of the dataset's atoms; NULL or 0: every atom): the rotation and
 * translation of least RMSD of those atoms, every atom weighing the same,
 * applied to all the frame's atoms. Needs 3 columns an atom. */
MDANCE_API int mdance_dataset_superpose(mdance_dataset_t* d, const int* fit, int nfit, int ref_frame);

MDANCE_API int mdance_dataset_nframes(const mdance_dataset_t* d);
MDANCE_API int mdance_dataset_ncols(const mdance_dataset_t* d);
MDANCE_API int mdance_dataset_natoms(const mdance_dataset_t* d);
/* How many frames have been set (the algorithms need all of them). */
MDANCE_API int mdance_dataset_nset(const mdance_dataset_t* d);
/* Copy a frame's values into out (n must be ncols). */
MDANCE_API int mdance_dataset_get_frame(const mdance_dataset_t* d, int frame, double* out, int n);
MDANCE_API void mdance_dataset_free(mdance_dataset_t* d);

/* ======================================================================
 * Options
 * ====================================================================== */

/* Named parameters of a call, as text ("10", "MSD", "0.25") or lists of
 * whole numbers (labels). The call reads them, checks every one (a number
 * that is not one, out of range, or a name it does not take is an error that
 * names the option), and leaves them unchanged: one options object can serve
 * many calls. Setting a name again replaces its value. */
typedef struct mdance_options mdance_options_t;

MDANCE_API mdance_options_t* mdance_options_new(void);
MDANCE_API int  mdance_options_set(mdance_options_t* o, const char* name, const char* value);
MDANCE_API int  mdance_options_set_int(mdance_options_t* o, const char* name, long long value);
MDANCE_API int  mdance_options_set_double(mdance_options_t* o, const char* name, double value);
MDANCE_API int  mdance_options_set_ints(mdance_options_t* o, const char* name, const int* values, int n);
MDANCE_API int  mdance_options_unset(mdance_options_t* o, const char* name);
MDANCE_API void mdance_options_free(mdance_options_t* o);

/* ======================================================================
 * Controls: cancelling a call, and how far it is
 * ====================================================================== */

typedef struct mdance_control mdance_control_t;

MDANCE_API mdance_control_t* mdance_control_new(void);
/* Ask the call using the control to stop (from any thread): it then fails
 * with the error "cancelled" at its next check, within a fraction of a
 * second for the long loops. */
MDANCE_API void        mdance_control_cancel(mdance_control_t* c);
MDANCE_API int         mdance_control_cancelled(const mdance_control_t* c);
/* The fraction done of the current stage (0..1) and its name. */
MDANCE_API double      mdance_control_progress(const mdance_control_t* c);
MDANCE_API const char* mdance_control_stage(const mdance_control_t* c);
/* Clear the cancel flag and the progress, to use the control again. */
MDANCE_API void        mdance_control_reset(mdance_control_t* c);
MDANCE_API void        mdance_control_free(mdance_control_t* c);

/* ======================================================================
 * Calls on a dataset (control may be NULL)
 * ====================================================================== */

/* Cluster the frames. algorithm and its options:
 *
 *  "kmeans"  k (required), metric MSD, kinit StratAll
 *            (StratAll StratReduced CompSim DivSelect KmeansPP Random
 *            VanillaKmeansPP), percentage 10 (0..100), seed -1 (-1: from the
 *            clock; the Random and k-means++ seedings only; the seed used is
 *            mdance_result_seed)
 *  "divine"  k (required; 0: until end_mode), metric MSD, split WeightedMSD
 *            (MSD Radius WeightedMSD), anchors NANI (NANI OutlierPair
 *            SplinterPair), kinit StratAll, refine 0|1 (1), threshold 0
 *            (0..1), end_mode k (k points), percentage 10
 *  "helm"    labels (required: a starting cluster for every frame), k (0
 *            with eps), eps -1, metric MSD, merge_scheme Inter (Inter Intra
 *            Half), trim_start 0|1, min_samples 0.01, trim_val 0, trim_k 0
 *  "equal"   threshold (required, >= 0), metric MSD, seed_method medoid
 *            (medoid comp_sim), n_seeds 1, check_sim 0|1, sim_threshold 0,
 *            reject_lowd 0|1, min_samples 10, percentage 10, align none
 *
 * metric is one of MSD BUB Fai Gle Ja JT RT RR SM SS1 SS2. */
MDANCE_API mdance_result_t* mdance_cluster(const mdance_dataset_t* d, const char* algorithm,
                                           const mdance_options_t* o, mdance_control_t* c);

/* iSIM of every frame, and of each cluster with labels: metric MSD,
 * labels (optional; one per frame, -1: none). */
MDANCE_API mdance_analysis_t* mdance_analyze(const mdance_dataset_t* d, const mdance_options_t* o,
                                             mdance_control_t* c);

/* PRIME: labels (required), metric RR (RR SM), trim_frac 0 (0..1),
 * weighted 0|1 (0). */
MDANCE_API mdance_prime_t* mdance_prime_run(const mdance_dataset_t* d, const mdance_options_t* o,
                                            mdance_control_t* c);

/* Frame selection: method diversity (diversity outliers repsample medoid
 * outlier), metric MSD, param 10, nbins 10 (see mdance_select below). */
MDANCE_API mdance_select_t* mdance_select_run(const mdance_dataset_t* d, const mdance_options_t* o,
                                              mdance_control_t* c);

/* A scan: the same clustering at every k of ks[0..nks-1] ("kmeans",
 * "divine" or "helm", with that algorithm's options but k), the values of k
 * run at once on threads (option "threads": how many, 0 or absent: one a
 * processor, never more than the values of k). Each value of k gives the
 * result a call of mdance_cluster with it would give, to the last digit: the
 * threads change the time only. A value of k that fails (too few frames to
 * seed it, say) has its error in its own result; the others are made. The
 * threads share the dataset, which is not copied. A control cancels every
 * value of k still running or waiting; its progress is the share of the
 * values done. mdance_scan_error is the scan's own failure (no dataset, no
 * such algorithm, no k given, "cancelled"), else NULL. */
typedef struct mdance_scan mdance_scan_t;

MDANCE_API mdance_scan_t* mdance_scan_run(const mdance_dataset_t* d, const char* algorithm,
                                          const int* ks, int nks,
                                          const mdance_options_t* o, mdance_control_t* c);
MDANCE_API int                    mdance_scan_count(const mdance_scan_t* s);
MDANCE_API int                    mdance_scan_k(const mdance_scan_t* s, int i);
MDANCE_API const mdance_result_t* mdance_scan_result(const mdance_scan_t* s, int i);
MDANCE_API int                    mdance_scan_threads(const mdance_scan_t* s);
MDANCE_API const char*            mdance_scan_error(const mdance_scan_t* s);
MDANCE_API void                   mdance_scan_free(mdance_scan_t* s);

/* ======================================================================
 * The first interface: coords is row-major, nframes rows of ncols values
 * ====================================================================== */

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
/* The k-means seed used by the seedings that draw (Random, KmeansPP,
 * VanillaKmeansPP); -1 for the other seedings and algorithms. */
MDANCE_API int           mdance_result_seed(const mdance_result_t* r);

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
 * empty clusters get isim 0 and outlier -1. A label is -1 (none) or
 * 0..nframes-1. */
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
