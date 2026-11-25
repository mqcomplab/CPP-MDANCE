/* mdance_bench: time every C-API entry point on a CSV of coordinates and
 * write what each returns, so that two builds can be compared for speed and
 * for identical answers.
 *
 *   mdance_bench data.csv natoms out.txt [repeat]
 *
 * The CSV has one frame per line, 3*natoms comma-separated numbers.
 * out.txt gets one line per call: its name, then what it returned (labels,
 * sizes, representatives, MSDs and scores at full precision). Compare two
 * runs with `diff`. The timings go to standard output.
 *
 * Written in C against mdance_capi.h only, so it builds against any version
 * of the library that has these entry points. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mdance_capi.h"

#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
#else
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
#endif

static double* read_csv(const char* path, int* nframes, int* ncols) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    size_t cap = 1 << 20, n = 0;
    double* v = malloc(cap * sizeof *v);
    int cols = 0, rows = 0, c = 0;
    char buf[64];
    int bl = 0, ch;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == ',' || ch == '\n' || ch == '\r') {
            if (bl) {
                buf[bl] = 0;
                if (n == cap) { cap *= 2; v = realloc(v, cap * sizeof *v); }
                v[n++] = strtod(buf, NULL);
                bl = 0; c++;
            }
            if (ch == '\n' && c) {
                if (!cols) cols = c;
                else if (c != cols) { fprintf(stderr, "row %d has %d columns, not %d\n", rows, c, cols); exit(1); }
                rows++; c = 0;
            }
        } else if (bl < 63) {
            buf[bl++] = (char)ch;
        }
    }
    if (bl) { buf[bl] = 0; v[n++] = strtod(buf, NULL); c++; }
    if (c) { if (!cols) cols = c; rows++; }
    fclose(f);
    *nframes = rows; *ncols = cols;
    return v;
}

static FILE* out;
static int repeat = 1;

static void put_ints(const char* k, const int* a, int n) {
    fprintf(out, " %s=", k);
    if (!a) { fprintf(out, "null"); return; }
    for (int i = 0; i < n; i++) fprintf(out, "%s%d", i ? "," : "", a[i]);
}
static void put_dbls(const char* k, const double* a, int n) {
    fprintf(out, " %s=", k);
    if (!a) { fprintf(out, "null"); return; }
    for (int i = 0; i < n; i++) fprintf(out, "%s%.10g", i ? "," : "", a[i]);
}

static void report(const char* name, double ms) {
    printf("%-34s %10.1f ms\n", name, ms / repeat);
    fflush(stdout);
}

static void put_result(const char* name, mdance_result_t* r, double ms) {
    report(name, ms);
    fprintf(out, "%s", name);
    const char* e = mdance_result_error(r);
    if (e) { fprintf(out, " error=%s\n", e); return; }
    int nf = mdance_result_nframes(r), k = mdance_result_nclusters(r);
    fprintf(out, " k=%d", k);
    put_ints("labels", mdance_result_labels(r), nf);
    put_ints("sizes", mdance_result_cluster_sizes(r), k);
    put_ints("reps", mdance_result_representatives(r), k);
    put_dbls("msd", mdance_result_cluster_msd(r), k);
    fprintf(out, " ch=%.10g db=%.10g", mdance_result_ch_score(r), mdance_result_db_score(r));
    int zr = mdance_result_zmatrix_rows(r), zc = mdance_result_zmatrix_cols(r);
    if (zr) put_dbls("z", mdance_result_zmatrix(r), zr * zc);
    fprintf(out, "\n");
    fflush(out);
}

#define TIME(expr, var) do { double t0_ = now_ms(); for (int r_ = 0; r_ < repeat; r_++) { if (r_) { FREE_##var; } var = (expr); } ms = now_ms() - t0_; } while (0)
#define FREE_r mdance_result_free(r)
#define FREE_a mdance_analysis_free(a)
#define FREE_p mdance_prime_free(p)
#define FREE_s mdance_select_free(s)

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s data.csv natoms out.txt [repeat] [only]\n", argv[0]);
        return 2;
    }
    int nf, nc;
    int na = atoi(argv[2]);
    if (argc > 4) repeat = atoi(argv[4]);
    const char* only = argc > 5 ? argv[5] : "";
    double t0 = now_ms();
    double* x = read_csv(argv[1], &nf, &nc);
    printf("%d frames, %d columns, %d atoms (read in %.0f ms)\n", nf, nc, na, now_ms() - t0);
    out = fopen(argv[3], "w");
    if (!out) { perror(argv[3]); return 1; }

    double ms;
    mdance_result_t* r;
    mdance_analysis_t* a;
    mdance_prime_t* p;
    mdance_select_t* s;
    char name[64];
#define WANT(tag) (!*only || strstr(only, tag))

    if (WANT("kmeans")) {
        static const char* kinits[] = {"StratAll", "StratReduced", "CompSim", "DivSelect"};
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof name, "kmeans k=10 %s", kinits[i]);
            TIME(mdance_kmeans(x, nf, nc, na, 10, "MSD", kinits[i], 10), r);
            put_result(name, r, ms);
            mdance_result_free(r);
        }
        /* a scan, as the plugin's elbow runs it */
        double tot = 0;
        for (int k = 2; k <= 20; k++) {
            double t = now_ms();
            r = mdance_kmeans(x, nf, nc, na, k, "MSD", "StratAll", 10);
            tot += now_ms() - t;
            snprintf(name, sizeof name, "scan kmeans k=%d", k);
            fprintf(out, "%s ch=%.10g db=%.10g\n", name, mdance_result_ch_score(r), mdance_result_db_score(r));
            mdance_result_free(r);
        }
        report("kmeans scan k=2..20 (total)", tot * repeat);
#if defined(MDANCE_API_VERSION) && MDANCE_API_VERSION >= 2
        /* the same scan as one call, its values of k on threads; each
         * value's scores must be the single calls' */
        {
            mdance_dataset_t* d = mdance_dataset_from_rows(x, nf, nc, na);
            int ks[19];
            for (int i = 0; i < 19; i++) ks[i] = i + 2;
            static const char* th[] = {"1", "2", "4", "0"};
            for (int t = 0; t < 4; t++) {
                mdance_options_t* o = mdance_options_new();
                mdance_options_set(o, "kinit", "StratAll");
                mdance_options_set(o, "percentage", "10");
                mdance_options_set(o, "threads", th[t]);
                double t0 = now_ms();
                mdance_scan_t* s = mdance_scan_run(d, "kmeans", ks, 19, o, NULL);
                double ms = now_ms() - t0;
                snprintf(name, sizeof name, "kmeans scan, %d thread%s", mdance_scan_threads(s),
                         mdance_scan_threads(s) > 1 ? "s" : "");
                report(name, ms * repeat);
                for (int i = 0; i < mdance_scan_count(s); i++) {
                    const mdance_result_t* ri = mdance_scan_result(s, i);
                    fprintf(out, "scan kmeans k=%d ch=%.10g db=%.10g\n", mdance_scan_k(s, i),
                            mdance_result_ch_score(ri), mdance_result_db_score(ri));
                }
                mdance_scan_free(s);
                mdance_options_free(o);
            }
            mdance_dataset_free(d);
        }
#endif
    }
    if (WANT("helm")) {
        r = mdance_kmeans(x, nf, nc, na, 40, "MSD", "StratAll", 50);
        int* init = malloc(nf * sizeof *init);
        if (mdance_result_labels(r)) memcpy(init, mdance_result_labels(r), nf * sizeof *init);
        else for (int i = 0; i < nf; i++) init[i] = i % 40;
        mdance_result_free(r);
        TIME(mdance_helm(x, nf, nc, na, 5, init, nf, "MSD", "Inter", -1.0f, 0, 0.01f, 0, 0), r);
        put_result("helm k=5 from 40", r, ms);
        mdance_result_free(r);
        TIME(mdance_helm(x, nf, nc, na, 5, init, nf, "MSD", "Intra", -1.0f, 1, 0.01f, 0, 0), r);
        put_result("helm k=5 Intra trim-start", r, ms);
        mdance_result_free(r);
        free(init);
    }
    if (WANT("divine")) {
        static const char* anchors[] = {"NANI", "OutlierPair", "SplinterPair"};
        for (int i = 0; i < 3; i++) {
            snprintf(name, sizeof name, "divine k=10 %s", anchors[i]);
            TIME(mdance_divine(x, nf, nc, na, 10, "MSD", "WeightedMSD", anchors[i], "StratAll", 0, 0.0, "k", 10), r);
            put_result(name, r, ms);
            mdance_result_free(r);
        }
        TIME(mdance_divine(x, nf, nc, na, 10, "MSD", "MSD", "OutlierPair", "StratAll", 1, 0.0, "k", 10), r);
        put_result("divine k=10 OutlierPair refine", r, ms);
        mdance_result_free(r);
    }
    if (WANT("equal")) {
        const char* th = getenv("BENCH_EQUAL_THRESHOLDS");
        double ths[8]; int nth = 0;
        if (th) { char* e; const char* q = th; while (nth < 8) { double v = strtod(q, &e); if (e == q) break; ths[nth++] = v; q = e; while (*q == ',' || *q == ' ') q++; } }
        if (!nth) { ths[0] = 2.0; nth = 1; }
        for (int i = 0; i < nth; i++) {
            snprintf(name, sizeof name, "equal medoid threshold=%g", ths[i]);
            TIME(mdance_equal(x, nf, nc, na, "MSD", ths[i], "medoid", 1, 0, 0, 0, 10, 10, "none"), r);
            put_result(name, r, ms);
            mdance_result_free(r);
            snprintf(name, sizeof name, "equal comp_sim threshold=%g", ths[i]);
            /* (BENCH_SKIP_COMPSIM: leave out the slowest call, eQUAL comp_sim,
             * which takes tens of minutes on the stock library) */
            if (getenv("BENCH_SKIP_COMPSIM")) continue;
            TIME(mdance_equal(x, nf, nc, na, "MSD", ths[i], "comp_sim", 1, 0, 0, 0, 10, 10, "none"), r);
            put_result(name, r, ms);
            mdance_result_free(r);
        }
    }
    if (WANT("analysis") || WANT("prime")) {
        r = mdance_kmeans(x, nf, nc, na, 10, "MSD", "StratAll", 10);
        int* lab = malloc(nf * sizeof *lab);
        if (mdance_result_labels(r)) memcpy(lab, mdance_result_labels(r), nf * sizeof *lab);
        else for (int i = 0; i < nf; i++) lab[i] = i % 10;
        mdance_result_free(r);
        if (WANT("analysis")) {
            TIME(mdance_analysis(x, nf, nc, na, "MSD", lab, nf), a);
            report("analysis MSD", ms);
            fprintf(out, "analysis isim=%.10g", mdance_analysis_isim(a));
            put_dbls("cisim", mdance_analysis_cluster_isim(a), mdance_analysis_nclusters(a));
            put_ints("out", mdance_analysis_cluster_outliers(a), mdance_analysis_nclusters(a));
            fprintf(out, "\n");
            mdance_analysis_free(a);
            TIME(mdance_analysis(x, nf, nc, na, "RR", lab, nf), a);
            report("analysis RR", ms);
            fprintf(out, "analysis RR isim=%.10g", mdance_analysis_isim(a));
            put_dbls("cisim", mdance_analysis_cluster_isim(a), mdance_analysis_nclusters(a));
            put_ints("out", mdance_analysis_cluster_outliers(a), mdance_analysis_nclusters(a));
            fprintf(out, "\n");
            mdance_analysis_free(a);
        }
        if (WANT("prime")) {
            static const char* pm[] = {"RR", "SM"};
            for (int i = 0; i < 2; i++) {
                snprintf(name, sizeof name, "prime %s", pm[i]);
                TIME(mdance_prime(x, nf, nc, na, pm[i], lab, nf, 0.1, 0), p);
                report(name, ms);
                const char* e = mdance_prime_error(p);
                fprintf(out, "%s %s pw=%d un=%d me=%d ou=%d all=%d c0=%d c0t=%d k=%d\n", name, e ? e : "",
                        mdance_prime_pairwise(p), mdance_prime_union(p), mdance_prime_medoid(p),
                        mdance_prime_outlier(p), mdance_prime_medoid_all(p), mdance_prime_medoid_c0(p),
                        mdance_prime_medoid_c0_trimmed(p), mdance_prime_nclusters(p));
                mdance_prime_free(p);
            }
        }
        free(lab);
    }
    if (WANT("select")) {
        static const char* meth[] = {"diversity", "outliers", "repsample", "medoid", "outlier"};
        static const char* metr[] = {"MSD", "RR"};
        for (int j = 0; j < 2; j++)
        for (int i = 0; i < 5; i++) {
            snprintf(name, sizeof name, "select %s %s", meth[i], metr[j]);
            TIME(mdance_select(x, nf, nc, na, metr[j], meth[i], 10, 10), s);
            report(name, ms);
            const char* e = mdance_select_error(s);
            fprintf(out, "%s %s", name, e ? e : "");
            put_ints("idx", mdance_select_indices(s), mdance_select_count(s));
            fprintf(out, "\n");
            mdance_select_free(s);
        }
    }
    fclose(out);
    free(x);
    return 0;
}
