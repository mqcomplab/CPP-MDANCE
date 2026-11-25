// Tests of the C interface (capi/mdance_capi.h), as another program uses it:
// datasets, options, controls, both interfaces' agreement, and every input
// that used to crash the library (each a test: it must now be an error).
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "mdance_capi.h"

namespace {

// Two blobs of `n` frames each, 3 atoms a frame, deterministic.
std::vector<double> blobs(int n, int natoms = 3, double gap = 10.0) {
    std::vector<double> v;
    for (int f = 0; f < 2 * n; ++f) {
        double off = f < n ? 0.0 : gap;
        for (int j = 0; j < 3 * natoms; ++j) v.push_back(off + 0.1 * std::sin(f * 1.7 + j * 0.9) + 0.01 * j);
    }
    return v;
}

std::string errOf(const mdance_result_t* r) {
    const char* e = mdance_result_error(r);
    return e ? e : "";
}

struct Opts {
    mdance_options_t* o = mdance_options_new();
    ~Opts() { mdance_options_free(o); }
    Opts& set(const char* k, const char* v) { EXPECT_EQ(mdance_options_set(o, k, v), 0); return *this; }
    Opts& ints(const char* k, const std::vector<int>& v) {
        EXPECT_EQ(mdance_options_set_ints(o, k, v.data(), (int)v.size()), 0);
        return *this;
    }
};

struct Data {
    mdance_dataset_t* d;
    explicit Data(const std::vector<double>& v, int nframes, int natoms)
        : d(mdance_dataset_from_rows(v.data(), nframes, (int)v.size() / nframes, natoms)) {}
    ~Data() { mdance_dataset_free(d); }
};

}  // namespace

TEST(CApi, Version) {
    EXPECT_EQ(mdance_api_version(), MDANCE_API_VERSION);
    EXPECT_EQ(std::string(mdance_version()).substr(0, 2), "2.");
}

TEST(CApi, FreeingNullIsSafe) {
    mdance_result_free(nullptr);
    mdance_analysis_free(nullptr);
    mdance_prime_free(nullptr);
    mdance_select_free(nullptr);
    mdance_dataset_free(nullptr);
    mdance_options_free(nullptr);
    mdance_control_free(nullptr);
    // a NULL object reads as failed, never as a success
    EXPECT_NE(mdance_result_error(nullptr), nullptr);
}

// ---------------------------------------------------------------- datasets

TEST(Dataset, RefusesBadShapes) {
    EXPECT_EQ(mdance_dataset_new(0, 3), nullptr);
    EXPECT_NE(std::string(mdance_last_error()).find("nframes"), std::string::npos);
    EXPECT_EQ(mdance_dataset_new(3, 0), nullptr);
    EXPECT_EQ(mdance_dataset_new_cols(3, 6, 0), nullptr);
}

TEST(Dataset, RefusesValuesThatAreNotFinite) {
    mdance_dataset_t* d = mdance_dataset_new(2, 1);
    double ok[3] = {1, 2, 3};
    double bad[3] = {1, std::numeric_limits<double>::quiet_NaN(), 3};
    EXPECT_EQ(mdance_dataset_set_frame(d, 0, ok, 3), 0);
    EXPECT_EQ(mdance_dataset_set_frame(d, 1, bad, 3), -1);
    EXPECT_NE(std::string(mdance_last_error()).find("frame 1, column 1"), std::string::npos);
    EXPECT_EQ(mdance_dataset_set_frame(d, 2, ok, 3), -1);    // no frame 2
    EXPECT_EQ(mdance_dataset_set_frame(d, 1, ok, 2), -1);    // wrong length
    EXPECT_EQ(mdance_dataset_nset(d), 1);
    mdance_dataset_free(d);

    std::vector<double> v = blobs(5);
    v[7] = std::numeric_limits<double>::infinity();
    mdance_result_t* r = mdance_kmeans(v.data(), 10, 9, 3, 2, "MSD", "StratAll", 50);
    EXPECT_NE(errOf(r).find("not a finite number"), std::string::npos);
    mdance_result_free(r);
}

TEST(Dataset, UnsetFramesAreRefused) {
    mdance_dataset_t* d = mdance_dataset_new(3, 1);
    double v[3] = {1, 2, 3};
    mdance_dataset_set_frame(d, 0, v, 3);
    Opts o;
    o.set("k", "1");
    mdance_result_t* r = mdance_cluster(d, "kmeans", o.o, nullptr);
    EXPECT_NE(errOf(r).find("frame 1 of the dataset was never set"), std::string::npos);
    mdance_result_free(r);
    mdance_dataset_free(d);
}

TEST(Dataset, FramesFromAProgramsFloatArrays) {
    // VMD's layout: every atom's x y z; the dataset takes atoms 2 and 0
    float pos[9] = {0, 1, 2, 10, 11, 12, 20, 21, 22};
    int atoms[2] = {2, 0};
    mdance_dataset_t* d = mdance_dataset_new(1, 2);
    ASSERT_EQ(mdance_dataset_set_frame_atoms_float(d, 0, pos, 3, atoms, 2), 0);
    double out[6];
    ASSERT_EQ(mdance_dataset_get_frame(d, 0, out, 6), 0);
    EXPECT_EQ(out[0], 20);
    EXPECT_EQ(out[2], 22);
    EXPECT_EQ(out[3], 0);
    int bad[2] = {2, 3};
    EXPECT_EQ(mdance_dataset_set_frame_atoms_float(d, 0, pos, 3, bad, 2), -1);
    mdance_dataset_free(d);
}

TEST(Dataset, SuperposeUndoesARigidMotion) {
    // 5 atoms, frame 1 = frame 0 turned by 90 degrees about z and moved
    std::vector<double> f0 = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 1, 1, 1};
    std::vector<double> rows = f0;
    for (int a = 0; a < 5; ++a) {
        double x = f0[3 * a], y = f0[3 * a + 1], z = f0[3 * a + 2];
        rows.push_back(-y + 4);
        rows.push_back(x - 1);
        rows.push_back(z + 7);
    }
    Data d(rows, 2, 5);
    ASSERT_NE(d.d, nullptr);
    ASSERT_EQ(mdance_dataset_superpose(d.d, nullptr, 0, 0), 0);
    double out[15];
    mdance_dataset_get_frame(d.d, 1, out, 15);
    for (int j = 0; j < 15; ++j) EXPECT_NEAR(out[j], f0[j], 1e-12);
    int fit[2] = {0, 9};
    EXPECT_EQ(mdance_dataset_superpose(d.d, fit, 2, 0), -1);
    EXPECT_EQ(mdance_dataset_superpose(d.d, nullptr, 0, 2), -1);
}

// a little-endian CHARMM DCD with a unit cell, x = 100 f + 10 a + c
static void writeDcd(const char* path, int nf, int na, int cut = 0, bool swap = false) {
    std::vector<unsigned char> b;
    auto i32 = [&](int32_t v) {
        uint32_t u = (uint32_t)v;
        if (swap) u = (u >> 24) | ((u >> 8) & 0xff00) | ((u << 8) & 0xff0000) | (u << 24);
        unsigned char c[4];
        std::memcpy(c, &u, 4);
        b.insert(b.end(), c, c + 4);
    };
    auto f32 = [&](float v) { int32_t i; std::memcpy(&i, &v, 4); i32(i); };
    i32(84);
    b.insert(b.end(), {'C', 'O', 'R', 'D'});
    int icntrl[20] = {nf, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 24};
    for (int v : icntrl) i32(v);
    i32(84);
    i32(84); i32(1); for (int i = 0; i < 80; ++i) b.push_back(' '); i32(84);
    i32(4); i32(na); i32(4);
    for (int f = 0; f < nf; ++f) {
        i32(48); for (int i = 0; i < 12; ++i) i32(0); i32(48);
        for (int c = 0; c < 3; ++c) {
            i32(4 * na);
            for (int a = 0; a < na; ++a) f32((float)(100 * f + 10 * a + c));
            i32(4 * na);
        }
    }
    b.resize(b.size() - cut);
    FILE* fh = std::fopen(path, "wb");
    std::fwrite(b.data(), 1, b.size(), fh);
    std::fclose(fh);
}

TEST(Dataset, ReadsDcd) {
    std::string p = testing::TempDir() + "mdance_capi_test.dcd";
    for (bool swap : {false, true}) {
        writeDcd(p.c_str(), 5, 4, 0, swap);
        int atoms[2] = {3, 1};
        mdance_dataset_t* d = mdance_dataset_read_dcd(p.c_str(), atoms, 2, 1, -1, 2);
        ASSERT_NE(d, nullptr) << mdance_last_error();
        EXPECT_EQ(mdance_dataset_nframes(d), 2);    // frames 1, 3
        double out[6];
        mdance_dataset_get_frame(d, 1, out, 6);
        EXPECT_EQ(out[0], 330);
        EXPECT_EQ(out[2], 332);
        EXPECT_EQ(out[3], 310);
        mdance_dataset_free(d);
    }
    writeDcd(p.c_str(), 5, 4, 3);
    EXPECT_EQ(mdance_dataset_read_dcd(p.c_str(), nullptr, 0, 0, -1, 1), nullptr);
    EXPECT_NE(std::string(mdance_last_error()).find("cut short"), std::string::npos);
    writeDcd(p.c_str(), 5, 4);
    EXPECT_EQ(mdance_dataset_read_dcd(p.c_str(), nullptr, 0, 0, 5, 1), nullptr);     // no frame 5
    int far[1] = {4};
    EXPECT_EQ(mdance_dataset_read_dcd(p.c_str(), far, 1, 0, -1, 1), nullptr);       // no atom 4
    EXPECT_EQ(mdance_dataset_read_dcd(p.c_str(), nullptr, 0, 0, -1, 0), nullptr);   // stride 0
    std::remove(p.c_str());
    EXPECT_EQ(mdance_dataset_read_dcd(p.c_str(), nullptr, 0, 0, -1, 1), nullptr);
}

// ---------------------------------------------------------------- options

TEST(Options, EachIsChecked) {
    std::vector<double> v = blobs(10);
    Data d(v, 20, 3);
    auto run = [&](Opts& o) {
        mdance_result_t* r = mdance_cluster(d.d, "kmeans", o.o, nullptr);
        std::string e = errOf(r);
        mdance_result_free(r);
        return e;
    };
    { Opts o; EXPECT_NE(run(o).find("k is required"), std::string::npos); }
    { Opts o; o.set("k", "2").set("kinti", "StratAll"); EXPECT_NE(run(o).find("unknown option kinti"), std::string::npos); }
    { Opts o; o.set("k", "2x"); EXPECT_NE(run(o).find("not a whole number"), std::string::npos); }
    { Opts o; o.set("k", "21"); EXPECT_NE(run(o).find("outside 1 .. 20"), std::string::npos); }
    { Opts o; o.set("k", "2").set("percentage", "101"); EXPECT_NE(run(o).find("outside 0 .. 100"), std::string::npos); }
    { Opts o; o.set("k", "2").set("metric", "msd"); EXPECT_NE(run(o).find("not one of"), std::string::npos); }
    { Opts o; o.set("k", "2").set("percentage", "50"); EXPECT_EQ(run(o), ""); }
    // "010" is ten, as a person means it (never octal)
    { Opts o; o.set("k", "02").set("percentage", "050"); EXPECT_EQ(run(o), ""); }
    { Opts o; o.set("k", "2"); mdance_result_t* r = mdance_cluster(d.d, "kmean", o.o, nullptr);
      EXPECT_NE(errOf(r).find("unknown algorithm"), std::string::npos); mdance_result_free(r); }
    // numbers: no hex, no inf, no nan
    EXPECT_NE(mdance_options_set_double(nullptr, "x", 1), 0);
    { Opts o; o.set("threshold", "0x1F"); mdance_result_t* r = mdance_cluster(d.d, "equal", o.o, nullptr);
      EXPECT_NE(errOf(r).find("not a finite number"), std::string::npos); mdance_result_free(r); }
    { Opts o; o.set("threshold", "inf"); mdance_result_t* r = mdance_cluster(d.d, "equal", o.o, nullptr);
      EXPECT_NE(errOf(r).find("not a finite number"), std::string::npos); mdance_result_free(r); }
}

// ---------------------------------------------------------------- the two interfaces agree

TEST(CApi, BothInterfacesGiveTheSameAnswers) {
    std::vector<double> v = blobs(30);
    Data d(v, 60, 3);
    mdance_result_t* a = mdance_kmeans(v.data(), 60, 9, 3, 2, "MSD", "StratAll", 20);
    Opts o;
    o.set("k", "2").set("kinit", "StratAll").set("percentage", "20");
    mdance_result_t* b = mdance_cluster(d.d, "kmeans", o.o, nullptr);
    ASSERT_EQ(errOf(a), "");
    ASSERT_EQ(errOf(b), "");
    ASSERT_EQ(mdance_result_nclusters(a), 2);
    for (int i = 0; i < 60; ++i) EXPECT_EQ(mdance_result_labels(a)[i], mdance_result_labels(b)[i]);
    // and they are the two blobs
    for (int i = 1; i < 30; ++i) EXPECT_EQ(mdance_result_labels(a)[i], mdance_result_labels(a)[0]);
    for (int i = 31; i < 60; ++i) EXPECT_EQ(mdance_result_labels(a)[i], mdance_result_labels(a)[30]);
    EXPECT_NE(mdance_result_labels(a)[0], mdance_result_labels(a)[30]);
    EXPECT_EQ(mdance_result_cluster_sizes(a)[0], 30);
    EXPECT_DOUBLE_EQ(mdance_result_ch_score(a), mdance_result_ch_score(b));
    mdance_result_free(a);
    mdance_result_free(b);
}

TEST(CApi, KmeansSeedMakesRandomSeedingRepeatable) {
    std::vector<double> v = blobs(20);
    Data d(v, 40, 3);
    Opts o;
    o.set("k", "4").set("kinit", "Random").set("seed", "42");
    mdance_result_t* a = mdance_cluster(d.d, "kmeans", o.o, nullptr);
    mdance_result_t* b = mdance_cluster(d.d, "kmeans", o.o, nullptr);
    ASSERT_EQ(errOf(a), "");
    EXPECT_EQ(mdance_result_seed(a), 42);
    for (int i = 0; i < 40; ++i) EXPECT_EQ(mdance_result_labels(a)[i], mdance_result_labels(b)[i]);
    mdance_result_free(a);
    mdance_result_free(b);
    // a seed from the clock is reported, and gives the same again
    Opts c;
    c.set("k", "4").set("kinit", "KmeansPP");
    mdance_result_t* r = mdance_cluster(d.d, "kmeans", c.o, nullptr);
    ASSERT_EQ(errOf(r), "");
    int seed = mdance_result_seed(r);
    ASSERT_GE(seed, 0);
    c.set("seed", std::to_string(seed).c_str());
    mdance_result_t* r2 = mdance_cluster(d.d, "kmeans", c.o, nullptr);
    ASSERT_EQ(errOf(r2), "");
    for (int i = 0; i < 40; ++i) EXPECT_EQ(mdance_result_labels(r)[i], mdance_result_labels(r2)[i]);
    mdance_result_free(r);
    mdance_result_free(r2);
}

// ---------------------------------------------------------------- inputs that crashed

TEST(Crashes, CompSimSeedingFromTooFewFrames) {
    // 5 frames at 10%: the seeds were drawn from 0 frames (a segfault)
    std::vector<double> v = blobs(3);
    mdance_result_t* r = mdance_kmeans(v.data(), 5, 9, 3, 2, "MSD", "CompSim", 10);
    EXPECT_NE(errOf(r).find("from 0 frames"), std::string::npos);
    mdance_result_free(r);
}

TEST(Crashes, PercentageOver100) {
    std::vector<double> v = blobs(25);
    mdance_result_t* r = mdance_kmeans(v.data(), 50, 9, 3, 2, "MSD", "CompSim", 200);
    EXPECT_NE(errOf(r).find("percentage"), std::string::npos);
    mdance_result_free(r);
}

TEST(Crashes, DivineRefineOnIdenticalFrames) {
    std::vector<double> v(20 * 9, 1.5);
    mdance_result_t* r = mdance_divine(v.data(), 20, 9, 3, 3, "MSD", "WeightedMSD", "OutlierPair",
                                       "StratAll", 1, 0.0, "k", 10);
    if (mdance_has_divine()) {
        EXPECT_EQ(errOf(r), "");
        EXPECT_GE(mdance_result_nclusters(r), 1);
    }
    mdance_result_free(r);
}

TEST(Crashes, PrimeAndAnalysisWithHugeLabels) {
    std::vector<double> v = blobs(5);
    std::vector<int> lab(10, 0);
    lab[3] = INT_MAX;
    mdance_prime_t* p = mdance_prime(v.data(), 10, 9, 3, "RR", lab.data(), 10, 0, 0);
    EXPECT_NE(std::string(mdance_prime_error(p) ? mdance_prime_error(p) : "").find("outside"), std::string::npos);
    mdance_prime_free(p);
    lab[3] = 2000000000;
    mdance_analysis_t* a = mdance_analysis(v.data(), 10, 9, 3, "MSD", lab.data(), 10);
    EXPECT_NE(mdance_analysis_error(a), nullptr);
    mdance_analysis_free(a);
}

TEST(Crashes, HelmNegativeTrimK) {
    std::vector<double> v = blobs(10);
    std::vector<int> init(20);
    for (int i = 0; i < 20; ++i) init[i] = i / 4;
    mdance_result_t* r = mdance_helm(v.data(), 20, 9, 3, 2, init.data(), 20, "MSD", "Inter", -1, 1, 0.01f, 0, -1);
    EXPECT_NE(errOf(r), "");
    mdance_result_free(r);
}

TEST(Helm, FewerStartingClustersThanKAreLeftAsTheyAre) {
    std::vector<double> v = blobs(10);
    std::vector<int> init(20);
    for (int i = 0; i < 20; ++i) init[i] = i < 10 ? 0 : 1;
    // k 5 with 2 starting clusters merged them all into one
    mdance_result_t* r = mdance_helm(v.data(), 20, 9, 3, 5, init.data(), 20, "MSD", "Inter", -1, 0, 0.01f, 0, 0);
    ASSERT_EQ(errOf(r), "");
    EXPECT_EQ(mdance_result_nclusters(r), 2);
    mdance_result_free(r);
}

TEST(Select, DiversityPicksRealFrames) {
    // RR on coordinates: every candidate's value was below the start (-1),
    // and the "frame" picked was the number of columns, over and over
    std::vector<double> v = blobs(20);
    mdance_select_t* s = mdance_select(v.data(), 40, 9, 3, "RR", "diversity", 50, 10);
    ASSERT_EQ(mdance_select_error(s), nullptr);
    std::vector<int> idx(mdance_select_indices(s), mdance_select_indices(s) + mdance_select_count(s));
    EXPECT_EQ(idx.size(), 20u);
    for (int i : idx) { EXPECT_GE(i, 0); EXPECT_LT(i, 40); }
    std::sort(idx.begin(), idx.end());
    EXPECT_EQ(std::unique(idx.begin(), idx.end()), idx.end());
    mdance_select_free(s);
}

TEST(Msd, NoIntOverflowForManyFramesOfManyAtoms) {
    // 10 000 frames of 30 atoms: N * N * nAtoms overflowed an int
    const int nf = 10000, na = 30;
    std::vector<double> v((size_t)nf * 3 * na);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (double)((i * 7919) % 1000) / 1000.0;
    mdance_analysis_t* a = mdance_analysis(v.data(), nf, 3 * na, na, "MSD", nullptr, 0);
    ASSERT_EQ(mdance_analysis_error(a), nullptr);
    EXPECT_GT(mdance_analysis_isim(a), 0);     // was negative
    EXPECT_LT(mdance_analysis_isim(a), 1e3);
    mdance_analysis_free(a);
}

// ---------------------------------------------------------------- controls

TEST(Control, CancelsALongCallFromAnotherThread) {
    // eQUAL with comp_sim seeds on 3000 frames: tens of seconds
    const int nf = 3000;
    std::vector<double> v((size_t)nf * 30);
    for (int f = 0; f < nf; ++f)
        for (int c = 0; c < 30; ++c) v[(size_t)f * 30 + c] = std::sin(f * 0.37 + c) * 3 + (f % 11);
    Data d(v, nf, 10);
    Opts o;
    o.set("threshold", "0.01").set("seed_method", "comp_sim");
    mdance_control_t* ctl = mdance_control_new();
    std::thread t([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        mdance_control_cancel(ctl);
    });
    auto t0 = std::chrono::steady_clock::now();
    mdance_result_t* r = mdance_cluster(d.d, "equal", o.o, ctl);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t.join();
    EXPECT_EQ(errOf(r), "cancelled");
    EXPECT_LT(s, 5.0);
    EXPECT_GT(mdance_control_progress(ctl), 0.0);
    EXPECT_STREQ(mdance_control_stage(ctl), "eQUAL");
    mdance_result_free(r);
    mdance_control_reset(ctl);
    EXPECT_EQ(mdance_control_cancelled(ctl), 0);
    mdance_control_free(ctl);
}

TEST(Control, ConcurrentCallsAgree) {
    // the generator was global: concurrent k-means raced on it
    std::vector<double> v = blobs(50);
    Data d(v, 100, 3);
    Opts o;
    o.set("k", "5").set("kinit", "Random").set("seed", "7");
    mdance_result_t* ref = mdance_cluster(d.d, "kmeans", o.o, nullptr);
    std::vector<std::thread> ts;
    std::atomic<int> same{0};
    for (int i = 0; i < 4; ++i) {
        ts.emplace_back([&] {
            for (int k = 0; k < 10; ++k) {
                mdance_result_t* r = mdance_cluster(d.d, "kmeans", o.o, nullptr);
                if (std::memcmp(mdance_result_labels(r), mdance_result_labels(ref), 100 * sizeof(int)) == 0) same++;
                mdance_result_free(r);
            }
        });
    }
    for (auto& t : ts) t.join();
    EXPECT_EQ(same.load(), 40);
    mdance_result_free(ref);
}
