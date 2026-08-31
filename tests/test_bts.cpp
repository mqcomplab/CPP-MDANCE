#include <gtest/gtest.h>

#include "../src/tools/bts.h"
#include "../src/cluster/KMeansRex/KMeans.h"

namespace {

// 12 points, 2 features: three loose groups and one far outlier (row 10).
// The expected values in these tests were computed with a NumPy
// transcription of the MDANCE Python reference (bts.py @ 016bd9a).
ArrayXXd makeToyData() {
    ArrayXXd X(12, 2);
    X << 1.0, 2.0,
         2.0, 2.0,
         2.0, 3.0,
         1.5, 2.5,
         8.0, 7.0,
         8.0, 8.0,
         7.5, 7.5,
         8.5, 7.2,
         4.0, 5.0,
         4.5, 4.5,
         25.0, 80.0,
         5.0, 5.0;
    return X;
}

void expectKeptRows(const ArrayXXd& trimmed, const ArrayXXd& X, const std::vector<int>& kept) {
    ASSERT_EQ(trimmed.rows(), (Index)kept.size());
    for (size_t i = 0; i < kept.size(); ++i) {
        EXPECT_TRUE((trimmed.row(i) == X.row(kept[i])).all()) << "kept row " << i;
    }
}

}  // namespace

TEST(BtsTools, MedoidAndOutlier) {
    ArrayXXd X = makeToyData();
    EXPECT_EQ(calculateMedoid(X), 5);
    EXPECT_EQ(calculateOutlier(X), 10);
}

TEST(BtsTools, TrimOutliersCompSim) {
    ArrayXXd X = makeToyData();
    // reference trims rows 10 (outlier), 0 and 1: the lowest comp_sim values
    expectKeptRows(trimOutliers(X, 3, 1, false), X, {2, 3, 4, 5, 6, 7, 8, 9, 11});
}

TEST(BtsTools, TrimOutliersSimToMedoid) {
    ArrayXXd X = makeToyData();
    // the medoid is row 5; the reference trims the three frames FARTHEST from
    // it (rows 10, 0, 3). The inverted criterion used to trim the closest.
    expectKeptRows(trimOutliers(X, 3, 1, true), X, {1, 2, 4, 5, 6, 7, 8, 9, 11});
}

TEST(BtsTools, TrimOutliersFraction) {
    // docstring example from the Python reference
    ArrayXXd X(6, 2);
    X << 1, 2,  2, 2,  2, 3,  8, 7,  8, 8,  25, 80;
    expectKeptRows(trimOutliers(X, 0.6f, 1, false), X, {2, 3, 4});
}

TEST(BtsTools, TrimOutliersEdgeCases) {
    ArrayXXd X = makeToyData();
    EXPECT_EQ(trimOutliers(X, 0, 1, false).rows(), X.rows());
    EXPECT_EQ(trimOutliers(X, (int)X.rows(), 1, false).rows(), 0);
    EXPECT_EQ(trimOutliers(X, 100, 1, true).rows(), 0);
}

TEST(BtsTools, RepSampleMatchesReference) {
    ArrayXXd X = makeToyData();
    std::vector<int> expected = {10, 0, 1, 3, 2};
    ArrayXi s = repSample(X, MD::Metric::MSD, 1, 3, 5, true);
    ASSERT_EQ(s.size(), (Index)expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(s[i], expected[i]) << "sample " << i;
    }
    // the soft cap gives the same result here (the capping pass adds nothing)
    ArrayXi soft = repSample(X, MD::Metric::MSD, 1, 3, 5, false);
    ASSERT_EQ(soft.size(), (Index)expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(soft[i], expected[i]) << "sample " << i;
    }
}

TEST(BtsTools, RepSampleFraction) {
    ArrayXXd X = makeToyData();
    // 0.5 of 12 rows -> 6 samples
    std::vector<int> expected = {10, 0, 1, 3, 2, 9};
    ArrayXi s = repSample(X, MD::Metric::MSD, 1, 3, 0.5, true);
    ASSERT_EQ(s.size(), (Index)expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(s[i], expected[i]) << "sample " << i;
    }
}

TEST(BtsTools, RepSampleMoreThanAvailable) {
    // must terminate and return every object exactly once (the reference
    // implementation loops forever here)
    ArrayXXd X = makeToyData();
    ArrayXi s = repSample(X, MD::Metric::MSD, 1, 3, 100, true);
    ASSERT_EQ(s.size(), X.rows());
    std::vector<bool> seen(X.rows(), false);
    for (Index i = 0; i < s.size(); ++i) {
        ASSERT_GE(s[i], 0);
        ASSERT_LT(s[i], (int)X.rows());
        EXPECT_FALSE(seen[s[i]]) << "duplicate index " << s[i];
        seen[s[i]] = true;
    }
}

TEST(BtsTools, RepSampleIdenticalObjects) {
    // zero comp_sim range: everything falls into one bin; must not crash
    ArrayXXd X = ArrayXXd::Ones(8, 3);
    ArrayXi s = repSample(X, MD::Metric::MSD, 1, 10, 4, true);
    ASSERT_EQ(s.size(), 4);
    for (Index i = 0; i < s.size(); ++i) {
        ASSERT_GE(s[i], 0);
        ASSERT_LT(s[i], (int)X.rows());
    }
}

TEST(BtsTools, DiversitySelectionRejectsOverPercentage) {
    ArrayXXd X = makeToyData();
    EXPECT_THROW(diversitySelection(X, 200), std::runtime_error);
}

TEST(KMeansNaniInit, KmeansPPProducesRealCenters) {
    // KinitType::KmeansPP used to fall through init_Mu without initializing
    // the centers, silently clustering around the zero matrix.
    ArrayXXd X = makeToyData();
    KmeansNANI km(X, 3, MD::Metric::MSD, MD::KinitType::KmeansPP, 1, 10);
    EXPECT_GT(km.getCenters().abs().sum(), 0.0);
    ArrayXi labels = km.getLabels();
    ASSERT_EQ(labels.size(), X.rows());
    for (Index i = 0; i < labels.size(); ++i) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], 3);
    }
}

TEST(KMeansNaniInit, RejectsInvalidKClusters) {
    ArrayXXd X = makeToyData();
    EXPECT_THROW(KmeansNANI(X, 0, MD::Metric::MSD, MD::KinitType::Random, 1, 10), std::invalid_argument);
    EXPECT_THROW(KmeansNANI(X, 13, MD::Metric::MSD, MD::KinitType::Random, 1, 10), std::invalid_argument);
    Mat badCenters = Mat::Zero(2, 2);
    EXPECT_THROW(KmeansNANI(X, 3, MD::Metric::MSD, badCenters, 1, 10), std::invalid_argument);
}
