#pragma once

#include <map>
#include <string>
#include <stdexcept>

#include "types.h"

inline MD::Metric parseMetric(const std::string& s) {
    static const std::map<std::string, MD::Metric> m = {
        {"MSD", MD::Metric::MSD}, {"BUB", MD::Metric::BUB}, {"Fai", MD::Metric::Fai},
        {"Gle", MD::Metric::Gle}, {"Ja", MD::Metric::Ja}, {"JT", MD::Metric::JT},
        {"RT", MD::Metric::RT}, {"RR", MD::Metric::RR}, {"SM", MD::Metric::SM},
        {"SS1", MD::Metric::SS1}, {"SS2", MD::Metric::SS2}
    };
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("Unknown metric: " + s);
    return it->second;
}

inline MD::KinitType parseKinit(const std::string& s) {
    static const std::map<std::string, MD::KinitType> m = {
        {"StratAll", MD::KinitType::StratAll}, {"StratReduced", MD::KinitType::StratReduced},
        {"CompSim", MD::KinitType::CompSim}, {"DivSelect", MD::KinitType::DivSelect},
        {"KmeansPP", MD::KinitType::KmeansPP}, {"Random", MD::KinitType::Random},
        {"VanillaKmeansPP", MD::KinitType::VanillaKmeansPP}
    };
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("Unknown kinit: " + s);
    return it->second;
}

inline MD::DivineSplit parseSplit(const std::string& s) {
    static const std::map<std::string, MD::DivineSplit> m = {
        {"MSD", MD::DivineSplit::MSD}, {"Radius", MD::DivineSplit::Radius},
        {"WeightedMSD", MD::DivineSplit::WeightedMSD}
    };
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("Unknown split type: " + s);
    return it->second;
}

inline MD::DivineAnchors parseAnchors(const std::string& s) {
    static const std::map<std::string, MD::DivineAnchors> m = {
        {"NANI", MD::DivineAnchors::NANI}, {"OutlierPair", MD::DivineAnchors::OutlierPair},
        {"SplinterPair", MD::DivineAnchors::SplinterPair}
    };
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("Unknown anchor type: " + s);
    return it->second;
}

inline MD::MergeScheme parseMergeScheme(const std::string& s) {
    static const std::map<std::string, MD::MergeScheme> m = {
        {"Intra", MD::MergeScheme::Intra}, {"Inter", MD::MergeScheme::Inter},
        {"Half", MD::MergeScheme::Half}
    };
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("Unknown merge scheme: " + s);
    return it->second;
}

inline MD::EqualSeed parseEqualSeed(const std::string& s) {
    if (s == "comp_sim" || s == "CompSim") return MD::EqualSeed::CompSim;
    if (s == "medoid"   || s == "Medoid")  return MD::EqualSeed::Medoid;
    if (s == "greedy" || s == "vanilla" || s == "mini_batch_kmeans") {
        throw std::runtime_error("eQUAL seed method '" + s +
            "' is sklearn-based and not supported in this build; use 'comp_sim' or 'medoid'.");
    }
    throw std::runtime_error("Unknown eQUAL seed method: " + s);
}

inline MD::AlignMethod parseAlign(const std::string& s) {
    if (s == "none" || s == "None" || s.empty()) return MD::AlignMethod::None;
    if (s == "uni"  || s == "Uni")  return MD::AlignMethod::Uni;
    if (s == "kron" || s == "Kron") return MD::AlignMethod::Kron;
    throw std::runtime_error("Unknown alignment method: " + s);
}
