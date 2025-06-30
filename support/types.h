#pragma once
#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
using std::string, std::set, std::vector, std::cerr, Eigen::MatrixXd, Eigen::VectorXd, Eigen::RowVectorXd, Eigen::VectorXi, Eigen::Index;
#define sortvd vector<std::pair<double, int>>
enum DataType {full, condensed};

/*
 * Valid values for metric are:
 *
 *  ``MSD``: Mean Square Deviation.
 *  
 *  Extended or Instant Similarity Metrics : 
 *  
 *  | ``AC``: Austin-Colwell, ``BUB``: Baroni-Urbani-Buser, 
 *  | ``CTn``: Consoni-Todschini n, ``Fai``: Faith, 
 *  | ``Gle``: Gleason, ``Ja``: Jaccard, 
 *  | ``Ja0``: Jaccard 0-variant, ``JT``: Jaccard-Tanimoto, 
 *  | ``RT``: Rogers-Tanimoto, ``RR``: Russel-Rao,
 *  | ``SM``: Sokal-Michener, ``SSn``: Sokal-Sneath n.
*/
enum Metric {MSD, AC, CTn, GLe, Ja0, RT, SMm};
enum SimIndex {RR, JT, SMs};
enum TrimCriteria{compSimilarity, simToMedoid};
enum InitiateDiversity{strat, medoid, outlier, randomly};
enum InitiateKTypes{stratAll, stratReduced, compSim, divSelect, kMeanspp, random, vanillakpp};
enum class ThresholdType{None, Dissimilar, Percent, Integer};

struct Threshold {
    ThresholdType type;
    int value;

    static Threshold None(int nObjects) {return {ThresholdType::None, nObjects % 2}; }
    static Threshold Dissimilar(int nObjects) { return {ThresholdType::Dissimilar, ceil(nObjects * 0.5)}; }
    static Threshold Percentage(float p, int nObjects) { return {ThresholdType::Percent, round(nObjects * p)}; }
    static Threshold Integer(int i) { return {ThresholdType::Integer, i}; }
};

/*
 * Indices
 * AC: Austin-Colwell, BUB: Baroni-Urbani-Buser, CTn: Consoni-Todschini n
 * Fai: Faith, Gle: Gleason, Ja: Jaccard, Ja0: Jaccard 0-variant
 * JT: Jaccard-Tanimoto, RT: Rogers-Tanimoto, RR: Russel-Rao
 * SM: Sokal-Michener, SSn: Sokal-Sneath n
*/
class Indices{
    double bub;
    double fai;
    double gle;
    double ja;
    double jt;
    double rt;
    double rr;
    double sm;
    double ss1;
    double ss2;

public: 
    Indices(double bub, double fai, double gle, double ja, double jt, double rt, double rr, double sm, double ss1, double ss2) :
        bub(bub), fai(fai), gle(gle), ja(ja), jt(jt), rt(rt), rr(rr), sm(sm), ss1(ss1), ss2(ss2) {}; 
    double getIndex(Metric mt){
        switch(mt) {
            case GLe: return gle;
            case RT: return rt;
            case SMm: return sm;
        }
        return 0;
    }
};
struct Counters{
    int a;
    double wa;
    int d;
    double wd;
    int totalSim;
    double totalWsim;
    int totalDis;
    double totalWdis;
    int p;
    double wp;

    Counters(int a, double wa, int d, double wd, int totalDis, double totalWdis): a(a), wa(wa), d(d), wd(wd), totalSim(a+d), totalWsim(wa+wd), totalDis(totalDis), totalWdis(totalWdis), p(a+d+totalDis), wp(wa+wd+totalWdis) {};
};

inline DataType parseDataType(string dt) {
    DataType out;
    if (dt.compare("full"))
        out=full;
    else if (dt.compare("condensed"))
        out=condensed;
    else
        cerr << "That is not a valid DataType!\n";
    return out;
}
inline Metric parseMetric(string mt) {
    Metric out;
    if (mt.compare("MSD"))
        out=MSD;
    else if (mt.compare("AC"))
        out=AC;
    else if (mt.compare("CTn"))
        out=CTn;
    else if (mt.compare("GLe"))
        out=GLe;
    else if (mt.compare("Ja0"))
        out=Ja0;
    else if (mt.compare("RT"))
        out=RT;
    else if (mt.compare("SM"))
        out=Metric::SMm;
    else
        cerr << "That is not a valid DataType!\n";
    return out;
}
inline SimIndex parseSimIndex(string si) {
    SimIndex out;
    if (si.compare("RR"))
        out=RR;
    else if (si.compare("JT"))
        out=JT;
    else if (si.compare("SM"))
        out=SMs;
    else
        cerr << "That is not a valid SimIndex!\n";
    return out;
}
