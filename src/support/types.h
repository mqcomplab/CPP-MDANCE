#pragma once
#include <Eigen/Dense>
using Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXi;

/*
 * Valid values for metric are:
 *
 *  ``MSD``: Mean Square Deviation.
 *  
 *  Extended or Instant Similarity Metrics : 
 *  
 *  | Not implemented: ``AC``: Austin-Colwell, 
 *  | ``BUB``: Baroni-Urbani-Buser, 
 *  | Not implemented: ``CTn``: Consoni-Todschini n, 
 *  | ``Fai``: Faith, 
 *  | ``Gle``: Gleason, 
 *  | ``Ja``: Jaccard, 
 *  | Not implemented: ``Ja0``: Jaccard 0-variant, 
 *  | ``JT``: Jaccard-Tanimoto, 
 *  | ``RT``: Rogers-Tanimoto, 
 *  | ``RR``: Russel-Rao,
 *  | ``SM``: Sokal-Michener, 
 *  | ``SS1``: Sokal-Sneath 1.
 *  | ``SS2``: Sokal-Sneath 2.
*/
enum class Metric {MSD, BUB, Fai, Gle, Ja, JT, RT, RR, SM, SS1, SS2};

/* Coincidence Threshold
 * type: The way the Threshold is defined.
 * value: The threshold value.
 *
 * Reference: https://github.com/mqcomplab/MDANCE/blob/main/src/mdance/tools/esim.py#L29
*/
enum class ThresholdType{None, Dissimilar, Percent, Integer};
struct Threshold {
    ThresholdType type;
    int value;

    Threshold(ThresholdType t, int nObjects) : type(t) { 
        switch(t) {
            case ThresholdType::None: value = nObjects % 2; break;
            case ThresholdType::Dissimilar: value = (int) ceil(nObjects * 0.5); break;
            case ThresholdType::Integer: value = nObjects; break;
            default: value=0;
        }
    }
    Threshold(float p, int nObjects) : type(ThresholdType::Percent) {
        value = (int) round(nObjects * p);
    }
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
            case Metric::BUB: return bub;
            case Metric::Fai: return fai;
            case Metric::Gle: return gle;
            case Metric::Ja: return ja;
            case Metric::JT: return jt;
            case Metric::RT: return rt;
            case Metric::RR: return rr;
            case Metric::SM: return sm;
            case Metric::SS1: return ss1;
            case Metric::SS2: return ss2;
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
