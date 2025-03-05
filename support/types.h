#pragma once
#include <string>
#include <iostream>
#include <Eigen/Dense>
using std::string, std::cerr, Eigen::MatrixXf, Eigen::VectorXf, Eigen::VectorXi, Eigen::Index;

enum Weight {fraction, powerN};
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
enum Metric {MSD, AC, CTn, GLe, Ja0, RT, SM};
enum SimIndex {RR, JT, SM};
enum ThresholdType{None, Dissimilar};
enum TrimCriteria{compSim, simToMedoid};
enum InitiateDiversity{medoid, outlier, random};

/*
 * Indices
 * AC: Austin-Colwell, BUB: Baroni-Urbani-Buser, CTn: Consoni-Todschini n
 * Fai: Faith, Gle: Gleason, Ja: Jaccard, Ja0: Jaccard 0-variant
 * JT: Jaccard-Tanimoto, RT: Rogers-Tanimoto, RR: Russel-Rao
 * SM: Sokal-Michener, SSn: Sokal-Sneath n
*/
class Indices{
    float bub;
    float fai;
    float gle;
    float ja;
    float jt;
    float rt;
    float rr;
    float sm;
    float ss1;
    float ss2;

public: 
    Indices(float bub, float fai, float gle, float ja, float jt, float rt, float rr, float sm, float ss1, float ss2) :
        bub(bub), fai(fai), gle(gle), ja(ja), jt(jt), rt(rt), rr(rr), sm(sm), ss1(ss1), ss2(ss2) {}; 
    float getIndex(Metric mt){
        switch(mt) {
            case GLe: return gle;
            case RT: return rt;
            case SM: return sm;
        }
    }
};
struct Counters{
    float a;
    float wa;
    float d;
    float wd;
    float totalSim;
    float totalWsim;
    float totalDis;
    float totalWdis;
    float p;
    float wp;
};
struct Threshold{
    ThresholdType type;
    int c;

    Threshold(ThresholdType type = None, int c = 0): type(type), c(c) {};
};

Weight parseWeight(string wt) {
    Weight out;
    if (wt.compare("fraction"))
        out=fraction;
    else if (wt.compare("powerN"))
        out=powerN;
    else
        cerr << "That is not a valid Weight!\n";
    return out;
}
DataType parseDataType(string dt) {
    DataType out;
    if (dt.compare("full"))
        out=full;
    else if (dt.compare("condensed"))
        out=condensed;
    else
        cerr << "That is not a valid DataType!\n";
    return out;
}
Metric parseMetric(string mt) {
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
        out=Metric::SM;
    else
        cerr << "That is not a valid DataType!\n";
    return out;
}
SimIndex parseSimIndex(string si) {
    SimIndex out;
    if (si.compare("RR"))
        out=RR;
    else if (si.compare("JT"))
        out=JT;
    else if (si.compare("SM"))
        out=SimIndex::SM;
    else
        cerr << "That is not a valid SimIndex!\n";
    return out;
}
