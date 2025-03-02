#pragma once
#include <string>
#include <iostream>
using std::string, std::cerr;

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
/*
 * Indices
 * AC: Austin-Colwell, BUB: Baroni-Urbani-Buser, CTn: Consoni-Todschini n
 * Fai: Faith, Gle: Gleason, Ja: Jaccard, Ja0: Jaccard 0-variant
 * JT: Jaccard-Tanimoto, RT: Rogers-Tanimoto, RR: Russel-Rao
 * SM: Sokal-Michener, SSn: Sokal-Sneath n
*/
struct indices{
    float ac;
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
};
struct counters{
    float a;
    float d;
    float totalSim;
    float totalDis;
    float p;
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
