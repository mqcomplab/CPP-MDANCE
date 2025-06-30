#pragma once
#include "../support/types.h"
// isim is not used in the MDANCE code so I am not converting it

Indices genSimIdx(VectorXd& cTotal, int nObjects, Threshold cThreshold, int wt);
Counters calculateCounters(VectorXd& cTotal, int nObjects, Threshold cThreshold, int wFactor = INT_MIN);