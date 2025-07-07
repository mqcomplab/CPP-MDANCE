#pragma once
#include "../support/types.h"
#include <math.h>

Indices genSimIdx(VectorXd& cTotal, int nObjects, Threshold cThreshold, int wt);
Counters calculateCounters(VectorXd& cTotal, int nObjects, Threshold cThreshold, int wFactor = INT_MIN);