#pragma once
#include "types.h"
#include <math.h>

MD::Indices genSimIdx(ArrayXd& cTotal, int nObjects, MD::Threshold& cThreshold, int wt);
MD::Counters calculateCounters(ArrayXd& cTotal, int nObjects, MD::Threshold& cThreshold, int wFactor = INT_MIN);