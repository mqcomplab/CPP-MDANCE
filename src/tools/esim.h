#pragma once

#include <math.h>

#include "types.h"

MD::Indices genSimIdx(const ArrayXd& cTotal, int nObjects, MD::Threshold& cThreshold, int wt);
MD::Counters calculateCounters(const ArrayXd& cTotal, int nObjects, MD::Threshold& cThreshold, int wFactor = INT_MIN);
