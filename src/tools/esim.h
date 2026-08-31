#pragma once

#include <math.h>

#include "types.h"

MD::Indices genSimIdx(const ArrayXd& cTotal, int nObjects, MD::Threshold& cThreshold, int wt);
// wFactor = 0 selects the "fraction" weight function (the reference default);
// any other value n selects the power_n weights.
MD::Counters calculateCounters(const ArrayXd& cTotal, int nObjects, MD::Threshold& cThreshold, int wFactor = 0);
