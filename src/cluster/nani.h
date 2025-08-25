#include "../tools/types.h"

vector<Index> initiateKmeans(ArrayXXd data, int nClusters, MD::Metric mt, int nAtoms = 1, MD::KinitType kinit = MD::KinitType::CompSim, int percentage = 10); // TODO: maybe have default Metric=MSD?
