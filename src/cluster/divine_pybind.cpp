#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "divine.h"

namespace py = pybind11;

PYBIND11_MODULE(cluster, m) {
    m.def("run_kmeans", &run_kmeans);
}