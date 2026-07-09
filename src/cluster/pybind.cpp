#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include "pybind.h"
namespace py=pybind11;

Veci runKmeans(Mat X, Mat initiators){
    //py::gil_scoped_acquire acquire;
    Eigen::MatrixXd M=X.matrix();
    Eigen::MatrixXd initMat=initiators.matrix();

    py::module sklearnCluster=py::module::import("sklearn.cluster");
    py::object Kmeans=sklearnCluster.attr("KMeans");
    py::object model=Kmeans(py::arg("n_clusters")=2,
                            py::arg("init")=initMat,
                            py::arg("n_init")=1,
                            py::arg("max_iter")=300,
                            py::arg("algorithm")="elkan");
    py::object labelsPy=model.attr("fit_predict")(M);
    Veci labels=labelsPy.cast<Veci>();
    return labels;
}