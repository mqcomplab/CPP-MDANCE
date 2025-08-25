#include "../tools/types.h"
#include "../tools/bts.h"
#include "KMeansRex/KMeans.cpp"

class Divine{
    Mat data;
    Veci labels;
    MD::Metric mt;
    MD::DivineSplit splitType;
    MD::DivineAnchors anchorType;
    MD::KinitType kinit;
    int end;
    int kClusters;
    bool refine;
    int nAtoms;
    double threshold;
    int percentage;

    vector<vector<Index>> clusters;

    void divisiveAlgorithm() {
        int minFrames = std::max(1, (int)round(threshold * data.rows()));
        while (true) {
            vector<bool> failedSplits(clusters.size(), false);
            bool didSplit = false;
            while (!didSplit) {
                Index clusterToSplit = selectClusterToSplit(failedSplits);
                if (clusterToSplit < 0) {
                    std::cerr << "No more cluster splits possible that would yield valid subclusters (minFrames = " << minFrames << "). Consider loosening the threshold (currently " << threshold << ") Current number of clusters: " << clusters.size() << std::endl;
                    break;
                }   
                didSplit = splitCluster(clusterToSplit, minFrames);
                failedSplits[clusterToSplit] = !didSplit;
            }
        }
    };
    Index selectClusterToSplit(vector<bool>& failedSplits) {
        Index topCluster = -1;
        double bestScore = -1;

        for (Index i = 0; i < clusters.size(); ++i) {
            if (failedSplits[i] || clusters[i].size() < 2) continue;

            double score = -1;
            Mat subdata = data(clusters[i], Eigen::all);
            if (splitType == MD::DivineSplit::MSD) {
                score = extendedComparison(subdata, 0, nAtoms, false, mt);
            } else if (splitType == MD::DivineSplit::Radius) {
                Index medoidIdx = calculateMedoid(subdata, nAtoms, mt);
                Vec medoid = subdata.row(medoidIdx);
                Vec dists = (subdata.rowwise() - medoid.transpose()).square().rowwise().sum() / nAtoms;
                score = dists.maxCoeff();
            } else if (splitType == MD::DivineSplit::WeightedMSD) {
                score = clusters[i].size() * extendedComparison(subdata, 0, nAtoms, false, mt);      
            }

            if (score > bestScore) {
                bestScore = score;
                topCluster = i;
            }
        }
        return topCluster;
    };
    bool splitCluster(Index clusterToSplit, int minFrames) {
        if (clusters[clusterToSplit].size() < 2 * minFrames) {
            std::cerr << "There are not enough poitns to split the cluster further." << std::endl;
            return false;
        }
        Mat subdata = data(clusters[clusterToSplit], Eigen::all);
        if (anchorType == MD::DivineAnchors::NANI) {
            KmeansNANI kmeans(subdata, 2, mt, nAtoms, kinit, percentage);
            ArrayXi sublabels = kmeans.getLabels();
            vector<Index> cluster1, cluster2;
            for (Index i = 0; i < sublabels.size(); ++i) {
                if (sublabels[i] == 0) {
                    cluster1.push_back(clusters[clusterToSplit][i]);
                } else {
                    cluster2.push_back(clusters[clusterToSplit][i]);
                }
            }
            if (cluster1.size() < minFrames || cluster2.size() < minFrames) {
                std::cerr << "One of the clusters after the split is smaller than the minimum frame requirement." << std::endl;
                return false;
            }
            clusters[clusterToSplit] = cluster1;
            clusters.push_back(cluster2);
        } else if (anchorType == MD::DivineAnchors::OutlierPair || anchorType == MD::DivineAnchors::SplinterPair) {
            Index outlierIdx = calculateOutlier(subdata, nAtoms, mt);
            Vec anchorA = subdata.row(outlierIdx);
            Vec dists = (subdata.rowwise() - anchorA.transpose()).square().rowwise().sum() / nAtoms;
            Index idxFurthest;
            dists.maxCoeff(&idxFurthest);
            Vec anchorB = subdata.row(idxFurthest);

            Mat dataC = data(clusters[clusterToSplit], Eigen::all);
            Vec dA = (dataC.rowwise() - anchorA.transpose()).square().rowwise().sum() / nAtoms;
            Vec dB = (dataC.rowwise() - anchorB.transpose()).square().rowwise().sum() / nAtoms;

            vector<Index> initialMask;
            vector<Index> notInitialMask;
            initialMask.reserve(subdata.rows());
            notInitialMask.reserve(subdata.rows());
            for (Index i = 0; i < dA.size(); ++i) {
                if (dA[i] < dB[i]) {
                    initialMask.push_back(clusters[clusterToSplit][i]);
                } else {
                    notInitialMask.push_back(clusters[clusterToSplit][i]);
                }
            }

            if (refine) {
                Mat groupA = subdata(initialMask, Eigen::all);
                Mat groupB = subdata(notInitialMask, Eigen::all);
                Index medoidA = groupA.size() <= 2 ? 0 : calculateMedoid(groupA, nAtoms, mt);
                Index medoidB = groupB.size() <= 2 ? 0 : calculateMedoid(groupB, nAtoms, mt);
                Mat initiators = Mat::Zero(2, data.row(0).size());
                initiators.row(0) = groupA.row(medoidA);
                initiators.row(1) = groupB.row(medoidB);
                KmeansNANI kmeans(subdata, 2, mt, nAtoms, initiators);
                Veci sublabels = kmeans.getLabels();
                vector<Index> cluster1, cluster2;
                for (Index i = 0; i < sublabels.size(); ++i) {
                    if (sublabels[i] == 0) {
                        cluster1.push_back(clusters[clusterToSplit][i]);
                    } else {
                        cluster2.push_back(clusters[clusterToSplit][i]);
                    }
                }
                if (cluster1.size() < minFrames || cluster2.size() < minFrames) {
                    std::cerr << "One of the clusters after the split is smaller than the minimum frame requirement." << std::endl;
                    return false;
                }
                clusters[clusterToSplit] = cluster1;
                clusters.push_back(cluster2);

            } else {
                if (initialMask.size() < minFrames || notInitialMask.size() < minFrames) {
                    std::cerr << "One of the clusters after the split is smaller than the minimum frame requirement." << std::endl;
                    return false;
                }
                clusters[clusterToSplit] = initialMask;
                clusters.push_back(notInitialMask);
                
            }
        } else if (anchorType == MD::DivineAnchors::SplinterPair) {
            Index splinterIdx = calculateOutlier(subdata, nAtoms, mt);
            Vec splinterPoint = subdata.row(splinterIdx);

            Index medoidIdx = calculateMedoid(subdata, nAtoms, mt);
            Vec medoidPoint = subdata.row(medoidIdx);

            vector<Index> splinterGroup = {splinterIdx};
            vector<Index> mainGroup;
            splinterGroup.reserve(subdata.rows() - 1);
            mainGroup.reserve(subdata.rows() - 1);

            for (Index i = 0; i < subdata.rows(); ++i) {
                if (i == splinterIdx) continue;

                double dS = (subdata.row(i) - splinterPoint).square().sum() / nAtoms;
                double dM = (subdata.row(i) - medoidPoint).square().sum() / nAtoms;

                if (dS < dM) {
                    splinterGroup.push_back(i);
                } else {
                    mainGroup.push_back(i);
                }
            }
            if (refine) {
                Mat groupA = subdata(mainGroup, Eigen::all);
                Mat groupB = subdata(splinterGroup, Eigen::all);
                Index medoidA = splinterGroup.size() <= 2 ? 0 : calculateMedoid(groupA, nAtoms, mt);
                Index medoidB = groupB.size() <= 2 ? 0 : calculateMedoid(groupB, nAtoms, mt);
                Mat initiators = Mat::Zero(2, data.row(0).size());
                initiators.row(0) = groupA.row(medoidA);
                initiators.row(1) = groupB.row(medoidB);
                KmeansNANI kmeans(subdata, 2, mt, nAtoms, initiators);
                Veci sublabels = kmeans.getLabels();
                vector<Index> cluster1, cluster2;
                for (Index i = 0; i < sublabels.size(); ++i) {
                    if (sublabels[i] == 0) {
                        cluster1.push_back(clusters[clusterToSplit][i]);
                    } else {
                        cluster2.push_back(clusters[clusterToSplit][i]);
                    }
                }
                if (cluster1.size() < minFrames || cluster2.size() < minFrames) {
                    std::cerr << "One of the clusters after the split is smaller than the minimum frame requirement." << std::endl;
                    return false;
                }
                clusters[clusterToSplit] = cluster1;
                clusters.push_back(cluster2);

            } else {
                if (mainGroup.size() < minFrames || splinterGroup.size() < minFrames) {
                    std::cerr << "One of the clusters after the split is smaller than the minimum frame requirement." << std::endl;
                    return false;
                }
                clusters[clusterToSplit] = mainGroup;
                clusters.push_back(splinterGroup);
                
            }

        }
        return true;
    };
public:
    Divine(Mat data, MD::DivineSplit splitType = MD::DivineSplit::WeightedMSD, MD::DivineAnchors anchorType = MD::DivineAnchors::NANI, MD::KinitType kinit = MD::KinitType::StratAll, int k = 0, bool refine = true, int nAtoms = 1, double threshold = 0, int percenntage = 10) : data(data), splitType(splitType), anchorType(anchorType), kinit(kinit), refine(refine), nAtoms(nAtoms), threshold(threshold), percentage(percenntage), mt(MD::Metric::MSD) {
        if (k == 0) {
            kClusters = data.rows();
        } else {
            kClusters = k;
        }
        divisiveAlgorithm();
    };
};