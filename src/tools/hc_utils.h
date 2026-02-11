#ifndef HC_UTILS_H
#define HC_UTILS_H

#include <Eigen/Dense>
#include <list>

#include "types.h"

class HCTreeNode{
    // Class for the hierarchical clustering tree node
    // each node stores:
    //     fingerprints: the columwise sum of the fingerprints
    //     obj_indices: indices of the molecules in the cluster
    //     n_objects: number of molecules in the cluster
    // in addition to pointers to the left and right child nodes. 

    friend class HCTree;

    private:
        int n_objects; // number of molecules in node/cluster
        int z_index; // cluster index for Z matrix
        Vec cSum; // columnwise sum of data
        Vec sqSum; // columnwise sum of squares of data
        std::list<int> obj_indices;
        HCTreeNode* leftPtr; // pointer to left child node
        HCTreeNode* rightPtr; // pointer to right child node
        
    public:
        HCTreeNode(std::list<int> idx, Vec cSum, Vec sqsum, int nObjects, int z_ind);

        Vec getCSum();

        Vec getSQSum();

        void setCSum(Vec cSum);

        void setSQSum(Vec sqSum);

        HCTreeNode* getLeftPtr();

        void setLeftPtr(HCTreeNode* input_left);

        HCTreeNode* getRightPtr();

        void setRightPtr(HCTreeNode* input_right);

        void setNObjects(int n_mol);

        int getNObjects();

        void setIndices(std::list<int> idx);

        std::list<int> getIndices();

        void setIdx(int i);

        int getIdx();
};

class HCTree{
    private:
        HCTreeNode* rootPtr;
    public:
        HCTree(); // constructor that sets root ptr to NULL

        HCTreeNode* getRoot();

        void setRoot(HCTreeNode* input_root);

        Vec getRootCSum();

        Vec getRootSQSum();

        int getRootNObjects();

        std::list<int> getRootIndices();

        int getRootIdx();

        void insertRoot(std::list<int> idx, Vec cSum, Vec sqsum, int nObjects, int z_ind);

        void combineTrees(HCTree other_tree, int new_z_ind);
};

#endif