#!/usr/bin/env python
import ctypes


rbf = ctypes.CDLL("librbf.so")


feature_type = ctypes.c_char
rownum_type = ctypes.c_uint32
colnum_type = ctypes.c_uint32
stats_type = ctypes.c_uint32
treeindex_type = ctypes.c_size_t


class RbfConfig(ctypes.Structure):
    _fields_ = [("num_trees", ctypes.c_size_t),
                ("tree_depth", ctypes.c_size_t),
                ("leaf_size", ctypes.c_size_t),
                ("num_rows", ctypes.c_uint32),
                ("num_features", ctypes.c_uint32),
                ("num_features_to_compare", ctypes.c_uint32)]

    def __init__(self, num_trees, tree_depth, leaf_size, num_rows, num_features, num_features_to_compare):
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.leaf_size = leaf_size
        self.num_rows = num_rows
        self.num_features = num_features
        self.num_features_to_compare = num_features_to_compare

    def __repr__(self):
        return f"num_trees: {self.num_trees}, tree_depth: {self.tree_depth}, leaf_size: {self.leaf_size}, num_rows: {self.num_rows}, num_features: {self.num_features}, num_features_to_compare: {self.num_features_to_compare}"

class RandomBinaryTree(ctypes.Structure):
    _fields_ = [("row_index", ctypes.POINTER(rownum_type)),
                ("num_rows", rownum_type),
                ("tree_first", ctypes.POINTER(rownum_type)),
                ("tree_second", ctypes.POINTER(rownum_type)),
                ("tree_size", treeindex_type),
                ("num_internal_nodes", treeindex_type),
                ("num_leaves", treeindex_type)]

class RandomBinaryForest(ctypes.Structure):
    _fields_ = [("config", ctypes.POINTER(RbfConfig)),
                ("trees", ctypes.POINTER(RandomBinaryTree))]


class RbfResults(ctypes.Structure):
    _fields_ = [("tree_results", ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))), ("tree_results_counts", ctypes.POINTER(ctypes.c_int))]


train_forest = rbf.__getattr__("train_forest")
train_forest.restype = ctypes.POINTER(RandomBinaryForest)
train_forest.argtypes = [ctypes.POINTER(feature_type), ctypes.POINTER(RbfConfig)]

query_forest = rbf.__getattr__("query_forest")
query_forest.restype = ctypes.POINTER(RbfResults)
query_forest.argtypes = [ctypes.c_void_p, ctypes.POINTER(feature_type), ctypes.c_size_t]


# import sys
# sys.path.append("/home/moy/signifyd/addresses/rbf")
# import logging
# import time
# from typing import List
#
# from multiprocessing import Pool
# from multiprocessing.sharedctypes import RawArray
#
# import fastDamerauLevenshtein as fdl
#
# import followgrams as f
# followgrams = f.get_followgrams("this is the way the world ends")


if __name__ == '__main__':
    with open("features.bin", "rb") as f:
        feature_array = f.read()

    config = RbfConfig(20, 25, 64, 151511, 37*37, 37)
    forest = train_forest(feature_array, ctypes.byref(config))

    results = query_forest(forest, feature_array[:1369], 1369)
    results_obj = results[0]
    # print("Results: ", results_obj)
    for i in range(20):
        print(results_obj.tree_results_counts[i])
