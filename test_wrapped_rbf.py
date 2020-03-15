#!/usr/bin/env python
import ctypes


rbf = ctypes.CDLL("librbf.so")


feature_type = ctypes.c_uint8   # but note that the tranpose and query functions take type ctypes.c_char because we want to pass byte-strings
rownum_type = ctypes.c_int32
colnum_type = ctypes.c_int32
stats_type = ctypes.c_int32
treeindex_type = ctypes.c_size_t


class RbfConfig(ctypes.Structure):
    _fields_ = [("num_trees", ctypes.c_size_t),
                ("tree_depth", ctypes.c_size_t),
                ("leaf_size", ctypes.c_size_t),
                ("num_rows", rownum_type),
                ("num_features", colnum_type),
                ("num_features_to_compare", colnum_type)]

    def __init__(self, num_trees, tree_depth, leaf_size, num_rows, num_features, num_features_to_compare):
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.leaf_size = leaf_size
        self.num_rows = num_rows
        self.num_features = num_features
        self.num_features_to_compare = num_features_to_compare

    def __repr__(self):
        return f"num_trees: {self.num_trees}, tree_depth: {self.tree_depth}, leaf_size: {self.leaf_size}, num_rows: {self.num_rows}, num_features: {self.num_features}, num_features_to_compare: {self.num_features_to_compare}"

# These don't need to be visible in Python: just treat the RBF* as a void*.
#
# class RandomBinaryTree(ctypes.Structure):
#     _fields_ = [("row_index", ctypes.POINTER(rownum_type)),
#                 ("num_rows", rownum_type),
#                 ("tree_first", ctypes.POINTER(rownum_type)),
#                 ("tree_second", ctypes.POINTER(rownum_type)),
#                 ("tree_size", treeindex_type),
#                 ("num_internal_nodes", treeindex_type),
#                 ("num_leaves", treeindex_type)]
#
# class RandomBinaryForest(ctypes.Structure):
#     _fields_ = [("config", ctypes.POINTER(RbfConfig)),
#                 ("trees", ctypes.POINTER(RandomBinaryTree))]


rbf_type = ctypes.c_void_p

class RbfResults(ctypes.Structure):
    _fields_ = [("tree_results", ctypes.POINTER(ctypes.POINTER(rownum_type))),
                ("tree_results_counts", ctypes.POINTER(ctypes.c_size_t)),
                ("total_count", ctypes.c_size_t)]


transpose = rbf.__getattr__("transpose")
transpose.restype = ctypes.POINTER(feature_type)
transpose.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_size_t, ctypes.c_size_t]

train_forest = rbf.__getattr__("train_forest")
train_forest.restype = rbf_type
train_forest.argtypes = [ctypes.POINTER(feature_type), ctypes.POINTER(RbfConfig)]

query_forest_all_results = rbf.__getattr__("query_forest_all_results")
query_forest_all_results.restype = ctypes.POINTER(RbfResults)
query_forest_all_results.argtypes = [rbf_type, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]

batch_query_forest_all_results = rbf.__getattr__("batch_query_forest_all_results")
batch_query_forest_all_results.restype = ctypes.POINTER(RbfResults)
batch_query_forest_all_results.argtypes = [rbf_type, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t, ctypes.c_size_t]

query_forest_dedup_results = rbf.__getattr__("query_forest_dedup_results")
query_forest_dedup_results.restype = ctypes.POINTER(rownum_type)
query_forest_dedup_results.argtypes = [rbf_type, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]

batch_query_forest_dedup_results = rbf.__getattr__("batch_query_forest_dedup_results")
batch_query_forest_dedup_results.restype = ctypes.POINTER(ctypes.POINTER(rownum_type))
batch_query_forest_dedup_results.argtypes = [rbf_type, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]

l2_square_dist = rbf.__getattr__("l2_square_dist")
l2_square_dist.restype = ctypes.c_int
# TODO: fix the type issues here: they should both be of the same type.
# Works right now because you're testing with arg1 == features (training feature array), which has been returned from `transpose` so is of the right type
# In real use it'll likely be of the same type as arg2 (which is right now test_features, so of type `bytes`)
l2_square_dist.argtypes = [ctypes.POINTER(feature_type), ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]


if __name__ == '__main__':
    with open("features.bin", "rb") as f:
        pre_transpose_features = f.read()

    config = RbfConfig(20, 25, 64, 151511, 37*37, 37)
    features = transpose(pre_transpose_features, config.num_rows, config.num_features)
    forest = train_forest(features, ctypes.byref(config))

    results = query_forest(forest, pre_transpose_features[:1369], 1369)
    results_obj = results[0]
    for i in range(20):
        print(results_obj.tree_results_counts[i])
    real_results = [results_obj.tree_results[i][j] for i in range(20) for j in range(results_obj.tree_results_counts[i])]
