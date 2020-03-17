#ifndef __RBF_H__
#define __RBF_H__

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define NUM_BITS 32  // we want to store some shorts but also some ints, so need 4 bytes
#define HIGH_BIT (NUM_BITS - 1)  // low bit is 0, high bit is 15
#define HIGH_BIT_1 (-1 << HIGH_BIT)
#define NUM_CHARS 256


// typedef these so they're easier to change if ever needed
typedef uint8_t feature_type;   // HAS TO BE AN UNSIGNED TYPE!
typedef int32_t rownum_type;    // HAS TO BE A SIGNED TYPE! (see "Ugliness alert" below)
typedef int32_t colnum_type;
typedef int32_t stats_type;
typedef size_t treeindex_type;


typedef struct {
	// We have arrays of arrays of features. Instead of expensively moving those rows around when
	// sorting and partitioning we have an index into those and move the index elements around.
	// Lookups will be slightly slower but we'll save time overall.
    rownum_type *row_index;
    rownum_type num_rows;

	// Ugliness alert:
	// Each tree node is a pair. For speed and space efficiency we'll store the tree in 2 arrays
	// using the standard trick for storing a binary tree in an array (with indexing starting at 0,
	// left child of n goes in 2n+1, right child goes in 2n+2). The pairs are either:
	// - if it's an internal node: the feature number and the value at which to split the feature
	// - if it's a leaf node: start and end indices in the rowIndex array; that view in the rowIndex
	//   array tells us the indices of rows in the original training set that are in this leaf
	// We distinguish the two cases by doing some bit-arithmetic.
	// 1. Yes, I know this is ugly, but the alternative is to have a whole 'nother pair of large arrays.
	// 2. Yes, I considered using hashmaps instead [in Go], but they're much slower (expected) and also
	//    take WAY more memory (which surprised me).
    rownum_type *tree_first;
    rownum_type *tree_second;
    treeindex_type tree_size;
    treeindex_type num_internal_nodes;
    treeindex_type num_leaves;
} RandomBinaryTree;

typedef struct {
    size_t num_trees;
    size_t tree_depth;
    size_t leaf_size;
    rownum_type num_rows;
    colnum_type num_features;
    colnum_type num_features_to_compare;
} RbfConfig;

typedef struct {
    RbfConfig *config;
    RandomBinaryTree *trees;
} RandomBinaryForest;

typedef struct {
    rownum_type **tree_results;
    size_t *tree_result_counts;
    size_t total_count;
} RbfResults;


void rbf_init();

RandomBinaryForest *train_forest(feature_type *feature_array, RbfConfig *config);

RbfResults *query_forest_all_results(const RandomBinaryForest *forest, const feature_type *point,
        const size_t point_dimension);
RbfResults *batch_query_forest_all_results(const RandomBinaryForest *forest, const feature_type *points,
        const size_t point_dimension, const size_t num_points);

rownum_type *query_forest_dedup_results(const RandomBinaryForest *forest, const feature_type *point,
        const size_t point_dimension, size_t *count);
rownum_type **batch_query_forest_dedup_results(const RandomBinaryForest *forest, const feature_type *points,
        const size_t point_dimension, const size_t num_points, size_t **counts);

rownum_type **batch_query_forest_dedup_results_sorted(const RandomBinaryForest *forest, feature_type *ref_points,
        feature_type *points, const size_t point_dimension, size_t num_points,
        const int (*compare)(const void *, const void *),
        size_t **ret_counts);

feature_type *transpose(feature_type *input, size_t rows, size_t cols);

int l2_compare(const void *pre_v1, const void *pre_v2);

// Used for debugging
void print_time(char *msg);

#endif /* __RBF_H__ */
