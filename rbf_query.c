#include <assert.h>
#include "rbf.h"


#include <stdio.h>

// A "point" is a feature-array. Search for one point in this tree.
void query_tree(RandomBinaryForest *forest, size_t tree_num, feature_type *point, rownum_type **tree_results, size_t *tree_result_counts) {
    size_t array_pos = 0;
    rownum_type first = forest->trees[tree_num].tree_first[array_pos];
	// the condition checks if it's an internal node (== 0) or a leaf (== -1):
    while (first >> HIGH_BIT == 0) {
		// Internal node, so first (the entry in tree.tree_first) is a feature-number and
		// the entry in tree.tree_second is the feature-value at which to split.
        // Decide whether we want to recurse down the left subtree or the right subtree:
        if (point[(size_t) first] <= forest->trees[tree_num].tree_second[array_pos]) {
			array_pos = (2 * array_pos) + 1; // left subtree
        } else {
			array_pos = (2 * array_pos) + 2; // right subtree
		}
        first = forest->trees[tree_num].tree_first[array_pos];
    }
	// found a leaf; get values and return
	rownum_type index_start = HIGH_BIT_1 ^ first;
	rownum_type index_end = HIGH_BIT_1 ^ (forest->trees[tree_num].tree_second[array_pos]);
    tree_result_counts[tree_num] = index_end - index_start;
    tree_results[tree_num] = malloc(sizeof(rownum_type) * (index_end - index_start));
    for (rownum_type rownum = 0; rownum < index_end - index_start; rownum++) {
        tree_results[tree_num][rownum] = forest->trees[tree_num].row_index[index_start + rownum];
    }
	return;
}

// A "point" is a feature-array. Search for one point in this forest.
// Return indices into the training feature-array (since the caller/wrapper might have
// different things they want to do with this).
RbfResults *query_forest(RandomBinaryForest *forest, feature_type *point, size_t point_dimension) {
    assert(point_dimension == forest->config->num_features);
    rownum_type **tree_results = malloc(sizeof(rownum_type*) * forest->config->num_trees);
    size_t *tree_result_counts = malloc(sizeof(size_t) * forest->config->num_trees);

    for (size_t i = 0; i < forest->config->num_trees; i++) {
        query_tree(forest, i, point, tree_results, tree_result_counts);
    }

    RbfResults *results = malloc(sizeof(RbfResults));
    results->tree_results = tree_results;
    results->tree_result_counts = tree_result_counts;
    return results;
}
