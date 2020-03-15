#include <apr_pools.h>
#include <apr_tables.h>
#include <assert.h>
#include "rbf.h"
#include "_rbf_utils.h"


extern apr_pool_t *memory_pool;


// A "point" is a feature-array. Search for one point in this tree.
void query_tree(const RandomBinaryForest *forest, const size_t tree_num, const feature_type *point,
                rownum_type **tree_results, size_t *tree_result_counts) {
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
    if (!tree_results[tree_num]) {
        die_alloc_err("query_tree", "tree_results[tree_num]");
    }
    for (rownum_type rownum = 0; rownum < index_end - index_start; rownum++) {
        tree_results[tree_num][rownum] = forest->trees[tree_num].row_index[index_start + rownum];
    }
	return;
}


// A "point" is a feature-array. Search for one point in this forest.
// Return indices into the training feature-array (since the caller/wrapper might have
// different things they want to do with this).
RbfResults *query_forest_all_results(const RandomBinaryForest *forest, const feature_type *point, const size_t point_dimension) {
    assert(point_dimension == forest->config->num_features);
    rownum_type **tree_results = malloc(sizeof(rownum_type*) * forest->config->num_trees);
    size_t *tree_result_counts = malloc(sizeof(size_t) * forest->config->num_trees);
    size_t total_count = 0;

    for (size_t i = 0; i < forest->config->num_trees; i++) {
        query_tree(forest, i, point, tree_results, tree_result_counts);
        total_count += tree_result_counts[i];
    }

    RbfResults *results = malloc(sizeof(RbfResults));
    results->tree_results = tree_results;
    results->tree_result_counts = tree_result_counts;
    results->total_count = total_count;
    return results;
}


RbfResults *batch_query_forest_all_results(const RandomBinaryForest *forest, const feature_type *points,
        const size_t point_dimension, const size_t num_points) {
    assert(point_dimension == forest->config->num_features);
    RbfResults *all_results = malloc(sizeof(RbfResults) * num_points);
    #pragma omp parallel for
    for (size_t i = 0; i < num_points; i++) {
        all_results[i] = *query_forest_all_results(forest, &(points[i * point_dimension]), point_dimension);
    }
    return all_results;
}


rownum_type *query_forest_dedup_results(const RandomBinaryForest *forest, const feature_type *point, const size_t point_dimension, size_t *count) {
    char local_true = 1;

    // get all results, and accordingly allocate space for tracker and return
    RbfResults *all_results = query_forest_all_results(forest, point, point_dimension);
    apr_table_t *results_seen = apr_table_make(memory_pool, all_results->total_count);
    // TODO: speed/memory tradeoff here: allocating too much space right now
    rownum_type *deduped_results = malloc(sizeof(rownum_type) * all_results->total_count);

    *count = 0;
    for (size_t i = 0; i < forest->config->num_trees; i++) {
        for (size_t j = 0; j < all_results->tree_result_counts[i]; j++) {
            rownum_type *result_pos = &(all_results->tree_results[i][j]);
            if (!apr_table_get(results_seen, (char *) result_pos)) {
                deduped_results[*count] = *result_pos;
                apr_table_set(results_seen, (char *) result_pos, &local_true);
                *count += 1;
            }
        }
    }

    return deduped_results;
}


rownum_type **batch_query_forest_dedup_results(const RandomBinaryForest *forest, const feature_type *points,
        const size_t point_dimension, size_t num_points, size_t *counts) {
    assert(point_dimension == forest->config->num_features);
    rownum_type **all_results = malloc(sizeof(rownum_type*) * num_points);
    counts = malloc(sizeof(size_t) * num_points);
    #pragma omp parallel for
    for (size_t i = 0; i < num_points; i++) {
        all_results[i] = query_forest_dedup_results(forest, &(points[i * point_dimension]), point_dimension, &(counts[i]));
    }
    return all_results;
}
