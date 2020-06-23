#define _GNU_SOURCE     /* Expose declaration of tdestroy() */
#include <search.h>
#include <assert.h>
#include "rbf.h"
#include "_rbf_utils.h"


// A "point" is a feature-array. Search for one point in this tree.
void query_tree(const RandomBinaryForest *forest, const size_t tree_num, const feature_type *point,
                rownum_type **tree_results, size_t *tree_result_counts) {
// TODO (BUG): for my original application I wanted the single nearest neighbor.
// If we want k > 1 neighbors then for now we restrict leaf-size and get the k nearest neighbors
// found by the whole forest. Need to fix this to return k neighbors as follows:
// At each node, also store the start and end indices of points stored under it.
// Then, when querying, if the child has fewer points than we want, then don't recurse.
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


/*
 * A "point" is a feature-array. Search for one point in this forest.
 * Return:
 * - tree_result_counts: for each tree, a count of the number of results.
 * - tree_results: for each tree, indices into the training feature-array
 *                 (since the caller/wrapper might have different things they want to do with this).
 * - total_count: total count of results from all trees
 */
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


/*
 * Identical to query_forest_all_results except queries for a batch of points at a time.
 * So: `points` is now a pointer to multiple points, not a single point.
 * And the return is an array of RbfResults instead of a pointer to a single RbfResults object.
 */
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


/*
 * A "point" is a feature-array. Search for one point in this forest.
 * Return: combine and dedup result indices from all trees. Results are indices into the training
 *         feature-array (since the caller/wrapper might have different things they want to do with this).
 */
static int compare(const void *pa, const void *pb) {
   if (*(rownum_type *) pa < *(rownum_type *) pb)
       return -1;
   if (*(rownum_type *) pa > *(rownum_type *) pb)
       return 1;
   return 0;
}

rownum_type *query_forest_dedup_results(const RandomBinaryForest *forest, const feature_type *point, const size_t point_dimension, size_t *count) {
    // get all results, and accordingly allocate space for tracker and return
    RbfResults *all_results = query_forest_all_results(forest, point, point_dimension);
    // TODO: speed/memory tradeoff here: allocating too much space right now
    rownum_type *deduped_results = malloc(sizeof(rownum_type) * all_results->total_count);

    void *results_seen = NULL;
    rownum_type *result_pos;
    *count = 0;
    for (size_t i = 0; i < forest->config->num_trees; i++) {
        for (size_t j = 0; j < all_results->tree_result_counts[i]; j++) {
            result_pos = &(all_results->tree_results[i][j]);
            if (!tfind(result_pos, &results_seen, compare)) {
                deduped_results[*count] = *result_pos;
                tsearch(result_pos, &results_seen, compare);    // insert
                *count += 1;
            }
        }
    }

    // TODO: No idea why this fails!
    //if (results_seen) {
    //    tdestroy(results_seen, free);
    //}
    return deduped_results;
}


/*
 * Identical to query_forest_dedup_results except queries for a batch of points at a time.
 * So: `points` is now a pointer to multiple points, not a single point.
 * Returns an array of arrays. Param return: ret_counts array of counts.
 * Outer array contains num_points results; each result is an array.
 * The ith of these arrays contains ret_counts[i]-many indices.
 */
rownum_type **batch_query_forest_dedup_results(const RandomBinaryForest *forest, const feature_type *points,
        const size_t point_dimension, const size_t num_points, size_t **ret_counts) {
    assert(point_dimension == forest->config->num_features);
    rownum_type **all_results = malloc(sizeof(rownum_type*) * num_points);
    *ret_counts = malloc(sizeof(size_t) * num_points);
    #pragma omp parallel for
    for (size_t i = 0; i < num_points; i++) {
        all_results[i] = query_forest_dedup_results(forest, &(points[i * point_dimension]), point_dimension, &((*ret_counts)[i]));
    }
    return all_results;
}


// Internal use only.
// Problem: we want to qsort indices into the row-index array by distance of each indexed reference point
// from the query point. To use qsort we'll have to carry some metadata along with each index.
// This function builds structs containing the needed metadata (index, query point, reference point,
// and the two points' dimensions).
results_comparison_node *make_comp_nodes(rownum_type *unsorted_results, size_t count,
        feature_type *ref_points, feature_type *point, size_t point_dimension) {
    results_comparison_node *comp_nodes = malloc(sizeof(results_comparison_node) * count);
    for (size_t i = 0; i < count; i++) {
        comp_nodes[i].query_point = point;
        comp_nodes[i].ref_point = &(ref_points[unsorted_results[i] * point_dimension]);
        comp_nodes[i].ref_index = unsorted_results[i];
        comp_nodes[i].point_dimension = point_dimension;
    }
    return comp_nodes;
}


/*
 * A "point" is a feature-array. Search for one point in this forest.
 * Return: combine and dedup result indices from all trees, sorted by the given comparison function.
 *         Results are indices into the training feature-array (since the caller/wrapper might have
 *         different things they want to do with this).
 */
rownum_type *query_forest_dedup_results_sorted(const RandomBinaryForest *forest, feature_type *point,
        feature_type *ref_points, const size_t point_dimension, size_t *count,
        int (*compare)(const void *, const void *)) {
    rownum_type *results = query_forest_dedup_results(forest, point, point_dimension, count);
    results_comparison_node *results_for_sort = make_comp_nodes(results, *count, ref_points, point, point_dimension);
    qsort(results_for_sort, *count, sizeof(results_comparison_node), compare);
    for (size_t i = 0; i < *count; i++) {
        results[i] = results_for_sort[i].ref_index;
    }
    free(results_for_sort);
    return results;
}


/*
 * Identical to query_forest_dedup_results_sorted except queries for a batch of points at a time.
 * So: `points` is now a pointer to multiple points, not a single point.
 * Returns an array of arrays. Param return: ret_counts array of counts.
 * Outer array contains num_points results; each result is an array.
 * The ith of these arrays contains ret_counts[i]-many indices.
 */
rownum_type **batch_query_forest_dedup_results_sorted(const RandomBinaryForest *forest, feature_type *ref_points,
        feature_type *points, const size_t point_dimension, size_t num_points,
        const int (*compare)(const void *, const void *),
// Note: the params here are actually all const (except ret_counts)
// but I can't declare them const because of make_comp_nodes.
        size_t **ret_counts) {
    assert(point_dimension == forest->config->num_features);
    rownum_type **all_results = (rownum_type **) malloc(sizeof(rownum_type*) * num_points);
    *ret_counts = (size_t *) malloc(sizeof(size_t) * num_points);
    if (!all_results || !ret_counts) {
        die_alloc_err("batch_query_forest_dedup_results_sorted", "all_results or ret_counts failed");
    }
    #pragma omp parallel for
    for (size_t i = 0; i < num_points; i++) {
        all_results[i] = query_forest_dedup_results_sorted(forest, &(points[i * point_dimension]), ref_points, point_dimension, &((*ret_counts)[i]), compare);
    }
    return all_results;
}
