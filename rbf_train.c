#include <stdio.h>
#include <stdlib.h>
#include "rbf.h"
#include "_rbf_train.h"

// A "random binary forest" is a hybrid between kd-trees and random forests.
// For nearest neighbors this ends up being similar to Minhash Forests and to
// Spotify's annoy library.
//
// We build an ensemble of roughly-binary search trees, with each tree being
// built as follows: pick a random subset of features at each split, look for
// the "best" feature, split on that feature, and then recurse.
//
// We want the split to be close to the median for the best search speeds (as
// this will give us trees that are almost binary), but we want to maximize
// variance for accuracy-optimization (e.g. if we have two features
// A = [5, 5, 5, 6, 6, 6] and B = [0, 0, 0, 10, 10, 10], then we want to choose
// B so that noisy data is less likely to fall on the wrong side of the split).
//
// These two goals can conflict, so right now we just use a simple split
// function that splits closest to the median. This has the added advantage that
// you don't need to normalize features to have similar distributions.
//
// We have another split function that takes variance into account, but this is
// currently unused.


// Convert a feature column into bins. Since our features are integers in the range [0, 255],
// statistics will be faster this way.
// Returns: for feature `feature_num`:
// - the frequency of each integer value in [0, 255]
// - the sum of all feature values (i.e. the weighted sum over the frequency array)
void get_feature_frequencies(rownum_t *local_row_index, feature_t local_feature_array[],
       colnum_t feature_num, colnum_t num_features, rownum_t index_start, rownum_t index_end,
       // returns:
       stats_t *ret_counts, stats_t *ret_weighted_total) {
    // get frequencies:
    for (rownum_t rownum = index_start; rownum < index_end; rownum++) {
        feature_t feature_value = local_feature_array[local_row_index[rownum] * num_features + feature_num];
        ret_counts[(size_t) feature_value] += 1;
        ret_weighted_total[0] += feature_value;
    }
    return;
	// Is it faster to calculate weightedTotal with additions inside the loop here,
	// or with 256 multiplications and additions on the counts list later?
	// Notes:
	// - "n-4" below is because when we get down close to the leaves we don't do this any more.
	// - This is all assuming it's a binary tree, which is obviously very approximate.
	// Calculations:
	// - additions inside loop:
	//   \\sum_{k=0}^{n-4} numNodes x numAdditions = \\sum_{k=0}^{n-4} 2^k 2^{n-k} = (n-3) * 2^n
	// - 2 x 256 = 2^9 multiplications and additions on the counts list later:
	//   \\sum_{k=0}^{n-4} numNodes x 2 x 256 = \\sum_{k=0}^{n-4} 2^k 2^9 = 2^9 * (2^(n-3) - 1) = 2^6 2^n - 2^9
	// For our datasets n is 25-30, so for the full tree it's roughly a wash, maybe slightly faster
	// to do them inside the above loop.
}


// Select a random subset of features and get the frequencies for those features.
void select_random_features_and_get_frequencies(rownum_t *row_index, feature_t *feature_array,
        colnum_t num_features, colnum_t num_features_to_compare, rownum_t index_start, rownum_t index_end,
        // returns:
        colnum_t *ret_feature_subset, stats_t *ret_feature_frequencies, stats_t *ret_feature_weighted_totals) {
    bool *features_already_selected = (bool *) calloc(sizeof(bool), num_features);
    for (colnum_t i = 0; i < num_features_to_compare; i++) {
        colnum_t feature_num = rand() % num_features;
        while (features_already_selected[(size_t) feature_num]) {
            feature_num = rand() % num_features;
        }
        features_already_selected[feature_num] = 1;
        ret_feature_subset[i] = feature_num;
        get_feature_frequencies(row_index, feature_array,
                                feature_num, num_features, index_start, index_end,
                                &(ret_feature_frequencies[i * NUM_CHARS]), &(ret_feature_weighted_totals[i]));
    }
    free(features_already_selected);
}



/* display a Point value */
void show_point(Point point) {
    printf("Point in C      is (%d, %d)\n", point.x, point.y);
}

/* Increment a Point which was passed by value */
void move_point(Point point) {
    show_point(point);
    point.x++;
    point.y++;
    show_point(point);
}

/* Increment a Point which was passed by reference */
void move_point_by_ref(Point *point) {
    show_point(*point);
    point->x++;
    point->y++;
    show_point(*point);
}

/* Return by value */
Point get_point(void) {
    static int counter = 0;
    int x = counter ++;
    int y = counter ++;
    Point point = {x, y};
    printf("Returning Point    (%d, %d)\n", point.x, point.y);
    return point;
}
