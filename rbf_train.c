#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
       colnum_t feature_num, rownum_t num_rows, rownum_t index_start, rownum_t index_end,
       // returns:
       stats_t *ret_counts, stats_t *ret_weighted_total) {
    // get frequencies:
    for (rownum_t rownum = index_start; rownum < index_end; rownum++) {
        feature_t feature_value = local_feature_array[num_rows * feature_num + local_row_index[rownum]];
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
        rbf_config_t *cfg, rownum_t index_start, rownum_t index_end,
        // returns:
        colnum_t *ret_feature_subset, stats_t *ret_feature_frequencies, stats_t *ret_feature_weighted_totals) {
    bool *features_already_selected = (bool *) calloc(sizeof(bool), cfg->num_features);
    for (colnum_t i = 0; i < cfg->num_features_to_compare; i++) {
        colnum_t feature_num = rand() % cfg->num_features;
        while (features_already_selected[(size_t) feature_num]) {
            feature_num = rand() % cfg->num_features;
        }
        features_already_selected[feature_num] = 1;
        ret_feature_subset[i] = feature_num;
        get_feature_frequencies(row_index, feature_array,
                                feature_num, cfg->num_rows, index_start, index_end,
                                &(ret_feature_frequencies[i * NUM_CHARS]), &(ret_feature_weighted_totals[i]));
    }
    free(features_already_selected);
}


// Split a set of rows on one feature, trying to get close to the median but also maximizing
// variance.
//
// NOTE: We're no longer using the variance but I'm leaving all this (code and comments) unchanged.
// We'll make it an option later. For our current use our features are sufficiently skewed that
// using variance is unhelpful, so we simply find the split closest to the median. So we're using
// `getSimpleBestFeature` instead of `getBestFeature`.
//
// We want something as close to the median as possible so as to make the tree more balanced.
// And we want to calculate the "variance" about this split to compare features.
//
// CLEVERNESS ALERT (violating the "don't be clever" rule for speed):
// Except we'll actually use the mean absolute deviation instead of the variance as it's easier and
// better, esp since we're thinking of this in terms of Manhattan distance anyway. In fact, for our
// purposes it suffices to calculate the *total* absolute deviation, i.e. the total moment: we don't
// really need the mean since the denominator, the number of rows, is the same for all features that
// we're going to compare.
//
// The total moment to the right of some b, say for example b = 7.5, is
//     \\sum_{i=8}^{255} (i-7.5) * x_i = [ \\sum_{i=0}^255 (i-7.5) * x_i ] - [ \\sum_{i=0}^7 (i-7.5) x_i ]
// That second term is actually just -(the moment to the left of b), so the total moment
// (i.e. left + right) simplifies down to
//     \\sum_{i=0}^255 i x_i - \\sum_{i=0}^255 7.5 x_i + 2 \\sum_{i=0}^7 7.5 x_i - 2 \\sum_{i=0}^7 i x_i
// So we only need to track the running left-count and the running left-moment (w.r.t. 0), and then
// we can calculate the total moment w.r.t. median when we're done.
//
// Summary: Starting at 0.5 (no use starting at 0), iterate (a) adding to simple count, and (b)
// adding to left-side total moment. Stop as soon as the count is greater than half the total number
// of rows, and at that point we have a single expression for the total moment.
void split_one_feature(stats_t *feature_bins, stats_t total_zero_moment, stats_t count,
        // returns:
        double *total_moment, size_t *pos, stats_t *left_count) {
    *pos = 0;
    *left_count = feature_bins[0];
    stats_t fifty_percentile = count / 2;
    stats_t left_zero_moment = 0, this_item_moment;
    stats_t this_item_count;
    while (*left_count <= fifty_percentile) {
        *pos += 1;
        this_item_count = feature_bins[*pos];
        this_item_moment = this_item_count * *pos;
        *left_count += this_item_count;
        left_zero_moment += this_item_moment;
    }
    double real_pos = *pos + 0.5;   // want moment about e.g. 7.5, not 7 (using numbers in example above)
                                    // See moment computation example in comment above
    *total_moment = total_zero_moment - (real_pos * count) + (2 * ((real_pos * *left_count) - left_zero_moment));
    return;
}


// From the given features find the one which splits closest to the median.
void get_simple_best_feature(stats_t *feature_frequencies,
        colnum_t num_features_to_compare, stats_t *feature_weighted_totals, stats_t total_count,
        // returns:
        colnum_t *best_feature_num, feature_t *best_feature_split_value) {
    *best_feature_num = 0;
    *best_feature_split_value = 0;
    stats_t min_split_balance = total_count, split_balance;
    stats_t left_count;
    size_t split_value;
    double ignore;
    for (colnum_t i = 0; i < num_features_to_compare; i++) {
        split_one_feature(&(feature_frequencies[i * NUM_CHARS]), feature_weighted_totals[i], total_count,
                &ignore, &split_value, &left_count);
        split_balance = abs(left_count - (total_count - left_count));   // left_count - right_count
        if (split_balance < min_split_balance) {
            min_split_balance = split_balance;
            *best_feature_split_value = split_value;
            *best_feature_num = i;
        }
    }
    return;
}


// quicksort-type partitioning of row_index[index_start..index_end] based on whether the
// feature `feature_num` is less-than or greater-than-or-equal-to split_value
rownum_t quick_partition(rownum_t *local_row_index, feature_t *local_feature_array,
        rownum_t num_rows, rownum_t index_start, rownum_t index_end, colnum_t feature_num, feature_t split_value) {
    if (index_end <= index_start) {
        return index_start;
    }

    rownum_t i = index_start, j = index_end - 1;
    while (i < j) {
        while ((i < index_end) && (local_feature_array[num_rows * feature_num + local_row_index[i]] <= split_value)) {
            i += 1;
        }
        while ((j >= index_start) && (local_feature_array[num_rows * feature_num + local_row_index[j]] > split_value) && (j > 0)) {
            j -= 1;
        }
        if (i >= j) {
            return i;
        }
        rownum_t tmp = local_row_index[i];
        local_row_index[i] = local_row_index[j];
        local_row_index[j] = tmp;
    }
    return index_start;  // should never get here
}


// Get a random subset of features, find the best one of those features,
// and split this set of nodes on that feature.
void _split_node(rownum_t *row_index, feature_t *feature_array, rbf_config_t *cfg,
        rownum_t index_start, rownum_t index_end,
        // returns:
        colnum_t *best_feature_num, feature_t *best_feature_split_value, rownum_t *split_pos) {
    colnum_t *feature_subset = (colnum_t *) calloc(sizeof(colnum_t), cfg->num_features_to_compare);
    stats_t *feature_frequencies = (stats_t *) calloc(sizeof(stats_t), cfg->num_features_to_compare * NUM_CHARS);
    stats_t *weighted_totals = (stats_t *) calloc(sizeof(stats_t), cfg->num_features_to_compare);
    colnum_t _best_feature_index;

    select_random_features_and_get_frequencies(row_index, feature_array,
            cfg, index_start, index_end,
            feature_subset, feature_frequencies, weighted_totals);
    get_simple_best_feature(feature_frequencies, cfg->num_features_to_compare, weighted_totals, index_end - index_start,
            &_best_feature_index, best_feature_split_value);
    *best_feature_num = feature_subset[_best_feature_index];
    // return values:
    *split_pos = quick_partition(row_index, feature_array, cfg->num_rows, index_start, index_end, *best_feature_num, *best_feature_split_value);
    free(feature_subset);
    free(feature_frequencies);
    free(weighted_totals);
    return;
}


// Calculate the split (or leaf) at one node (and its descendants).
// So this is doing all the real work of building the tree.
// Params:
// - tree we're building
// - feature array
// - leaf size, total number of features, and number of features to compare
//   (not adding these to the tree struct b/c they're only needed at training time)
// - num_rows: number of rows in the feature-array and in the tree's row_index
// - index_start and index_end: the view into row_index that we're considering right now
// - tree_array_pos: the position of this node in the tree arrays
// - TODO: REMOVE depth of this node in the tree
// Guarantees:
// - Parallel calls to `calculate_one_node` will look at non-intersecting views.
// - Child calls will look at distinct sub-views of this view.
// - No two calls to `calculate_one_node` will have the same tree_array_pos
void calculate_one_node(RandomBinaryTree *tree, feature_t *feature_array, rbf_config_t *config,
        rownum_t index_start, rownum_t index_end, treeindex_t tree_array_pos, size_t depth) {
    if (2 * tree_array_pos + 2 >= tree->tree_size) {
    // Special termination condition to regulate depth.
        tree->tree_first[tree_array_pos] = HIGH_BIT_1 ^ index_start;
        tree->tree_second[tree_array_pos] = HIGH_BIT_1 ^ index_end;
        tree->num_leaves += 1;
// fmt.Fprintf(tree_statsFile, "%d,%d,depth-based-leaf,%d,%d,%d,%d,%d,%d,\n", tree_array_pos, depth, index_start, index_end, index_end-index_start, 0, 0, 0)
        return;
    }

    if (index_end - index_start < config->leaf_size) {
    // Not enough items left to split. Make a leaf.
        tree->tree_first[tree_array_pos] = HIGH_BIT_1 ^ index_start;
        tree->tree_second[tree_array_pos] = HIGH_BIT_1 ^ index_end;
        tree->num_leaves += 1;
// fmt.Fprintf(tree_statsFile, "%d,%d,size-based-leaf,%d,%d,%d,%d,%d,%d,\n", tree_array_pos, depth, index_start, index_end, index_end-index_start, 0, 0, 0)
    } else {
    // Not a leaf. Get a random subset of numFeaturesToCompare features, find the best one, and split this node.
    // TODO (not sure where): pick feature so that each side has at least a third of data, else don't bother splitting if below a threshold
    //      or look at more features or something
        colnum_t best_feature_num;
        feature_t best_feature_split_value;
        rownum_t index_split;
        _split_node(tree->row_index, feature_array, config, index_start, index_end,
                    &best_feature_num, &best_feature_split_value, &index_split);

        tree->tree_first[tree_array_pos] = best_feature_num;
        tree->tree_second[tree_array_pos] = best_feature_split_value;
// fmt.Fprintf(tree_statsFile, "%d,%d,internal,%d,%d,%d,%d,%d,%d,%s\n", tree_array_pos, depth, index_start, index_end,
//        index_end - index_start, index_split, featureNum, featureSplitValue, features.CHAR_REVERSE_MAP[featureNum])
        tree->num_internal_nodes += 1;
        calculate_one_node(tree, feature_array, config, index_start, index_split, (2*tree_array_pos)+1, depth+1);
        calculate_one_node(tree, feature_array, config, index_split, index_end, (2*tree_array_pos)+2, depth+1);
    }
}


RandomBinaryTree *create_rbt(rbf_config_t *config) {
    treeindex_t tree_size = 1 << config->tree_depth;
    // TODO: deal with NULLs here (but there's no intelligent way to recover, so maybe just fail)
    RandomBinaryTree *tree = (RandomBinaryTree *) malloc(sizeof(RandomBinaryTree));

    tree->row_index = (rownum_t *) malloc(sizeof(rownum_t) * config->num_rows);
    for (rownum_t i = 0; i < config->num_rows; i++) {
        tree->row_index[i] = i;
    }
    tree->num_rows = config->num_rows;

    tree->tree_first = (rownum_t *) calloc(sizeof(rownum_t), (size_t) tree_size);
    tree->tree_second = (rownum_t *) calloc(sizeof(rownum_t), (size_t) tree_size);
    tree->tree_size = tree_size;

    tree->num_internal_nodes = 0;
    tree->num_leaves = 0;
    return tree;
}


char buffer[26];
void print_time(char *msg) {
    //time_t my_time = time(NULL);
    //strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", localtime(&my_time));
    //printf("%s: %s\n", buffer, msg);

    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    double count;
    count = t.tv_sec * 1e9;
    count = (count + t.tv_nsec) * 1e-9;
    printf("%f: %s\n", count, msg);
}


RandomBinaryTree *train_one_tree(feature_t *feature_array, rbf_config_t *config) {
    RandomBinaryTree *tree = create_rbt(config);
    print_time("starting tree");
    calculate_one_node(tree, feature_array, config, 0, config->num_rows, 0, 0);
    print_time("finished tree");
    return tree;
}


feature_t *transpose(feature_t *input, size_t rows, size_t cols) {
    feature_t *output = (feature_t *) malloc(sizeof(feature_t) * rows * cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
    return output;
}


RandomBinaryTree **train_forest_with_feature_array(feature_t *feature_array, rbf_config_t *config) {
    RandomBinaryTree **forest = (RandomBinaryTree **) malloc(sizeof(void *) * config->num_trees);
    #pragma omp parallel for
    for (size_t i = 0; i < config->num_trees; i++) {
        forest[i] = train_one_tree(feature_array, config);
    }
    return forest;
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
