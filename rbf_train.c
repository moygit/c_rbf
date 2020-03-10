/*
 * A "random binary forest" is a hybrid between kd-trees and random forests.
 * For nearest neighbors this ends up being similar to Minhash Forests and to
 * Spotify's annoy library.
 *
 * We build an ensemble of roughly-binary search trees, with each tree being
 * built as follows: pick a random subset of features at each split, look for
 * the "best" feature, split on that feature, and then recurse.
 *
 * We want the split to be close to the median for the best search speeds (as
 * this will give us trees that are almost binary), but we want to maximize
 * variance for accuracy-optimization (e.g. if we have two features
 * A = [4, 4, 4, 6, 6, 6] and B = [0, 0, 0, 10, 10, 10], then we want to choose
 * B so that noisy data is less likely to fall on the wrong side of the split).
 *
 * These two goals can conflict, so right now we just use a simple split
 * function that splits closest to the median. This has the added advantage that
 * you don't need to normalize features to have similar distributions.
 *
 * We have another split function that takes variance into account, but this is
 * currently unused.
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "rbf.h"
#include "_rbf_train.h"



void print_array(int32_t *arr, size_t count) {
    for (size_t i = 0; i < count; i++) {
        printf("%d ", arr[i]);
    }
}

// Call when *alloc returns null
static void die_alloc_err(char *func_name, char *vars) {
    fprintf(stderr, "fatal error: function %s, allocating memory for %s\n", func_name, vars);
    exit(EXIT_FAILURE);
}

/*
 * Convert a feature column, e.g. [0, 1, 1, 1, 2, 2] into bins, e.g. [1 (for 0), 3 (for 1), 2 (for 2)].
 * Since our features are integers in the range [0, 255], statistics will be faster this way.
 * Returns: for feature `feature_num`:
 * - the frequency of each integer value in [0, 255]
 * - the sum of all feature values (i.e. the weighted sum over the frequency array)
 */
void feature_column_to_bins(rownum_type *row_index, feature_type feat_array[],
       colnum_type feat_num, rownum_type num_rows, rownum_type index_start, rownum_type index_end,
       // returns:
       stats_type *ret_counts, stats_type *ret_weighted_total) {
    // get frequencies:
    for (rownum_type rownum = index_start; rownum < index_end; rownum++) {
        feature_type feat_val = feat_array[num_rows * feat_num + row_index[rownum]];
        ret_counts[(size_t) feat_val] += 1;
        ret_weighted_total[0] += (stats_type) feat_val;
    }
    return;
    /*
	 * Is it faster to calculate weighted_total with additions inside the loop here,
	 * or with 256 multiplications and additions on the counts list later?
	 * Notes:
	 * - "n-4" below is because when we get down close to the leaves we don't do this any more.
	 * - This is all assuming it's a binary tree, which is obviously very approximate.
	 * Calculations:
	 * - additions inside loop:
	 *   \\sum_{k=0}^{n-4} num_nodes x num_additions = \\sum_{k=0}^{n-4} 2^k 2^{n-k} = (n-3) * 2^n
	 * - 2 x 256 = 2^9 multiplications and additions on the counts list later:
	 *   \\sum_{k=0}^{n-4} num_nodes x 2 x 256 = \\sum_{k=0}^{n-4} 2^k 2^9 = 2^9 * (2^(n-3) - 1) = 2^6 2^n - 2^9
	 * For our datasets n is 25-30, so for the full tree it's roughly a wash, maybe slightly faster
	 * to do them inside the above loop.
     */
}


// Select a random subset of features and get the frequencies for those features.
// Ugly to do two things here but ends up cleaner from a memory-management perspective.

colnum_type pos = -1;
colnum_type get_random_feature(colnum_type num_features) {
    //pos = ((pos + 1) % num_features);
    //return pos;
    return (colnum_type) (rand() % num_features);
}

void select_random_features_and_get_frequencies(rownum_type *row_index, feature_type *feat_array, bool *feats_already_selected,
        RbfConfig *cfg, rownum_type index_start, rownum_type index_end,
        // returns:
        colnum_type *ret_feat_subset, stats_type *ret_feat_freqs, stats_type *ret_feat_weighted_totals) {
    if (!feats_already_selected) {
        die_alloc_err("select_random_features_and_get_frequencies", "feats_already_selected");
    }
    for (colnum_type i = 0; i < cfg->num_features_to_compare; i++) {
        colnum_type feat_num = (colnum_type) (get_random_feature(cfg->num_features));
        while (feats_already_selected[(size_t) feat_num]) {
            feat_num = get_random_feature(cfg->num_features);
        }
        feats_already_selected[feat_num] = true;
        ret_feat_subset[i] = feat_num;
        feature_column_to_bins(row_index, feat_array,
                               feat_num, cfg->num_rows, index_start, index_end,
                               &(ret_feat_freqs[i * NUM_CHARS]), &(ret_feat_weighted_totals[i]));
    }
}


/*
 * Split a set of rows on one feature, trying to get close to the median but also maximizing
 * variance.
 *
 * NOTE: We're no longer using the variance but I'm leaving all this (code and comments) unchanged.
 * We'll make it an option later. For our current use our features are sufficiently skewed that
 * using variance is unhelpful, so we simply find the split closest to the median. So we're using
 * `get_simple_best_feature` instead of `get_best_feature`.
 *
 * We want something as close to the median as possible so as to make the tree more balanced.
 * And we want to calculate the "variance" about this split to compare features.
 *
 * CLEVERNESS ALERT (violating the "don't be clever" rule for speed):
 * Except we'll actually use the mean absolute deviation instead of the variance as it's easier and
 * better, esp since we're thinking of this in terms of Manhattan distance anyway. In fact, for our
 * purposes it suffices to calculate the *total* absolute deviation, i.e. the total moment: we don't
 * really need the mean since the denominator, the number of rows, is the same for all features that
 * we're going to compare.
 *
 * The total moment to the right of some b, say for example b = 7.5, is
 *     \\sum_{i=8}^{255} (i-7.5) * x_i = [ \\sum_{i=0}^255 (i-7.5) * x_i ] - [ \\sum_{i=0}^7 (i-7.5) x_i ]
 * That second term is actually just -(the moment to the left of b), so the total moment
 * (i.e. left + right) simplifies down to
 *     \\sum_{i=0}^255 i x_i - \\sum_{i=0}^255 7.5 x_i + 2 \\sum_{i=0}^7 7.5 x_i - 2 \\sum_{i=0}^7 i x_i
 * So we only need to track the running left-count and the running left-moment (w.r.t. 0), and then
 * we can calculate the total moment w.r.t. median when we're done.
 *
 * Summary: Starting at 0.5 (no use starting at 0), iterate (a) adding to simple count, and (b)
 * adding to left-side total moment. Stop as soon as the count is greater than half the total number
 * of rows, and at that point we have a single expression for the total moment.
 */
void split_one_feature(stats_type *feat_bins, stats_type total_zero_moment, stats_type count,
        // returns:
        double *total_moment, size_t *pos, stats_type *left_count) {
    *pos = 0;
    *left_count = feat_bins[0];
    stats_type fifty_percentile = count / 2;
    stats_type left_zero_moment = 0, this_item_moment;
    stats_type this_item_count;
    while (*left_count <= fifty_percentile) {
        *pos += 1;
        this_item_count = feat_bins[*pos];
        this_item_moment = this_item_count * *pos;
        *left_count += this_item_count;
        left_zero_moment += this_item_moment;
    }
    double real_pos = (double) *pos + 0.5;  // want moment about e.g. 7.5, not 7 (using numbers in example above)
                                            // See moment computation example in comment above
    *total_moment = (double) total_zero_moment - (real_pos * count) + (2 * ((real_pos * *left_count) - left_zero_moment));
    return;
}


// From the given features find the one which splits closest to the median.
void get_simple_best_feature(stats_type *feat_freqs,
        colnum_type num_feats_to_compare, stats_type *feat_weighted_totals, stats_type total_count,
        // returns:
        colnum_type *best_feat_num, feature_type *best_feat_split_val) {
    *best_feat_num = (colnum_type) 0;
    *best_feat_split_val = (feature_type) 0;
    stats_type min_split_balance = total_count, split_balance;
    stats_type left_count;
    size_t split_val;
    double ignore = 0.0;
    for (colnum_type i = 0; i < num_feats_to_compare; i++) {
        split_one_feature(&(feat_freqs[i * NUM_CHARS]), feat_weighted_totals[i], total_count,
                &ignore, &split_val, &left_count);
        split_balance = abs(left_count - (total_count - left_count));   // left_count - right_count
        if (split_balance < min_split_balance) {
            min_split_balance = split_balance;
            *best_feat_split_val = split_val;
            *best_feat_num = i;
        }
    }
    return;
}


// quicksort-type partitioning of row_index[index_start..index_end] based on whether the
// feature `feat_num` is less-than or greater-than-or-equal-to split_value
rownum_type quick_partition(rownum_type *row_index, feature_type *feat_array,
        rownum_type num_rows, rownum_type index_start, rownum_type index_end, colnum_type feat_num, feature_type split_val) {
    if (index_end <= index_start) {
        return index_start;
    }

    rownum_type i = index_start, j = index_end - 1;
    while (i < j) {
        while ((i < index_end) && (feat_array[num_rows * feat_num + row_index[i]] <= split_val)) {
            i += 1;
        }
        while ((j >= index_start) && (feat_array[num_rows * feat_num + row_index[j]] > split_val) && (j > 0)) {
            j -= 1;
        }
        if (i >= j) {
            return i;
        }
        rownum_type tmp = row_index[i];
        row_index[i] = row_index[j];
        row_index[j] = tmp;
    }
    return index_start;  // should never get here
}


// Get a random subset of features, find the best one of those features,
// and split this set of nodes on that feature.
static void _split_node(rownum_type *row_index, feature_type *feat_array, RbfConfig *cfg,
        rownum_type index_start, rownum_type index_end,
        // returns:
        colnum_type *best_feat_num, feature_type *best_feat_split_val, rownum_type *split_pos) {
    bool *feats_already_selected = (bool *) calloc(sizeof(bool), cfg->num_features);
    *split_pos = index_start;
    for (int attempt_num = 0; (attempt_num < 3) && ((*split_pos == index_start) || (*split_pos == index_end)); attempt_num++) {
        colnum_type *feat_subset = (colnum_type *) calloc(sizeof(colnum_type), cfg->num_features_to_compare);
        stats_type *feat_freqs = (stats_type *) calloc(sizeof(stats_type), cfg->num_features_to_compare * NUM_CHARS);
        stats_type *weighted_totals = (stats_type *) calloc(sizeof(stats_type), cfg->num_features_to_compare);
        if (!feat_subset || !feat_freqs || !weighted_totals) {
            die_alloc_err("_split_node", "feat_subset || feat_freqs || weighted_totals");
        }
        colnum_type _best_feat_index;

        select_random_features_and_get_frequencies(row_index, feat_array, feats_already_selected,
                cfg, index_start, index_end,
                feat_subset, feat_freqs, weighted_totals);
        get_simple_best_feature(feat_freqs, cfg->num_features_to_compare, weighted_totals, index_end - index_start,
                &_best_feat_index, best_feat_split_val);
        *best_feat_num = feat_subset[_best_feat_index];
        // return values:
        *split_pos = quick_partition(row_index, feat_array, cfg->num_rows, index_start, index_end, *best_feat_num, *best_feat_split_val);
        free(feat_subset);
        free(feat_freqs);
        free(weighted_totals);
    }
    free(feats_already_selected);
    return;
}


/*
 * Calculate the split (or leaf) at one node (and its descendants).
 * So this is doing all the real work of building the tree.
 * Params:
 * - tree we're building
 * - feature array
 * - leaf size, total number of features, and number of features to compare
 *   (not adding these to the tree struct b/c they're only needed at training time)
 * - num_rows: number of rows in the feature-array and in the tree's row_index
 * - index_start and index_end: the view into row_index that we're considering right now
 * - tree_array_pos: the position of this node in the tree arrays
 * - TODO: REMOVE depth of this node in the tree
 * Guarantees:
 * - Parallel calls to `calculate_one_node` will look at non-intersecting views.
 * - Child calls will look at distinct sub-views of this view.
 * - No two calls to `calculate_one_node` will have the same tree_array_pos
 */
static void calculate_one_node(RandomBinaryTree *tree, feature_type *feat_array, RbfConfig *config,
        rownum_type index_start, rownum_type index_end, treeindex_type tree_array_pos, size_t depth) {
    if (2 * tree_array_pos + 2 >= tree->tree_size) {
    // Special termination condition to regulate depth.
        tree->tree_first[tree_array_pos] = (rownum_type) (HIGH_BIT_1 ^ index_start);
        tree->tree_second[tree_array_pos] = (rownum_type) (HIGH_BIT_1 ^ index_end);
        tree->num_leaves += 1;
// fmt.Fprintf(tree_statsFile, "%d,%d,depth-based-leaf,%d,%d,%d,%d,%d,%d,\n", tree_array_pos, depth, index_start, index_end, index_end-index_start, 0, 0, 0)
        return;
    }

    if (index_end - index_start < config->leaf_size) {
    // Not enough items left to split. Make a leaf.
        tree->tree_first[tree_array_pos] = (rownum_type) (HIGH_BIT_1 ^ index_start);
        tree->tree_second[tree_array_pos] = (rownum_type) (HIGH_BIT_1 ^ index_end);
        tree->num_leaves += 1;
// fmt.Fprintf(tree_statsFile, "%d,%d,size-based-leaf,%d,%d,%d,%d,%d,%d,\n", tree_array_pos, depth, index_start, index_end, index_end-index_start, 0, 0, 0)
    } else {
    // Not a leaf. Get a random subset of num_features_to_compare features, find the best one, and split this node.
        colnum_type best_feat_num;
        feature_type best_feat_split_val;
        rownum_type index_split;
        _split_node(tree->row_index, feat_array, config, index_start, index_end,
                    &best_feat_num, &best_feat_split_val, &index_split);

        tree->tree_first[tree_array_pos] = best_feat_num;
        tree->tree_second[tree_array_pos] = (rownum_type) best_feat_split_val;
// fmt.Fprintf(tree_statsFile, "%d,%d,internal,%d,%d,%d,%d,%d,%d,%s\n", tree_array_pos, depth, index_start, index_end,
//        index_end - index_start, index_split, featureNum, featureSplitValue, features.CHAR_REVERSE_MAP[featureNum])
        tree->num_internal_nodes += 1;
        calculate_one_node(tree, feat_array, config, index_start, index_split, (2*tree_array_pos)+1, depth+1);
        calculate_one_node(tree, feat_array, config, index_split, index_end, (2*tree_array_pos)+2, depth+1);
    }
}


static RandomBinaryTree *create_rbt(RbfConfig *config) {
    treeindex_type tree_size = (treeindex_type) (1 << config->tree_depth);
    RandomBinaryTree *tree = (RandomBinaryTree *) malloc(sizeof(RandomBinaryTree));
    if (!tree) {
        die_alloc_err("create_rbt", "tree");
    }
    tree->row_index = (rownum_type *) malloc(sizeof(rownum_type) * config->num_rows);
    tree->tree_first = (rownum_type *) calloc(sizeof(rownum_type), (size_t) tree_size);
    tree->tree_second = (rownum_type *) calloc(sizeof(rownum_type), (size_t) tree_size);
    if (!(tree->row_index) || !(tree->tree_first) || !(tree->tree_second)) {
        die_alloc_err("create_rbt", "tree attributes");
    }

    for (rownum_type i = 0; i < config->num_rows; i++) {
        tree->row_index[i] = i;
    }
    tree->num_rows = config->num_rows;
    tree->tree_size = tree_size;
    tree->num_internal_nodes = 0;
    tree->num_leaves = 0;

    return tree;
}


void print_time(char *msg) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    double count;
    count = t.tv_sec * 1e9;
    count = (count + t.tv_nsec) * 1e-9;
    printf("%f: %s\n", count, msg);
}


static RandomBinaryTree *train_one_tree(feature_type *feat_array, RbfConfig *config) {
    RandomBinaryTree *tree = create_rbt(config);
    calculate_one_node(tree, feat_array, config, 0, config->num_rows, 0, 0);
    return tree;
}


feature_type *transpose(feature_type *input, size_t rows, size_t cols) {
    feature_type *output = (feature_type *) malloc(sizeof(feature_type) * rows * cols);
    if (!output) {
        die_alloc_err("transpose", "output");
    }
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
    return output;
}

RandomBinaryForest *train_forest(feature_type *feat_array, RbfConfig *config) {
    srand(2719);
    print_time("start training");
    RandomBinaryForest *forest = (RandomBinaryForest *) malloc(sizeof(RandomBinaryForest));
    if (!forest) {
        die_alloc_err("train_forest", "forest");
    }
    forest->config = config;
    forest->trees = (RandomBinaryTree *) malloc(sizeof(RandomBinaryTree) * config->num_trees);
    #pragma omp parallel for
    for (size_t i = 0; i < config->num_trees; i++) {
        forest->trees[i] = *train_one_tree(feat_array, config);
    }
    print_time("finish training");
//printf("trees[0].treeFirst: ");
//print_array(forest->trees[0].tree_first, 1 << config->tree_depth);
//printf("\n");
//printf("trees[0].treeSecond: ");
//print_array(forest->trees[0].tree_second, 1 << config->tree_depth);
//printf("\n");
    return forest;
}
