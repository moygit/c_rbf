#include <stdio.h>
#include "rbf.h"
#include "_rbf_train.h"
#include "_rbf_query.h"


bool _test_array_seg_eq_val(uint arr1[], size_t start, size_t end, uint val) {
    for (size_t i = start; i < end; i++) {
        if (arr1[i] != val) {
            return 0;
        }
    }
    return 1;
}

bool _test_array_equals(uint arr1[], size_t arr1_size, uint arr2[], size_t arr2_size) {
    if (arr1_size != arr2_size) {
        return 0;
    }
    for (size_t i = 0; i < arr1_size; i++) {
        if (arr1[i] != arr2[i]) {
            return 0;
        }
    }
    return 1;
}


bool test_feature_column_to_bins() {
    // given:
    uint row_index[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    feature_type feature_array[] = {0, 0, 5, 5, 5, 5, 7, 7, 7, 7,   // 0th feature
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1};  // 1st feature
    colnum_type feature_num = 0;
    rownum_type index_start = 0, index_end = 10, num_rows = 10;
    // when:
    stats_type *counts = (stats_type *) calloc(sizeof(stats_type), NUM_CHARS);
    stats_type weighted_total = 0;
    feature_column_to_bins(row_index, feature_array, feature_num, num_rows, index_start, index_end, counts, &weighted_total);
    // then:
    stats_type expected_counts[8] = {2, 0, 0, 0, 0, 4, 0, 4};
    return (weighted_total == 48)
            &&  _test_array_equals(counts, 8, expected_counts, 8)
            &&  _test_array_seg_eq_val(counts, 8, NUM_CHARS, 0);
}


bool test_select_random_features_and_get_frequencies() {
    // given:
    srand(2); // ensures we select feature 0
    bool selected[2] = {false, false};
    rownum_type row_index[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    feature_type feature_array[20] = {0, 0, 5, 5, 5, 5, 7, 7, 7, 7,
                                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    colnum_type num_features = 2, num_features_to_compare = 1;
    RbfConfig config = {0, 0, 0, 10, num_features, num_features_to_compare};
    // when:
    colnum_type feature_subset;
    stats_type *counts = (stats_type *) calloc(sizeof(stats_type), 1 * NUM_CHARS);
    stats_type weighted_total = 0;
    select_random_features_and_get_frequencies(row_index, feature_array, selected, &config, 0, 10, &feature_subset, counts, &weighted_total);
    // then:
    stats_type expected_counts[8] = {2, 0, 0, 0, 0, 4, 0, 4};
    bool test1_passed = (feature_subset == 0)
                         &&  (weighted_total == 48)
                         &&  _test_array_equals(counts, 8, expected_counts, 8)
                         &&  _test_array_seg_eq_val(counts, 8, NUM_CHARS, 0);

    srand(1); // ensures we select feature 1
    bool selected_2[2] = {false, false};
    weighted_total = 0;
    select_random_features_and_get_frequencies(row_index, feature_array, selected_2, &config, 0, 10, &feature_subset, counts, &weighted_total);
    bool test2_passed = (feature_subset == 1) && (weighted_total == 10) && (counts[1] == 10);
    return test1_passed && test2_passed;
}


void _get_moment_and_count(stats_type lis[], size_t len_lis,
        // ret vals:
        stats_type *moment, stats_type *count) {
    *moment = 0;
    *count = 0;
    for (size_t i = 0; i < len_lis; i++) {
        *count += lis[i];
        *moment += (i * lis[i]);
    }
    return;
}


bool test_split_one_feature() {
    stats_type moment1, moment2, moment3, count1, count2, count3;
    double total_moment1, total_moment2, total_moment3;
    size_t pos1, pos2, pos3;
    stats_type left_count1, left_count2, left_count3;
    // given:
    stats_type L1[8] = {10, 5, 4, 0, 0, 11, 12, 13};
    _get_moment_and_count(L1, 8, &moment1, &count1);      // 231, 55
    stats_type L2[5] = {10, 0, 0, 0, 0};
    _get_moment_and_count(L2, 5, &moment2, &count2);      // 0, 10
    stats_type L3[5] = {1, 1, 1, 1, 1};
    _get_moment_and_count(L3, 5, &moment3, &count3);      // 10, 5
    // when:
    split_one_feature(L1, moment1, count1, &total_moment1, &pos1, &left_count1);
    split_one_feature(L2, moment2, count2, &total_moment2, &pos2, &left_count2);
    split_one_feature(L3, moment3, count3, &total_moment3, &pos3, &left_count3);
    // then:
    return (total_moment1 == 122.5) && (pos1 == 5) && (left_count1 == 30)
            &&  (total_moment2 == 5.0) && (pos2 == 0) && (left_count2 == 10)
            &&  (total_moment3 == 6.5) && (pos3 == 2) && (left_count3 == 3);
}


bool test_get_simple_best_feature() {
    // given:
    stats_type feature_frequencies[10] = {1, 1, 1, 1, 1,  // first row of feature-frequencies
                                          5, 0, 0, 0, 0}; // second row of feature-frequencies
    stats_type weighted_totals[2] = {10, 0};              // 10 == (1 * 0) + (1 * 1) + (1 * 2) + (1 * 3) + (1 * 4) + (1 * 5)
                                                          //  0 == (5 * 0) + (0 * 1) + ... + (0 * 4)
    stats_type total_count = 5;
    // when:
    colnum_type best_feature_num;
    feature_type best_feature_split_value;
    get_simple_best_feature(feature_frequencies, 2, weighted_totals, total_count, &best_feature_num, &best_feature_split_value);
    // then f0 is "better" than f1 because its split is closer to the median
    colnum_type exp_best_feature_num = 0;
    feature_type exp_split_value = 2;
    return (best_feature_num == exp_best_feature_num)
            && (best_feature_split_value == exp_split_value);
}


bool _test_qp(rownum_type *row_index, size_t ri_size, feature_type *feature_array, feature_type split_value,
        rownum_type *exp_row_index, size_t exp_ri_size, feature_type exp_split, rownum_type index_end) {
    colnum_type feature_num = 0;
    rownum_type index_start = 0, num_rows = index_end;
    rownum_type split = quick_partition(row_index, feature_array, num_rows, index_start, index_end, feature_num, split_value);
    return(_test_array_equals(row_index, ri_size, exp_row_index, exp_ri_size) && (split == exp_split));
}

bool test_quick_partition() {
    // setup boilerplate (only 1 feature, we'll sort a 6-length array):
    rownum_type index_end = 6;

    // given a reverse-sorted list:
    rownum_type row_index_1[6] = {0, 1, 2, 3, 4, 5};
    feature_type feature_array_1[12] = {15, 14, 13, 12, 11, 10,  // selected feature
                                         1,  1,  1,  1,  1,  1}; // ignored feature
    // when we split at 12.5 (the median):
    feature_type split_value = 12;
    // then we split at position 3 to get feature order [10, 11, 12   |   13, 14, 15]
    feature_type exp_split = 3;
    rownum_type exp_row_index_1[6] = {5, 4, 3, 2, 1, 0};
    bool test1_passed = _test_qp(row_index_1, 6, feature_array_1, split_value, exp_row_index_1, 6, exp_split, index_end);
  
    // given a particular (random) order:
    rownum_type row_index_2[6] = {0, 1, 2, 3, 4, 5};
    feature_type feature_array_2[6] = {11, 10, 14, 12, 15, 13};
    // when we split at 12.5 (the median):
    split_value = 12;
    // then we split at position 3 to get feature order [11, 10, 12   |   14, 15, 13]
    exp_split = 3;
    rownum_type exp_row_index_2[6] = {0, 1, 3, 2, 4, 5};
    bool test2_passed = _test_qp(row_index_2, 6, feature_array_2, split_value, exp_row_index_2, 6, exp_split, index_end);
  
    // given a particular (random) order:
    rownum_type row_index_3[6] = {0, 1, 2, 3, 4, 5};
    feature_type feature_array_3[6] = {11, 10, 14, 12, 15, 13};
    // when split-value is less than all the values in the array
    split_value = 2;
    // then there's no split
    exp_split = 0;
    rownum_type exp_row_index_3[6] = {0, 1, 2, 3, 4, 5};   // and nothing gets moved
    bool test3_passed = _test_qp(row_index_3, 6, feature_array_3, split_value, exp_row_index_3, 6, exp_split, index_end);

    // given empty lists of rows and features:
    rownum_type row_index_4[0] = {};
    feature_type feature_array_4[0] = {};
    // "Dude, there's nothing to split here!"
    exp_split = 0;
    rownum_type exp_row_index_4[0] = {};
    bool test4_passed = _test_qp(row_index_4, 0, feature_array_4, split_value, exp_row_index_4, 0, exp_split, 0);

    return test1_passed && test2_passed && test3_passed && test4_passed;
}


bool test_transpose() {
    feature_type input[] = {1, 2, 3,  // matrix with 2 rows, 3 cols
                            4, 5, 6};
    feature_type *output = transpose(input, 2, 3);
    feature_type exp_transpose[] = {1, 4,
                                    2, 5,
                                    3, 6};
    bool compare = 1;
    for (size_t i = 0; i < 6; i++) {
        compare = compare && (output[i] == exp_transpose[i]);
    }
    return compare;
}

bool test_query() {
    // given:

    // Create two identical dummy trees for testing.
    //
    // The test tree looks like it was "trained" on 5-skip-bigrams of the strings "aaaa" and "abc"
    // (so "aaaa"'s 0th ("aa") entry is 6 whereas "abc"'s 0th entry is 0).
    //   root node:
    //     tree_first[0]: i.e. split on the 0 feature (i.e. "aa")
    //     tree_second[0]: split-value 1
    //   left child:
    //     tree_first[1]: (leaf) 1 ("abc")  (actually HIGH_BIT_1 ^ 1)
    //     tree_second[1]: (leaf) 2         (actualy HIGH_BIT_1 ^ 2)
    //   right child:
    //     tree_first[2]: (leaf) 0 ("aaaa") (actually HIGH_BIT_1 ^ 0)
    //     tree_second[2]: (leaf) 1         (actually HIGH_BIT_1 ^ 1)

    rownum_type row_index[] = {0, 1};
    int num_rows = 2;
    rownum_type tree_first[] = {0, HIGH_BIT_1 ^ 1, HIGH_BIT_1 ^ 0};
    rownum_type tree_second[] = {1, HIGH_BIT_1 ^ 2, HIGH_BIT_1 ^ 1};
    int tree_size = 3;
    RandomBinaryTree tree = {row_index, num_rows, tree_first, tree_second, tree_size,
                             0, 0}; // don't care about these last two
    RandomBinaryTree trees[] = {tree, tree};
    int num_trees = 2;
    int num_features = 1;
    RbfConfig config = {num_trees, 0, 0, 0, num_features, num_features};
    RandomBinaryForest forest = {&config, trees};

    feature_type point[] = {6};
    feature_type two_points[] = {6, 0}; // 1st point has one feature, 6 (for "aaaa"), 2nd point has one feature, 0 (for "abc")
    size_t num_points = 2;

    // when
    RbfResults *results = query_forest_all_results(&forest, point, num_features);
    RbfResults *batch_results = batch_query_forest_all_results(&forest, two_points, num_features, num_points);
    size_t count, **batch_counts;
    rownum_type *deduped_results = query_forest_dedup_results(&forest, point, num_features, &count);
    rownum_type **batch_deduped_results = batch_query_forest_dedup_results(&forest, two_points, num_features, num_points, batch_counts);

    // then
    bool all_result = (results->tree_result_counts[0] == 1)        // Each tree returns exactly 1 result
                       && (results->tree_result_counts[1] == 1)
                       && (results->tree_results[0][0] == 0)       // and the result is "aaaa"
                       && (results->tree_results[1][0] == 0);
    bool dedup_result = (count == 1) && (deduped_results[0] == 0);
    bool batch_all_result = (batch_results[0].tree_result_counts[0] == 1)        // Each tree returns exactly 1 result
                             && (batch_results[0].tree_result_counts[1] == 1)
                             && (batch_results[0].tree_results[0][0] == 0)       // and the result is "aaaa"
                             && (batch_results[0].tree_results[1][0] == 0)
                             && (batch_results[1].tree_result_counts[0] == 1)    // Each tree returns exactly 1 result
                             && (batch_results[1].tree_result_counts[1] == 1)
                             && (batch_results[1].tree_results[0][0] == 1)       // and the result is "abc"
                             && (batch_results[1].tree_results[1][0] == 1);
    bool batch_dedup_result = ((*batch_counts)[0] == 1) && (batch_deduped_results[0][0] == 0)       // only one result, "aaaa"
                              && ((*batch_counts)[1] == 1) && (batch_deduped_results[1][0] == 1);   // only one result, "abcd"
    return all_result && dedup_result && batch_all_result && batch_dedup_result;
}

bool test_query_sorted() {
    // given:

    // Create two identical dummy trees for testing.
    // We're only testing the sorting here, so the goal is for the tree to return indices into
    // some array, and then we want to check that those get sorted by distance from the query point.
    //   root node:
    //     tree_first[0]: i.e. split on the 0 feature
    //     tree_second[0]: split-value 1
    //   left child:
    //     tree_first[1]: (leaf) 0          (act0ally HIGH_BIT_1 ^ 0)
    //     tree_second[1]: (leaf) 2         (actualy HIGH_BIT_1 ^ 2)
    //   right child:
    //     tree_first[2]: (leaf) 3          (actually HIGH_BIT_1 ^ 3)
    //     tree_second[2]: (leaf) 7         (actually HIGH_BIT_1 ^ 7)

    rownum_type row_index[] = {0, 1, 2, 3, 4, 5, 6};
    feature_type ref_points[] = {1, 2, 0, 9, 8, 6, 5};
                                       // ^^^^^^^^^^ for first point (6), want these guys, sorted in order of L2 distance from 6.
                                       // so we should get 5, 6, 4, 3 (indices of 6, 5, 8, 9)
                              // ^^^^ for second point (0), want these guys (1, 2), sorted in order of L2 distance from 0.
                              // so we should get 0, 1 (indices of 1, -2).
    int num_rows = 7;
    rownum_type tree_first[] = {0, HIGH_BIT_1 ^ 0, HIGH_BIT_1 ^ 3};
    rownum_type tree_second[] = {1, HIGH_BIT_1 ^ 2, HIGH_BIT_1 ^ 7};
    int tree_size = 3;
    RandomBinaryTree tree = {row_index, num_rows, tree_first, tree_second, tree_size,
                             0, 0}; // don't care about these last two
    RandomBinaryTree trees[] = {tree, tree};
    int num_trees = 2;
    int num_features = 1;
    RbfConfig config = {num_trees, 0, 0, 0, num_features, num_features};
    RandomBinaryForest forest = {&config, trees};

    feature_type point[] = {6};
    feature_type two_points[] = {6, 0}; // 1st point has one feature, 6, 2nd point has one feature, 0
    size_t num_points = 2;

    // when
    RbfResults *batch_results = batch_query_forest_all_results(&forest, two_points, num_features, num_points);
    size_t **counts;
    rownum_type **results = batch_query_forest_dedup_results_sorted(&forest, ref_points, two_points, num_features, num_points, l2_compare, counts);

    // then
    return ((*counts)[0] == 4)
            && (results[0][0] == 5)     // see comments in defintion
            && (results[0][1] == 6)     // of `ref_points` above
            && (results[0][2] == 4)
            && (results[0][3] == 3)
            && ((*counts)[1] == 2)
            && (results[1][0] == 0)
            && (results[1][1] == 1);
}
