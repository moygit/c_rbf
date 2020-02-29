#include "rbf.h"
#include "_rbf_train.h"

bool test() {
    return 1;
}


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


bool test_get_feature_frequencies() {
    // given:
    uint local_row_index[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    feature_t local_feature_array[] = {0, 0, 5, 5, 5, 5, 7, 7, 7, 7};
    colnum_t feature_num = 0, num_features = 1;
    rownum_t index_start = 0, index_end = 10;
    // when:
    stats_t *counts = (stats_t *) calloc(sizeof(stats_t), NUM_CHARS);
    stats_t weighted_total = 0;
    get_feature_frequencies(local_row_index, local_feature_array, feature_num, num_features, index_start, index_end, counts, &weighted_total);
    // then:
    stats_t expected_counts[8] = {2, 0, 0, 0, 0, 4, 0, 4};
    return (weighted_total == 48)
            &&  _test_array_equals(counts, 8, expected_counts, 8)
            &&  _test_array_seg_eq_val(counts, 8, NUM_CHARS, 0);
}


bool test_select_random_features_and_get_frequencies() {
    // given:
    rownum_t row_index[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    feature_t feature_array[10] = {0, 0, 5, 5, 5, 5, 7, 7, 7, 7};
    colnum_t num_features = 1, num_features_to_compare = 1;
    // when:
    colnum_t *feature_subset = (colnum_t *) malloc(sizeof(colnum_t) * 1);
    stats_t *counts = (stats_t *) calloc(sizeof(stats_t), 1 * NUM_CHARS);
    stats_t *weighted_totals = (stats_t *) calloc(sizeof(stats_t), 1);
    select_random_features_and_get_frequencies(row_index, feature_array, 1, 1, 0, 10, feature_subset, counts, weighted_totals);
    // then:
    stats_t expected_counts[8] = {2, 0, 0, 0, 0, 4, 0, 4};
    return (weighted_totals[0] == 48)
            &&  _test_array_equals(counts, 8, expected_counts, 8)
            &&  _test_array_seg_eq_val(counts, 8, NUM_CHARS, 0);
}


void _get_moment_and_count(stats_t lis[], size_t len_lis,
        // ret vals:
        stats_t *moment, stats_t *count) {
    *moment = 0;
    *count = 0;
    for (size_t i = 0; i < len_lis; i++) {
        *count += lis[i];
        *moment += (i * lis[i]);
    }
    return;
}


bool test_split_one_feature() {
    stats_t moment1, moment2, moment3, count1, count2, count3;
    double total_moment1, total_moment2, total_moment3;
    size_t pos1, pos2, pos3;
    stats_t left_count1, left_count2, left_count3;
    // given:
    stats_t L1[8] = {10, 5, 4, 0, 0, 11, 12, 13};
    _get_moment_and_count(L1, 8, &moment1, &count1);      // 231, 55
    stats_t L2[5] = {10, 0, 0, 0, 0};
    _get_moment_and_count(L2, 5, &moment2, &count2);      // 0, 10
    stats_t L3[5] = {1, 1, 1, 1, 1};
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
    stats_t feature_frequencies[10] = {1, 1, 1, 1, 1,  // first row of feature-frequencies
                                       5, 0, 0, 0, 0}; // second row of feature-frequencies
    stats_t weighted_totals[2] = {10, 0};              // 10 == (1 * 0) + (1 * 1) + (1 * 2) + (1 * 3) + (1 * 4) + (1 * 5)
                                                       //  0 == (5 * 0) + (0 * 1) + ... + (0 * 4)
    stats_t total_count = 5;
    // when:
    colnum_t best_feature_num;
    feature_t best_feature_split_value;
    get_simple_best_feature(feature_frequencies, 2, weighted_totals, total_count, &best_feature_num, &best_feature_split_value);
    // then f0 is "better" than f1 because its split is closer to the median
    colnum_t exp_best_feature_num = 0;
    feature_t exp_split_value = 2;
    return (best_feature_num == exp_best_feature_num)
            && (best_feature_split_value == exp_split_value);
}


bool _test_qp(rownum_t *row_index, size_t ri_size, feature_t *feature_array, feature_t split_value,
        rownum_t *exp_row_index, size_t exp_ri_size, feature_t exp_split, rownum_t index_end) {
    colnum_t num_cols = 1, feature_num = 0;
    rownum_t index_start = 0;
    rownum_t split = quick_partition(row_index, feature_array, num_cols, index_start, index_end, feature_num, split_value);
    return(_test_array_equals(row_index, ri_size, exp_row_index, exp_ri_size) && (split == exp_split));
}

bool test_quick_partition() {
    // setup boilerplate (only 1 feature, we'll sort a 6-length array):
    rownum_t index_end = 6;

    // given a reverse-sorted list:
    rownum_t local_row_index_1[6] = {0, 1, 2, 3, 4, 5};
    feature_t local_feature_array_1[6] = {15, 14, 13, 12, 11, 10};
    // when we split at 12.5 (the median):
    feature_t split_value = 12;
    // then we split at position 3 to get feature order [10, 11, 12   |   13, 14, 15]
    feature_t exp_split = 3;
    rownum_t exp_row_index_1[6] = {5, 4, 3, 2, 1, 0};
    bool test1_passed = _test_qp(local_row_index_1, 6, local_feature_array_1, split_value, exp_row_index_1, 6, exp_split, index_end);
  
    // given a particular (random) order:
    rownum_t local_row_index_2[6] = {0, 1, 2, 3, 4, 5};
    feature_t local_feature_array_2[6] = {11, 10, 14, 12, 15, 13};
    // when we split at 12.5 (the median):
    split_value = 12;
    // then we split at position 3 to get feature order [11, 10, 12   |   14, 15, 13]
    exp_split = 3;
    rownum_t exp_row_index_2[6] = {0, 1, 3, 2, 4, 5};
    bool test2_passed = _test_qp(local_row_index_2, 6, local_feature_array_2, split_value, exp_row_index_2, 6, exp_split, index_end);
  
    // given a particular (random) order:
    rownum_t local_row_index_3[6] = {0, 1, 2, 3, 4, 5};
    feature_t local_feature_array_3[6] = {11, 10, 14, 12, 15, 13};
    // when split-value is less than all the values in the array
    split_value = 2;
    // then there's no split
    exp_split = 0;
    rownum_t exp_row_index_3[6] = {0, 1, 2, 3, 4, 5};   // and nothing gets moved
    bool test3_passed = _test_qp(local_row_index_3, 6, local_feature_array_3, split_value, exp_row_index_3, 6, exp_split, index_end);

    // given empty lists of rows and features:
    rownum_t local_row_index_4[0] = {};
    feature_t local_feature_array_4[0] = {};
    // "Dude, there's nothing to split here!"
    exp_split = 0;
    rownum_t exp_row_index_4[0] = {};
    bool test4_passed = _test_qp(local_row_index_4, 0, local_feature_array_4, split_value, exp_row_index_4, 0, exp_split, 0);

    return test1_passed && test2_passed && test3_passed && test4_passed;
}
