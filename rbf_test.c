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
