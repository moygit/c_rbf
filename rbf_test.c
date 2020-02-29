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
    char local_feature_array[] = {0, 0, 5, 5, 5, 5, 7, 7, 7, 7};
    int feature_num = 0, num_features = 1, index_start = 0, index_end = 10;
    // when:
    uint *counts = (uint *) calloc(sizeof(uint), NUM_CHARS);
    uint weighted_total = 0;
    get_feature_frequencies(local_row_index, local_feature_array, feature_num, num_features, index_start, index_end, counts, &weighted_total);
    // then:
    uint expected_counts[8] = {2, 0, 0, 0, 0, 4, 0, 4};
    return (weighted_total == 48)
            &&  _test_array_equals(counts, 8, expected_counts, 8)
            &&  _test_array_seg_eq_val(counts, 8, NUM_CHARS, 0);
}
