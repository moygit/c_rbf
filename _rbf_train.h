void get_feature_frequencies(rownum_type *local_row_index, feature_type local_feature_array[],
       colnum_type feature_num, colnum_type num_features, rownum_type index_start, rownum_type index_end,
       // returns:
       stats_type *ret_counts, stats_type *ret_weighted_total);

void select_random_features_and_get_frequencies(rownum_type *row_index, feature_type *feature_array,
        RbfConfig *cfg, rownum_type index_start, rownum_type index_end,
        // returns:
        colnum_type *ret_feature_subset, stats_type *ret_feature_frequencies, stats_type *ret_feature_weighted_totals);

void split_one_feature(stats_type *feature_bins, stats_type total_zero_moment, stats_type count,
        // returns:
        double *total_moment, size_t *pos, stats_type *left_count);

void get_simple_best_feature(stats_type *feature_frequencies,
        colnum_type num_features_to_compare, stats_type *feature_weighted_totals, stats_type total_count,
        // returns:
        colnum_type *best_feature_num, feature_type *best_feature_split_value);

rownum_type quick_partition(rownum_type *local_row_index, feature_type *local_feature_array,
        colnum_type num_features, rownum_type index_start, rownum_type index_end, colnum_type feature_num, feature_type split_value);

feature_type *transpose(feature_type *input, size_t rows, size_t cols);

bool test_get_feature_frequencies();
bool test_select_random_features_and_get_frequencies();
bool test_split_one_feature();
bool test_get_simple_best_feature();
bool test_quick_partition();
bool test_transpose();
void print_time(char *msg);
