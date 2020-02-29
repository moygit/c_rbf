void get_feature_frequencies(rownum_t *local_row_index, feature_t local_feature_array[],
       colnum_t feature_num, colnum_t num_features, rownum_t index_start, rownum_t index_end,
       // returns:
       stats_t *ret_counts, stats_t *ret_weighted_total);

void select_random_features_and_get_frequencies(rownum_t *row_index, feature_t *feature_array,
        colnum_t num_features, colnum_t num_features_to_compare, rownum_t index_start, rownum_t index_end,
        // returns:
        colnum_t *ret_feature_subset, stats_t *ret_feature_frequencies, stats_t *ret_feature_weighted_totals);

void split_one_feature(stats_t *feature_bins, stats_t total_zero_moment, stats_t count,
        // returns:
        double *total_moment, size_t *pos, stats_t *left_count);

bool test_get_feature_frequencies();
bool test_select_random_features_and_get_frequencies();
bool test_split_one_feature();
