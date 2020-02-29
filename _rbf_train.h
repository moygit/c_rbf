void get_feature_frequencies(uint *local_row_index, char local_feature_array[],
       int feature_num, int num_features, int index_start, int index_end,
       // returns:
       uint *ret_counts, uint *ret_weighted_total);

bool test_get_feature_frequencies();
