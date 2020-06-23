#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "rbf.h"
#include "_rbf_train.h"
#include "_rbf_query.h"

// gcc -fopenmp -lapr-1 -Wall -o mnist mnist.c rbf_train.o rbf_query.o rbf_utils.o

#define NUM_LABELS 10


typedef feature_type label_type;


feature_type *read_file(char *filename, size_t *byte_count) {
    FILE *f = fopen(filename, "rb");
    int file_num = fileno(f);
    struct stat st;
    fstat(file_num, &st);
    off_t file_size = st.st_size;
    feature_type *file_data = malloc(sizeof(feature_type) * file_size);
    *byte_count = fread(file_data, 1, file_size, f);
    assert(*byte_count == (size_t) file_size);
    fclose(f);
    return file_data;
}


// Of the given array of labels (upto index `count`), which one has the max frequency?
label_type get_winner(label_type *labels, size_t count) {
    // get frequency of each label:
    int counts[NUM_LABELS] = {0};
    for (size_t i = 0; i < count; i++) {
        counts[labels[i]] += 1;
    }

    // get argmax:
    size_t max = 0;
    label_type max_index = 0;
    for (label_type i = 0; i < NUM_LABELS; i++) {
        if (counts[i] > max) {
            max = counts[i];
            max_index = i;
        }
    }
    return max_index;
}


/*
 * Given results from multiple trees, merge them into a single array.
 */
rownum_type *merge_indices(RbfResults *results, size_t num_trees) {
    rownum_type *indices = malloc(sizeof(rownum_type) * results->total_count);
    size_t all_indices_pos = 0;
    for (size_t tree_num = 0; tree_num < num_trees; tree_num++) {
        size_t this_tree_results_count = results->tree_result_counts[tree_num];
        for (size_t this_tree_result_i = 0; this_tree_result_i < this_tree_results_count; this_tree_result_i++) {
            indices[all_indices_pos] = results->tree_results[tree_num][this_tree_result_i];
            all_indices_pos += 1;
        }
    }
    return indices;
}


/*
 * Given an array of row-indices, return the corresponding labels.
 */
label_type *get_labels(rownum_type *indices, size_t count, label_type *train_labels) {
    label_type *labels = malloc(sizeof(label_type) * count);
    for (size_t i = 0; i < count; i++) {
        labels[i] = train_labels[indices[i]];
    }
    return labels;
}


/**
 * Given a forest and a set of test points, evaluate each point by plurality,
 * i.e. find the argmax label among all neighbors returned by all trees, with no deduping:
 * if a point is returned by n trees then it gets counted n times.
 */
void eval_plurality(RandomBinaryForest *forest, RbfConfig cfg, feature_type *test_data, label_type *train_labels,
        label_type *test_labels, size_t num_test_rows, size_t num_features) {
    print_time("started eval_plurality");
    RbfResults *rbf_results = batch_query_forest_all_results(forest, test_data, num_features, num_test_rows);

    int *matches = malloc(sizeof(int) * num_test_rows);
    #pragma omp parallel for
    for (size_t i = 0; i < num_test_rows; i++) {
        RbfResults results = rbf_results[i];
        rownum_type *merged_indices = merge_indices(&results, cfg.num_trees);
        label_type *labels = get_labels(merged_indices, results.total_count, train_labels);
        matches[i] = (get_winner(labels, results.total_count) == test_labels[i]);
    }

    int match_count = 0;
    for (size_t i = 0; i < num_test_rows; i++) {
        match_count += matches[i];
    }
    print_time("finished eval_plurality");
    printf("match count: %d\n", match_count);
}


size_t min(size_t i, size_t j) {
    if (i < j) {
        return i;
    }
    return j;
}

/**
 * Given a forest and a set of test points, evaluate each point based on the nearest
 * `num_neighbors` neighbors by L2 distance.
 */
void eval_deduped_l2(RandomBinaryForest *forest, RbfConfig cfg, size_t num_neighbors,
                     feature_type *train_data, feature_type *test_data, label_type *train_labels, label_type *test_labels,
                     size_t num_test_rows, size_t num_features) {
    print_time("started eval_l2");
    // get deduped result indices sorted by l2 distance of reference (training) points from query points:
    size_t *counts;
    rownum_type **results = batch_query_forest_dedup_results_sorted(forest, train_data,
        test_data, num_features, num_test_rows, l2_compare, &counts);

    // for each query row, get labels of first `num_neighbors` results for that row, then get winner
    int match_count = 0;
    for (size_t i = 0; i < num_test_rows; i++) {
        label_type *labels = malloc(sizeof(label_type) * counts[i]);
        size_t count = min(counts[i], num_neighbors);
        for (size_t j = 0; j < count; j++) {
            labels[j] = train_labels[results[i][j]];
        }
        match_count += (get_winner(labels, count) == test_labels[i]);
        free(labels);
    }
    print_time("finished eval_l2");
    printf("match count: %d\n", match_count);
}

int main() {
    srand(2719);
    rbf_init();

    // read training data
    size_t bytes;
    feature_type *pre_train_data = read_file("fashion/train_images.bin", &bytes);
    size_t train_bytes = bytes;
    label_type *train_labels = (label_type *) read_file("fashion/train_labels.bin", &bytes);
    size_t num_rows = bytes;
    size_t num_features = train_bytes / bytes;

    // build RBF
    RbfConfig cfg = {256, // num_trees
                     20, // tree_depth
                      4, // leaf_size
               num_rows,
           num_features,
                     28};  // num_features_to_compare

    feature_type *train_data = transpose(pre_train_data, cfg.num_rows, cfg.num_features);
    free(pre_train_data);

    print_time("started training");
    RandomBinaryForest *forest = train_forest(train_data, &cfg);
    print_time("finished training");

    // read test data
    feature_type *test_data = read_file("fashion/test_images.bin", &bytes);
    label_type *test_labels = (label_type *) read_file("fashion/test_labels.bin", &bytes);
    size_t num_test_rows = bytes;

    // evaluate
    eval_plurality(forest, cfg, test_data, train_labels, test_labels, num_test_rows, num_features);
    eval_deduped_l2(forest, cfg, 5, train_data, test_data, train_labels, test_labels, num_test_rows, cfg.num_features);
}
