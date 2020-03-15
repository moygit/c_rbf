#include <apr_pools.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rbf.h"
#include "_rbf_utils.h"


extern apr_pool_t *memory_pool;


// Initial setup (memory etc).
void rbf_init() {
    apr_initialize();
    apr_pool_create_unmanaged(&memory_pool);
}


// Call when *alloc returns null
void die_alloc_err(char *func_name, char *vars) {
    fprintf(stderr, "fatal error: function %s, allocating memory for %s\n", func_name, vars);
    exit(EXIT_FAILURE);
}


// Transpose an nxm matrix represented as a single array.
// (Alternatively: convert between row-major and column-major representations.)
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


// Square of the L^2 distance between two points
int l2_square_dist(feature_type *v1, feature_type *v2, size_t vec_size) {
    int sum = 0;
    #pragma omp simd
    for (size_t i = 0; i < vec_size; i++) {
        int coord_diff = (int) v1[i] - (int) v2[i];
        sum += coord_diff * coord_diff;
    }
    return sum;
}


void print_time(char *msg) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    double count;
    count = t.tv_sec * 1e9;
    count = (count + t.tv_nsec) * 1e-9;
    printf("%f: %s\n", count, msg);
}
