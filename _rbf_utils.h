/*
 * EVERYTHING HERE IS FOR LOCAL USE ONLY.
 * FOR EXPORTED OBJECTS PLEASE SEE rbf.h.
 */

#ifndef __RBF_UTILS_H__
#define __RBF_UTILS_H__

#include <apr_pools.h>

apr_pool_t *memory_pool;

void die_alloc_err(char *func_name, char *vars);

typedef struct {
    feature_type *query_point;
    feature_type *ref_point;
    rownum_type ref_index;
    size_t point_dimension;
} results_comparison_node;

int l2_square_dist(feature_type *v1, feature_type *v2, size_t vec_size);

#endif /* __RBF_UTILS_H__ */
