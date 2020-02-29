#ifndef __POINT_H__
#define __POINT_H__

#include <stdlib.h>

typedef _Bool bool;

bool test();

#define NUM_TREES 20
// #define TREE_SIZE (1 << 25)     // 2^25, roughly 32M
#define LEAF_SIZE 8
#define NUM_BITS 32  // we want to store some shorts but also some ints, so need 4 bytes
#define HIGH_BIT 31  // low bit is 0, high bit is 15
#define HIGH_BIT_1 (1 << HIGH_BIT)
#define NUM_CHARS 256

typedef struct {
    uint *row_index;
    int num_rows;
    int *tree_first;
    int *tree_second;
    int tree_size;
    int num_internal_nodes;
    int num_leaves;
} RandomBinaryTree;

RandomBinaryTree **train_forest_with_feature_array(char *feature_array, size_t num_trees, int tree_depth, int leaf_size,
                                                   int num_rows, int num_features, int num_features_to_compare);

void query_forest(RandomBinaryTree *forest, int num_trees,
        // returns:
        int *results, int num_results);

void print_time(char *msg);

/* Simple structure for ctypes example */
typedef struct {
    int x;
    int y;
} Point;

void show_point(Point point);
void move_point(Point point);
void move_point_by_ref(Point *point);
Point get_point(void);

#endif /* __POINT_H__ */
