#ifndef __POINT_H__
#define __POINT_H__

#include <stdlib.h>

typedef _Bool bool;

bool test();

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
