#ifndef __POINT_H__
#define __POINT_H__

#include <stdint.h>
#include <stdlib.h>

typedef _Bool bool;
typedef char feature_type;
typedef uint32_t rownum_type;
typedef uint32_t colnum_type;
typedef uint32_t stats_type;
typedef size_t treeindex_type;

bool test();

#define NUM_TREES 20
// #define TREE_SIZE (1 << 25)     // 2^25, roughly 32M
#define LEAF_SIZE 8
#define NUM_BITS 32  // we want to store some shorts but also some ints, so need 4 bytes
#define HIGH_BIT 31  // low bit is 0, high bit is 15
#define HIGH_BIT_1 (1 << HIGH_BIT)
#define NUM_CHARS 256

typedef struct {
	// We have arrays of arrays of features. Instead of expensively moving those rows around when
	// sorting and partitioning we have an index into those and move the index elements around.
	// Lookups will be slightly slower but we'll save time overall.
    rownum_type *row_index;
    rownum_type num_rows;

	// Ugliness alert:
	// Each tree node is a pair. For speed and space efficiency we'll store the tree in 2 arrays
	// using the standard trick for storing a binary tree in an array (with indexing starting at 0,
	// left child of n goes in 2n+1, right child goes in 2n+2). The pairs are either:
	// - if it's an internal node: the feature number and the value at which to split the feature
	// - if it's a leaf node: start and end indices in the rowIndex array; that view in the rowIndex
	//   array tells us the indices of rows in the original training set that are in this leaf
	// We distinguish the two cases by doing some bit-arithmetic.
	// 1. Yes, I know this is ugly, but the alternative is to have a whole 'nother pair of large arrays.
	// 2. Yes, I considered using hashmaps instead [in Go], but they're much slower (expected) and also
	//    take WAY more memory (which surprised me).
    rownum_type *tree_first;
    rownum_type *tree_second;
    treeindex_type tree_size;
    treeindex_type num_internal_nodes;
    treeindex_type num_leaves;
} RandomBinaryTree;

typedef struct {
    size_t num_trees;
    size_t tree_depth;
    size_t leaf_size;
    rownum_type num_rows;
    colnum_type num_features;
    colnum_type num_features_to_compare;
} RbfConfig;

RandomBinaryTree **train_forest(feature_type *feature_array, RbfConfig *config);

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
