#include <stdio.h>
#include <stdlib.h>
#include "rbf.h"
#include "_rbf_train.h"


void get_feature_frequencies(rownum_t *local_row_index, feature_t local_feature_array[],
       colnum_t feature_num, colnum_t num_features, rownum_t index_start, rownum_t index_end,
       // returns:
       stats_t *ret_counts, stats_t *ret_weighted_total) {
    // get frequencies:
    for (rownum_t rownum = index_start; rownum < index_end; rownum++) {
        feature_t feature_value = local_feature_array[local_row_index[rownum] * num_features + feature_num];
        ret_counts[(size_t) feature_value] += 1;
        ret_weighted_total[0] += feature_value;
    }
    return;
}


/* display a Point value */
void show_point(Point point) {
    printf("Point in C      is (%d, %d)\n", point.x, point.y);
}

/* Increment a Point which was passed by value */
void move_point(Point point) {
    show_point(point);
    point.x++;
    point.y++;
    show_point(point);
}

/* Increment a Point which was passed by reference */
void move_point_by_ref(Point *point) {
    show_point(*point);
    point->x++;
    point->y++;
    show_point(*point);
}

/* Return by value */
Point get_point(void) {
    static int counter = 0;
    int x = counter ++;
    int y = counter ++;
    Point point = {x, y};
    printf("Returning Point    (%d, %d)\n", point.x, point.y);
    return point;
}
