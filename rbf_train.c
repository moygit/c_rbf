#include <stdio.h>
#include <stdlib.h>
#include "rbf.h"
#include "_rbf_train.h"


void get_feature_frequencies(uint *local_row_index, char local_feature_array[],
       int feature_num, int num_features, int index_start, int index_end,
       // returns:
       uint *ret_counts, uint *ret_weighted_total) {
    // get frequencies:
    for (int rownum = index_start; rownum < index_end; rownum++) {
        char feature_value = local_feature_array[local_row_index[rownum] * num_features + feature_num];
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
