#include <stdio.h>
#include <stdlib.h>
#include "rbf.h"


_Bool test() {
    return 1;
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
