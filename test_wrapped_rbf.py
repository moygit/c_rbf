#!/usr/bin/env python
import ctypes


class RbfConfig(ctypes.Structure):
    _fields_ = [("num_trees", ctypes.c_size_t),
                ("tree_depth", ctypes.c_size_t),
                ("leaf_size", ctypes.c_size_t),
                ("num_rows", ctypes.c_uint32),
                ("num_features", ctypes.c_uint32),
                ("num_features_to_compare", ctypes.c_uint32)]

    def __init__(self, num_trees, tree_depth, leaf_size, num_rows, num_features, num_features_to_compare):
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.leaf_size = leaf_size
        self.num_rows = num_rows
        self.num_features = num_features
        self.num_features_to_compare = num_features_to_compare

    def __repr__(self):
        return f"num_trees: {self.num_trees}, tree_depth: {self.tree_depth}, leaf_size: {self.leaf_size}, num_rows: {self.num_rows}, num_features: {self.num_features}, num_features_to_compare: {self.num_features_to_compare}"


class RbfResultsArray(ctypes.Structure):
    _fields_ = [("results", ctypes.POINTER(ctypes.c_uint32)), ("num_results", ctypes.c_size_t)]


class Point(ctypes.Structure):
    _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int)]
    _libc = ctypes.CDLL("./librbf.so")

    def __init__(self, x=None, y=None):
        if x:
            self.x = x
            self.y = y
        else:
            point = self.get_point()
            self.x = point.x
            self.y = point.y

    def __repr__(self):
        return '({0}, {1})'.format(self.x, self.y)

    def show_point(self):
        self.show_point_func(self)

    def move_point(self):
        self.move_point_func(self)

    def move_point_by_ref(self):
        self.move_point_by_ref_func(self)

    @staticmethod
    def wrap_function(funcname, restype, argtypes):
        func = Point._libc.__getattr__(funcname)
        func.restype = restype
        func.argtypes = argtypes
        return func

Point.get_point = Point.wrap_function('get_point', Point, None)
Point.show_point_func = Point.wrap_function('show_point', None, [Point])
Point.move_point_func = Point.wrap_function('move_point', None, [Point])
Point.move_point_by_ref_func = Point.wrap_function('move_point_by_ref', None,
                                                   [ctypes.POINTER(Point)])

if __name__ == '__main__':
    ###########################################################################
    print("Pass a struct into C")
    a = Point(1, 2)
    print("Point in python is", a)
    a.show_point()
    print()

    ###########################################################################
    print("Pass by value")
    a = Point(5, 6)
    print("Point in python is", a)
    a.move_point()
    print("Point in python is", a)
    print()

    ###########################################################################
    print("Pass by reference")
    a = Point(5, 6)
    print("Point in python is", a)
    a.move_point_by_ref()
    print("Point in python is", a)
    print()

    ###########################################################################
    print("Get Struct from C")
    a = Point()
    print("New Point in python (from C) is", a)
    a = Point()
    print("New Point in python (from C) is", a)
    a = Point()
    print("New Point in python (from C) is", a)
    a = Point()
    print("New Point in python (from C) is", a)
