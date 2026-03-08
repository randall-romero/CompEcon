# from numba import jit, void, int_, double
# import numba as nb
#
# # All methods must be given signatures
#
# @jit(nopython=True)
# class Shrubbery(object):
#     @void(nb.int_, nb.int_)
#     def __init__(self, w, h):
#         # All instance attributes must be defined in the initializer
#         self.width = w
#         self.height = h
#
#         # Types can be explicitly specified through casts
#         self.some_attr = double(1.0)
#
#     @nb.int_()
#     def area(self):
#         return self.width * self.height
#
#     @void()
#     def describe(self):
#         print("This shrubbery is ", self.width,
#               "by", self.height, "cubits.")
