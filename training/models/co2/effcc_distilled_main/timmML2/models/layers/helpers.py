""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
# from itertools import repeat
# # from torch._six import container_abcs
# import collections.abc

# # From PyTorch internals
# def _ntuple(n):
#     def parse(x):
#         if isinstance(x, container_abcs.Iterable):
#         if isinstance(x, collections.abc.Iterable):
#             return x
#         return tuple(repeat(x, n))
#     return parse

from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
    
tup_single = _ntuple(1)
tup_pair = _ntuple(2)
tup_triple = _ntuple(3)
tup_quadruple = _ntuple(4)
ntup = _ntuple





