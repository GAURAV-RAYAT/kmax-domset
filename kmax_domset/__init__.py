"""
kmax_domset
===========
K-Max algorithm for computing dominating sets in undirected graphs.

Quick start
-----------
>>> from kmax_domset import KMaxAlgorithm
>>> adj = [
...     [0, 1, 0, 0, 1],
...     [1, 0, 1, 0, 0],
...     [0, 1, 0, 1, 0],
...     [0, 0, 1, 0, 1],
...     [1, 0, 0, 1, 0],
... ]
>>> algo = KMaxAlgorithm(adj)
>>> sorted(algo.dominating_set())
[0, 2]
"""

from .algorithm import KMaxAlgorithm

__all__ = ["KMaxAlgorithm"]
__version__ = "0.1.0"
__author__ = "Gaurav Rayat"
