"""Dependency-free Vietoris-Rips persistent homology (H0/H1) for small clouds.

This module computes genuine persistent-homology barcodes over Z/2 using the
standard boundary-matrix reduction algorithm. It exists so that
``TopologicalAdamV4`` can measure the *persistent topology of recent updates*
— H1 classes (loops) in the point cloud traced by the optimizer's projected
update directions — without any external TDA dependency.

Scope and honesty
-----------------
- Exact persistence (not an approximation) for the Vietoris-Rips filtration
  of the given points, up to homological dimension 1.
- Intended for small clouds (N <= ~96): triangles scale as O(N^3). Callers
  should subsample larger clouds (`max_points`).
- H1 of a full Rips complex at maximal scale is trivial (the complex becomes
  a simplex), so every H1 class has a finite death; no infinite H1 bars.

References: Edelsbrunner, Letscher, Zomorodian (2002); Zomorodian & Carlsson
(2005). The triangle-column reduction below uses the standard fact that every
column of the edge-triangle boundary matrix is a 1-cycle, so its pivot (low)
is always a cycle-creating edge; reducing that matrix alone yields the H1
pairing.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import List, Sequence, Tuple

__all__ = ["rips_h1_persistence", "h0_persistence", "max_loop_score"]


def _pairwise_dists(points: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(points)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            s = 0.0
            for a, b in zip(pi, pj):
                diff = a - b
                s += diff * diff
            dij = math.sqrt(s)
            d[i][j] = dij
            d[j][i] = dij
    return d


def _as_point_list(points) -> List[List[float]]:
    """Accept a (N, D) torch tensor, numpy array, or nested sequence."""
    if hasattr(points, "detach"):  # torch tensor
        points = points.detach().cpu().tolist()
    elif hasattr(points, "tolist"):  # numpy array
        points = points.tolist()
    return [list(map(float, row)) for row in points]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def h0_persistence(points) -> List[Tuple[float, float]]:
    """H0 barcode (birth 0; deaths are merge scales; one infinite bar).

    The infinite bar is reported with ``death = math.inf``.
    """
    pts = _as_point_list(points)
    n = len(pts)
    if n == 0:
        return []
    d = _pairwise_dists(pts)
    edges = sorted(
        ((d[i][j], i, j) for i in range(n) for j in range(i + 1, n)),
        key=lambda e: (e[0], e[1], e[2]),
    )
    uf = _UnionFind(n)
    bars = []
    for diam, i, j in edges:
        if uf.union(i, j):
            bars.append((0.0, diam))
    bars.append((0.0, math.inf))
    return bars


def rips_h1_persistence(points) -> List[Tuple[float, float]]:
    """H1 barcode of the Vietoris-Rips filtration of ``points``.

    Returns a list of ``(birth, death)`` with ``death > birth`` (zero-length
    bars are dropped). Exact over Z/2 via boundary-matrix reduction.
    """
    pts = _as_point_list(points)
    n = len(pts)
    if n < 3:
        return []
    d = _pairwise_dists(pts)

    # Edges sorted by filtration value; index = position in sorted order.
    edges = sorted(
        ((d[i][j], i, j) for i in range(n) for j in range(i + 1, n)),
        key=lambda e: (e[0], e[1], e[2]),
    )
    edge_index = {(i, j): k for k, (_, i, j) in enumerate(edges)}
    edge_diam = [e[0] for e in edges]

    # Triangles sorted by filtration value (max edge length).
    tris = []
    for i, j, k in combinations(range(n), 3):
        diam = max(d[i][j], d[i][k], d[j][k])
        tris.append((diam, i, j, k))
    tris.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

    # Reduce the edge-triangle boundary matrix. Every column is a 1-cycle,
    # so pivots always land on cycle-creating edges; the resulting pairs
    # (pivot edge, triangle) are exactly the finite H1 intervals.
    low_to_col: dict = {}
    bars: List[Tuple[float, float]] = []
    for diam, i, j, k in tris:
        col = {edge_index[(i, j)], edge_index[(i, k)], edge_index[(j, k)]}
        while col:
            low = max(col)
            other = low_to_col.get(low)
            if other is None:
                break
            col ^= other
        if col:
            low = max(col)
            low_to_col[low] = col
            birth = edge_diam[low]
            if diam > birth:
                bars.append((birth, diam))
    bars.sort(key=lambda b: (b[1] - b[0]), reverse=True)
    return bars


def max_loop_score(points) -> float:
    """Scale-free prominence of the most persistent loop, in ``[0, 1]``.

    Defined as ``max(death - birth) / diameter`` over the H1 barcode, where
    ``diameter`` is the largest pairwise distance in the cloud. A clean
    circle scores ~0.8; an i.i.d. noise cloud scores near 0; a straight or
    collapsed trajectory scores exactly 0.
    """
    pts = _as_point_list(points)
    if len(pts) < 3:
        return 0.0
    bars = rips_h1_persistence(pts)
    if not bars:
        return 0.0
    d = _pairwise_dists(pts)
    diameter = max(max(row) for row in d)
    if diameter <= 0.0:
        return 0.0
    best = max(death - birth for birth, death in bars)
    return min(max(best / diameter, 0.0), 1.0)
