"""Periodic KDTree module.

A wrapper around scipy.spatial.kdtree to implement periodic boundary
conditions

Written by Patrick Varilly, 6 Jul 2012
Released under the scipy license:

Copyright 2001, 2002 Enthought, Inc.  All rights reserved.

Copyright 2003-2013 SciPy Developers.  All rights reserved.  Redistribution
and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.  Redistributions in binary
form must reproduce the above copyright notice, this list of conditions and
the following disclaimer in the documentation and/or other materials
provided with the distribution.  Neither the name of Enthought nor the
names of the SciPy Developers may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE,  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

import heapq
from itertools import chain

import numpy as np
from scipy.spatial import KDTree

import mbuild as mb


def _gen_relevant_images(x, bounds, distance_upper_bound):
    """Map x onto canonical unit cell and produce mirror images."""
    real_x = np.copy(x)

    for i, coord in enumerate(x):
        if bounds[i] > 0.0:
            real_x[i] = x[i] - np.floor(x[i] / bounds[i]) * bounds[i]

    m = len(x)

    xs = [real_x]
    for i in range(m):
        if bounds[i] > 0.0:
            disp = np.zeros(m)
            disp[i] = bounds[i]

            if distance_upper_bound == np.inf:
                xs = list(chain.from_iterable((_ + disp, _, _ - disp) for _ in xs))
            else:
                extra_xs = []

                # Point near lower boundary, include image on upper side
                if abs(real_x[i]) < distance_upper_bound:
                    extra_xs.extend(_ + disp for _ in xs)

                # Point near upper boundary, include image on lower side
                if abs(bounds[i] - real_x[i]) < distance_upper_bound:
                    extra_xs.extend(_ - disp for _ in xs)

                xs.extend(extra_xs)
    return xs


class PeriodicKDTree(KDTree):
    """Cython kd-tree for nearest-neighbor lookup with periodic boundaries.

    See scipy.spatial.kdtree for details on kd-trees.

    Searches with periodic boundaries are implemented by mapping all initial
    data points to one canonical periodic image, building an ordinary kd-tree
    with these points, then querying this kd-tree multiple times, if necessary,
    with all the relevant periodic images of the query point.

    Parameters
    ----------
    data : array-like, shape (n,m), required
        The n data points of dimension m to be indexed. This array is not copied
        unless this is necessary to produce a contiguous array of doubles, and
        so modifying this data will result in bogus results.
    bounds : array_like, shape (m,), required
        Size of the periodic box along each spatial dimension.  A
        negative or zero size for dimension k means that space is not
        periodic along k.
    leafsize : positive integer, optional, default=10
        The number of points at which the algorithm switches over to
        brute-force.

    Note
    ----
    To ensure that no two distinct images of the same point appear in the
    results, it is essential to restrict the maximum distance between a query
    point and a data point to half the smallest box dimension.
    """

    def __init__(self, bounds, data, leafsize=10):
        """Construct Cython kd-tree for nearest-neighbor lookup with periodic boundaries.

        Parameters
        ----------
        bounds : array_like, shape (m,)
            Size of the periodic box along each spatial dimension.  A
            negative or zero size for dimension k means that space is not
            periodic along k.
        data : array-like, shape (n,m)
            The n data points of dimension m to be indexed. This array is
            not copied unless this is necessary to produce a contiguous
            array of doubles, and so modifying this data will result in
            bogus results.
        leafsize : positive integer
            The number of points at which the algorithm switches over to
            brute-force.
        """
        # Map all points to canonical periodic image
        self.bounds = np.array(bounds)
        self.real_data = np.asarray(data)
        wrapped_data = self.real_data - np.where(
            self.bounds > 0.0,
            (np.floor(self.real_data / self.bounds) * self.bounds),
            0.0,
        )

        # Calculate maximum distance_upper_bound
        self.max_distance_upper_bound = np.min(
            np.where(self.bounds > 0, 0.5 * self.bounds, np.inf)
        )

        # Set up underlying kd-tree
        super(PeriodicKDTree, self).__init__(wrapped_data, leafsize)

    @classmethod
    def from_compound(cls, compound, leafsize=10):
        """Create a PeriodicKDTree from a compound.

        See scipy.spatial.kdtree for details on kd-trees.

        Searches with periodic boundaries are implemented by mapping all initial
        data points to one canonical periodic image, building an ordinary kd-tree
        with these points, then querying this kd-tree multiple times, if necessary,
        with all the relevant periodic images of the query point.

        Parameters
        ----------
        compound : mb.Compound, required
            The mbuild Compound to gather periodicity and box information for the
            PeriodicKDTree.
        leafsize : positive integer
            The number of points at which the algorithm switches over to
            brute-force.

        Note
        ----
        To ensure that no two distinct images of the same point appear in the
        results, it is essential to restrict the maximum distance between a query
        point and a data point to half the smallest box dimension.
        """
        if not isinstance(compound, mb.Compound):
            raise TypeError(
                f"Incorrect type of compound. Provided compound of type {type(compound)}. Expected mbuild.Compound"
            )
        if not isinstance(compound.box, mb.Box):
            raise TypeError(
                f"Incorrect type of box. Was provided box of type {type(compound.box)}. Expected mbuild.Box"
            )
        if not np.allclose(compound.box.angles, 90.0):
            raise NotImplementedError(
                "Periodic KDTree search only implemented"
                "for orthorhombic periodic boundaries"
            )
        # Map all points to canonical periodic image.
        compound_bounds = []
        for box_len, period in zip(compound.box.lengths, compound.periodicity):
            if period:
                compound_bounds.append(box_len)
            else:
                compound_bounds.append(0)

        # Set up underlying kd-tree
        return cls(bounds=compound_bounds, data=compound.xyz, leafsize=leafsize)

    # Ideally, KDTree and cKDTree would expose identical query and __query
    # interfaces.  But they don't, and cKDTree.__query is also inaccessible
    # from Python.  We do our best here to cope.
    def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        # This is the internal query method, which guarantees that x
        # is a single point, not an array of points
        #
        # A slight complication: k could be "None", which means "return
        # all neighbors within the given distance_upper_bound".

        # Cap distance_upper_bound
        distance_upper_bound = np.min(
            [distance_upper_bound, self.max_distance_upper_bound]
        )

        # Run queries over all relevant images of x
        hits_list = []
        for real_x in _gen_relevant_images(x, self.bounds, distance_upper_bound):
            d, i = super(PeriodicKDTree, self).query(
                real_x, k, eps, p, distance_upper_bound
            )
            if k > 1:
                hits_list.append(list(zip(d, i)))
            else:
                hits_list.append([(d, i)])

        # Now merge results
        if k > 1:
            return heapq.nsmallest(k, chain(*hits_list))
        elif k == 1:
            return [min(chain(*hits_list))]
        else:
            raise ValueError("Invalid k in periodic_kdtree._KDTree__query")

    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        """Query the kd-tree for nearest neighbors.

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : integer
            The number of nearest neighbors to return.
        eps : non-negative float
            Return approximate nearest neighbors; the kth returned value is
            guaranteed to be no further than (1+eps) times the distance to the
            real k-th nearest neighbor.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use:
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance.  This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Returns
        -------
        d : array of floats
            The distances to the nearest neighbors.
            If x has shape tuple+(self.m,), then d has shape tuple+(k,).
            Missing neighbors are indicated with infinite distances.
        i : ndarray of ints
            The locations of the neighbors in self.data.
            If `x` has shape tuple+(self.m,), then `i` has shape tuple+(k,).
            Missing neighbors are indicated with self.n.
        """
        x = np.asarray(x)
        if np.shape(x)[-1] != self.m:
            raise ValueError(
                "x must consist of vectors of length %d but has "
                "shape %s" % (self.m, np.shape(x))
            )
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        retshape = np.shape(x)[:-1]
        if not isinstance(retshape, tuple):
            if k > 1:
                dd = np.empty(retshape + (k,), dtype=float)
                dd.fill(np.inf)
                ii = np.empty(retshape + (k,), dtype=int)
                ii.fill(self.n)
            elif k == 1:
                dd = np.empty(retshape, dtype=float)
                dd.fill(np.inf)
                ii = np.empty(retshape, dtype=int)
                ii.fill(self.n)
            else:
                raise ValueError(
                    "Requested %s nearest neighbors; acceptable numbers are "
                    "integers greater than or equal to one, or None"
                )
            for c in np.ndindex(retshape):
                hits = self.__query(
                    x[c],
                    k=k,
                    eps=eps,
                    p=p,
                    distance_upper_bound=distance_upper_bound,
                )
                if k > 1:
                    for j, _ in enumerate(hits):
                        dd[c + (j,)], ii[c + (j,)] = hits[j]
                elif k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self.__query(
                x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound
            )
            if k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k, dtype=float)
                dd.fill(np.inf)
                ii = np.empty(k, dtype=int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError(
                    "Requested %s nearest neighbors; acceptable numbers are "
                    "integers greater than or equal to one, or None"
                )

    # Ideally, KDTree and cKDTree would expose identical __query_ball_point
    # interfaces.  But they don't, and cKDTree.__query_ball_point is also
    # inaccessible from Python.  We do our best here to cope.
    def __query_ball_point(self, x, r, p=2.0, eps=0):
        # This is the internal query method, which guarantees that x
        # is a single point, not an array of points

        # Cap r
        r = min(r, self.max_distance_upper_bound)

        # Run queries over all relevant images of x
        results = []
        for real_x in _gen_relevant_images(x, self.bounds, r):
            results.extend(
                super(PeriodicKDTree, self).query_ball_point(real_x, r, p, eps)
            )
        return results

    def query_ball_point(self, x, r, p=2.0, eps=0):
        """Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : positive float
            The radius of points to return.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may
        save substantial amounts of time by putting them in a
        PeriodicKDTree and using query_ball_tree.
        """
        x = np.asarray(x).astype(float)
        if x.shape[-1] != self.m:
            raise ValueError(
                "Searching for a {}d-dimensional point in a {}d-dimensional "
                "KDTree".format(x.shape[-1], self.m)
            )
        if len(x.shape) == 1:
            return self.__query_ball_point(x, r, p, eps)
        else:
            retshape = x.shape[:-1]
            result = np.empty(retshape, dtype=np.object)
            for c in np.ndindex(retshape):
                result[c] = self.__query_ball_point(x[c], r, p, eps)
            return result

    def query_ball_tree(self, other, r, p=2.0, eps=0):
        """Query the ball tree."""
        raise NotImplementedError()

    def query_pairs(self, r, p=2.0, eps=0):
        """Query the pairs."""
        raise NotImplementedError()

    def count_neighbors(self, other, r, p=2.0):
        """Count the neighbors."""
        raise NotImplementedError()

    def sparse_distance_matrix(self, other, max_distance, p=2.0):
        """Get a sparse distance matrix."""
        raise NotImplementedError()
