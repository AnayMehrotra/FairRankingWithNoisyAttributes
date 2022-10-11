# birkhoff.py - decompose a doubly stochastic matrix into permutation matrices
#
# Copyright 2015 Jeffrey Finkelstein.
#
# This file is part of Birkhoff.
#
# Birkhoff is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Birkhoff is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Birkhoff.  If not, see <http://www.gnu.org/licenses/>.
"""Provides a function for computing the Birkhoff--von Neumann decomposition of
a doubly stochastic matrix into a convex combination of permutation matrices.

"""
# Imports from built-in libraries.
from __future__ import division
import itertools
import time

# Imports from third-party libraries.
from networkx import from_numpy_matrix
from networkx import Graph
from networkx.algorithms.bipartite.matching import maximum_matching, minimum_weight_full_matching
import numpy as np

#: The current version of this package.
__version__ = '0.0.5'

#: Any number smaller than this will be rounded down to 0 when computing the
#: difference between NumPy arrays of floats.
TOLERANCE = 1e-8 

VERBOSE = False

debug = lambda str : f"print(\"{str}\",\"=\",eval(\"{str}\"))"

def to_permutation_matrix(matches, m, n):
    """Converts a permutation into a permutation matrix.

    `matches` is a dictionary whose keys are vertices and whose values are
    partners. For each vertex ``u`` and ``v``, entry (``u``, ``v``) in the
    returned matrix will be a ``1`` if and only if ``matches[u] == v``.

    Pre-condition: `matches` must be a permutation on an initial subset of the
    natural numbers.

    Returns a permutation matrix as a square NumPy array.

    """
    P = np.zeros((m, n))
    # This is a cleverer way of doing
    #
    #     for (u, v) in matches.items():
    #         P[u, v] = 1
    #
    P[list(zip(*(matches.items())))] = 1 # lol
    return P

def four_blocks(topleft, topright, bottomleft, bottomright):
    """Convenience function that creates a block matrix with the specified
    blocks.

    Each argument must be a NumPy matrix. The two top matrices must have the
    same number of rows, as must the two bottom matrices. The two left matrices
    must have the same number of columns, as must the two right matrices.

    """
    return np.vstack((np.hstack((topleft, topright)),
                  np.hstack((bottomleft, bottomright))))

def to_bipartite_matrix(A):
    """Returns the adjacency matrix of a bipartite graph whose biadjacency
    matrix is `A`.

    `A` must be a NumPy array.

    If `A` has **m** rows and **n** columns, then the returned matrix has **m +
    n** rows and columns.

    """
    m, n = A.shape
    return four_blocks(np.zeros((m, m)), A, A.T, np.zeros((n, n)))

def to_pattern_matrix(D):
    """Returns the Boolean matrix in the same shape as `D` with ones exactly
    where there are nonzero entries in `D`.

    `D` must be a NumPy array.

    """
    result = np.zeros_like(D)
    # This is a cleverer way of doing
    #
    #     for (u, v) in zip(*(D.nonzero())):
    #         result[u, v] = 1
    #
    result[D.nonzero()] = 1
    return result

def fast_decomposition(D):
    """Returns the Birkhoff--von Neumann decomposition of the doubly
    stochastic matrix `D`.

    The input `D` must be a square NumPy array representing a doubly
    stochastic matrix (that is, a matrix whose entries are nonnegative
    reals and whose row sums and column sums are all 1). Each doubly
    stochastic matrix is a convex combination of at most ``n ** 2``
    permutation matrices, where ``n`` is the dimension of the input
    array.

    The returned value is a list of pairs whose length is at most ``n **
    2``. In each pair, the first element is a real number in the interval **(0,
    1]** and the second element is a NumPy array representing a permutation
    matrix. This represents the doubly stochastic matrix as a convex
    combination of the permutation matrices.

    The input matrix may also be a scalar multiple of a doubly
    stochastic matrix, in which case the row sums and column sums must
    each be *c*, for some positive real number *c*. This may be useful
    in avoiding precision issues: given a doubly stochastic matrix that
    will have many entries close to one, multiply it by a large positive
    integer. The returned permutation matrices will be the same
    regardless of whether the given matrix is a doubly stochastic matrix
    or a scalar multiple of a doubly stochastic matrix, but in the
    latter case, the coefficients will all be scaled by the appropriate
    scalar multiple, and their sum will be that scalar instead of one.

    For example::

        >>> import numpy as np
        >>> from birkhoff import birkhoff_von_neumann_decomposition as decomp
        >>> D = np.ones((2, 2))
        >>> zipped_pairs = decomp(D)
        >>> coefficients, permutations = zip(*zipped_pairs)
        >>> coefficients
        (1.0, 1.0)
        >>> permutations[0]
        array([[ 1.,  0.],
               [ 0.,  1.]])
        >>> permutations[1]
        array([[ 0.,  1.],
               [ 1.,  0.]])
        >>> zipped_pairs = decomp(D / 2)  # halve each value in the matrix
        >>> coefficients, permutations = zip(*zipped_pairs)
        >>> coefficients  # will be half as large as before
        (0.5, 0.5)
        >>> permutations[0]  # will be the same as before
        array([[ 1.,  0.],
               [ 0.,  1.]])
        >>> permutations[1]
        array([[ 0.,  1.],
               [ 1.,  0.]])

    The returned list of pairs is given in the order computed by the algorithm
    (so in particular they are not sorted in any way).

    """
    m, n = D.shape
    # We allow non-square matrices
    # if m != n: raise ValueError('Input matrix must be square ({} x {})'.format(m, n))

    indices = list(itertools.product(range(m), range(n)))
    # These two lists will store the result as we build it up each iteration.
    coefficients = []
    matchings = []
    permutations = []
    # Create a copy of D so that we don't modify it directly. Cast the
    # entries of the matrix to floating point numbers, regardless of
    # whether they were integers.
    S = D.astype('float')
    ijk  = 0
    S[S < TOLERANCE] = 0.0
    tot_removed = 0

    while not np.all(S == 0):
        ijk += 1

        # Create an undirected graph whose adjacency matrix contains a 1
        # exactly where the matrix S has a nonzero entry.
        st = time.time()
        #
        V = np.sum(S, axis=1) # total weight indicent on items

        # find the heaviest item whose weight is smaller than 1-tot_removed
        mx = -1
        mxi = -1
        tmp_sum = 0
        for i in range(m):
            if V[i] < (1-1e-5)*(1-tot_removed):
                if mx < V[i]:
                    mx = max(mx, V[i])
                    mxi = i
            else:
                tmp_sum += 1
        # remove at most 1-mx weight in this iteration
        q_tmp = 1 - tot_removed - mx

        (r, c) = S.nonzero()
        edgesTmp = np.dstack((r, c + m))[0]
        edges = np.zeros((len(edgesTmp), 3))
        # set edge weights
        for i in range(len(edgesTmp)):
            # give "priority" to edges originating on items with weight one "after scaling" (i.e., with weight 1-tot_removed)
            if mxi == edgesTmp[i][0]:
                priority = m*n + S[edgesTmp[i][0]][edgesTmp[i][1]-m]
            elif V[edgesTmp[i][0]] >= (1-1e-5)*(1-tot_removed):
                priority = -m*n - S[edgesTmp[i][0]][edgesTmp[i][1]-m]
            else:
                priority = -S[edgesTmp[i][0]][edgesTmp[i][1]-m]
            edges[i] = tuple(edgesTmp[i])+ tuple([priority])

        G = Graph()
        G.add_nodes_from(range(n+m))
        G.add_weighted_edges_from(edges)

        # Construct the bipartite graph whose left and right vertices both
        # represent the vertex set of the pattern graph (whose adjacency matrix
        # is ``W``).
        # print('Starting')
        #
        if VERBOSE: print(f'Time taken by step 1={time.time() - st}')
        st = time.time()

        # The bipartite maximum matching algorithm requires specifying
        # the left set of nodes in the bipartite graph. By construction,
        # the left set of nodes is {0, ..., n - 1} and the right set is
        # {n, ..., 2n - 1}; see `to_bipartite_matrix()`.
        left_nodes = range(m)

        M = minimum_weight_full_matching(G, left_nodes)
        if VERBOSE: print(f'Time taken to find matching={time.time() - st}')
        st = time.time()

        # However, since we have both a left vertex set and a right vertex set,
        # each representing the original vertex set of the pattern graph
        # (``W``), we need to convert any vertex greater than ``n`` to its
        # original vertex number. To do this,
        #
        #   - ignore any keys greater than ``n``, since they are already
        #     covered by earlier key/value pairs,
        #   - ensure that all values are less than ``n``.
        #

        LII = {(u, v) for u, v in M.items() if u < m}
        LI = {(u, v - m) for u, v in LII}
        M = {u: v - m for u, v in LII}

        # Convert that perfect matching to a permutation matrix.
        P = to_permutation_matrix(M, m, n)
        if VERBOSE: print(f'Time taken by step 3={time.time() - st}')
        st = time.time()

        # Get the smallest entry of S corresponding to the 1 entries in the
        # permutation matrix.
        # The next step is linear time. We cannot iterate over all
        # elements of P (that's quadratic time)
        q = min(S[i, j] for (i, j) in LI) # indices if P[i, j] == 1)
        q = min(q, q_tmp)

        # Store the coefficient and the permutation matrix for later.
        coefficients.append(q)
        permutations.append(P)
        matchings.append(LII)

        tot_removed += q

        # Subtract P scaled by q. After this subtraction, S has a zero entry
        # where the value q used to live.
        S -= q * P

        if VERBOSE: print(f'Time taken by step 4={time.time() - st}')

        S[S < TOLERANCE] = 0.0
    return list(zip(coefficients, permutations, matchings))
