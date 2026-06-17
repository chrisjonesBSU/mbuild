"""Spatial cell list for accelerating hard-sphere overlap checks.

The :class:`CellList` buckets particles into a regular grid of cells whose side
length is at least the overlap cutoff. Because a candidate can only overlap
particles in its own cell or the 26 surrounding cells (3D), an overlap query
costs ``O(local density)`` instead of ``O(N)``.

This is designed for growth-type Monte Carlo (hard-sphere random walks) where:

- particles are appended one at a time and never moved during growth, so
  insertion is ``O(1)`` and no per-step rebuild is needed;
- a relaxation pass may move every particle at once, after which the list is
  rebuilt in one ``O(N)`` pass from the new positions.

Overlap decisions are delegated to :func:`mbuild.path.path_utils.check_path`,
the same kernel used by the brute-force path, so the accept/reject result is
identical (including the periodic minimum-image convention).
"""

import itertools

import numpy as np
from numba import njit

from mbuild.path.path_utils import check_path

# The 27 cell offsets (self + 26 neighbors) checked for every query.
_OFFSETS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)


@njit(cache=True, fastmath=True)
def first_valid_candidate(
    neighbor_positions, candidates, radius, tolerance, pbc, box_lengths
):
    """Index of the first candidate that overlaps none of ``neighbor_positions``.

    Compiled companion to :meth:`CellList.query_batch`. Checks each candidate
    (in order) against the shared local neighbor set and returns the index of
    the first one with no overlap, or ``-1`` if every candidate overlaps. Uses
    the same minimum-image convention as ``check_path``.
    """
    min_sq_dist = (radius - tolerance) ** 2
    n_neigh = neighbor_positions.shape[0]
    for c in range(candidates.shape[0]):
        overlap = False
        for i in range(n_neigh):
            dist_sq = 0.0
            for j in range(3):
                diff = neighbor_positions[i, j] - candidates[c, j]
                if pbc[j]:
                    diff -= np.round(diff / box_lengths[j]) * box_lengths[j]
                dist_sq += diff * diff
            if dist_sq < min_sq_dist:
                overlap = True
                break
        if not overlap:
            return c
    return -1


class CellList:
    """A spatial hash / linked-cell grid for hard-sphere overlap queries.

    Parameters
    ----------
    cutoff : float, required
        Overlap exclusion distance (the ``radius`` used by
        ``check_path``). Cells are sized so that any particle within ``cutoff``
        of a query point falls in the 27-cell neighborhood. Also used as the
        default overlap radius for :meth:`query`.
    mins : array-like (3,), optional
        Lower corner used as the origin for cell indexing. Defaults to the
        origin. For a bounded/periodic constraint pass the constraint's
        ``mins`` so periodic cells tile the box.
    box_lengths : array-like (3,), optional
        Box length along each axis, used both for periodic cell wrapping and
        for the minimum-image distance in :meth:`query`. Defaults to all
        ``inf`` (unbounded, non-periodic).
    pbc : array-like (3,) of bool, optional
        Per-axis periodic flags. Defaults to all ``False``.
    tolerance : float, optional, default 1e-5
        Rounding tolerance forwarded to ``check_path``.

    Notes
    -----
    Cell geometry (``mins``, ``box_lengths``, ``pbc`` and the cell sizing) is
    fixed at construction. The overlap threshold (``radius``/``tolerance``) used
    by :meth:`query` may be changed afterward as long as ``radius <=
    min(cell_size)``; see :meth:`is_compatible`.
    """

    def __init__(self, cutoff, mins=None, box_lengths=None, pbc=None, tolerance=1e-5):
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}.")
        self.cutoff = float(cutoff)
        self.radius = float(cutoff)
        self.tolerance = float(tolerance)

        self.mins = (
            np.zeros(3, dtype=np.float64)
            if mins is None
            else np.asarray(mins, dtype=np.float64)
        )
        self.box_lengths = (
            np.full(3, np.inf, dtype=np.float32)
            if box_lengths is None
            else np.asarray(box_lengths, dtype=np.float32)
        )
        self.pbc = (
            np.zeros(3, dtype=np.bool_)
            if pbc is None
            else np.asarray(pbc, dtype=np.bool_)
        )

        # Per-axis cell sizing. Periodic axes must tile the box with an integer
        # number of cells, each >= cutoff, so wrapped neighbor indices line up.
        # Non-periodic axes are open-ended (hashed) with cells of side `cutoff`.
        self.n_cells = np.zeros(3, dtype=np.int64)  # 0 == open / non-periodic
        self.cell_size = np.empty(3, dtype=np.float64)
        for a in range(3):
            if self.pbc[a]:
                n = max(1, int(np.floor(self.box_lengths[a] / self.cutoff)))
                self.n_cells[a] = n
                self.cell_size[a] = self.box_lengths[a] / n
            else:
                self.cell_size[a] = self.cutoff

        # cell-tuple -> list of float32 positions
        self.cells = {}
        # Number of points currently binned (used to detect out-of-band changes)
        self.n_binned = 0
        # Log of cell keys inserted in the current transaction, or None.
        self._journal = None

    # -- cell indexing ------------------------------------------------------

    def _cell_coord(self, pos):
        """Return the integer cell tuple a position falls into (periodic-wrapped)."""
        c0 = int(np.floor((pos[0] - self.mins[0]) / self.cell_size[0]))
        c1 = int(np.floor((pos[1] - self.mins[1]) / self.cell_size[1]))
        c2 = int(np.floor((pos[2] - self.mins[2]) / self.cell_size[2]))
        if self.n_cells[0]:
            c0 %= self.n_cells[0]
        if self.n_cells[1]:
            c1 %= self.n_cells[1]
        if self.n_cells[2]:
            c2 %= self.n_cells[2]
        return (c0, c1, c2)

    # -- building / inserting ----------------------------------------------

    def insert(self, pos):
        """Add a single position to its cell. ``O(1)`` amortized.

        Stores a copy so the grid never aliases the caller's coordinate buffer.
        """
        pos = np.array(pos, dtype=np.float32)
        key = self._cell_coord(pos)
        self.cells.setdefault(key, []).append(pos)
        self.n_binned += 1
        if self._journal is not None:
            self._journal.append(key)

    def begin_transaction(self):
        """Start logging inserts so they can be undone with :meth:`rollback`."""
        self._journal = []

    def commit(self):
        """Keep the current transaction's inserts and stop logging."""
        self._journal = None

    def rollback(self):
        """Undo every insert since :meth:`begin_transaction`.

        Inserts are popped in reverse, so each removed point is the most recent
        in its cell and committed (pre-transaction) points are left untouched.
        """
        if self._journal is None:
            return
        for key in reversed(self._journal):
            bucket = self.cells[key]
            bucket.pop()
            if not bucket:
                del self.cells[key]
            self.n_binned -= 1
        self._journal = None

    def rebuild(self, positions):
        """Clear and re-bin every position in one pass. ``O(N)``.

        Call this once at the start of a walk and after any relaxation pass that
        moves particles, since binned positions are copies and go stale when the
        source coordinates change in place.
        """
        self._journal = None
        self.cells.clear()
        self.n_binned = 0
        positions = np.asarray(positions, dtype=np.float32)
        for i in range(positions.shape[0]):
            self.insert(positions[i])

    # -- querying -----------------------------------------------------------

    def _gather(self, new_point):
        """Collect positions from the 27 cells around ``new_point``."""
        base = self._cell_coord(new_point)
        n0, n1, n2 = self.n_cells
        seen = set()
        candidates = []
        for off in _OFFSETS:
            c0 = base[0] + off[0]
            c1 = base[1] + off[1]
            c2 = base[2] + off[2]
            # Wrap periodic axes; dedupe so small boxes don't double-scan a cell.
            if n0:
                c0 %= n0
            if n1:
                c1 %= n1
            if n2:
                c2 %= n2
            key = (c0, c1, c2)
            if key in seen:
                continue
            seen.add(key)
            bucket = self.cells.get(key)
            if bucket:
                candidates.extend(bucket)
        return candidates

    def query(self, new_point):
        """Return ``True`` if ``new_point`` does not overlap any binned point.

        Mirrors ``check_path`` semantics exactly: a return of ``True`` means the
        point is safe to place. The distance test is delegated to ``check_path``
        over only the local candidates.
        """
        new_point = np.asarray(new_point, dtype=np.float32)
        candidates = self._gather(new_point)
        if not candidates:
            return True
        neighbor_positions = np.asarray(candidates, dtype=np.float32)
        return check_path(
            existing_points=neighbor_positions,
            new_point=new_point,
            radius=self.radius,
            tolerance=self.tolerance,
            pbc=self.pbc,
            box_lengths=self.box_lengths,
        )

    def _gather_cells(self, lo, hi):
        """Gather binned positions from every cell in the inclusive box [lo, hi]."""
        cells = self.cells
        candidates = []
        n0, n1, n2 = self.n_cells
        if not (n0 or n1 or n2):
            # Non-periodic: indices are already unique, no wrap/dedup needed.
            for c0 in range(lo[0], hi[0] + 1):
                for c1 in range(lo[1], hi[1] + 1):
                    for c2 in range(lo[2], hi[2] + 1):
                        bucket = cells.get((c0, c1, c2))
                        if bucket:
                            candidates.extend(bucket)
            return candidates
        # Periodic: wrap indices and dedupe so a box wider than the cell count
        # never scans the same cell twice.
        seen = set()
        for c0 in range(lo[0], hi[0] + 1):
            k0 = c0 % n0 if n0 else c0
            for c1 in range(lo[1], hi[1] + 1):
                k1 = c1 % n1 if n1 else c1
                for c2 in range(lo[2], hi[2] + 1):
                    k2 = c2 % n2 if n2 else c2
                    key = (k0, k1, k2)
                    if key in seen:
                        continue
                    seen.add(key)
                    bucket = cells.get(key)
                    if bucket:
                        candidates.extend(bucket)
        return candidates

    def query_batch(self, candidates):
        """Return the index of the first overlap-free candidate, or ``-1``.

        Candidates are expected to be spatially clustered (e.g. the trial batch
        generated around a chain tip). Their combined neighborhood is gathered
        in a single pass and the per-candidate distance test is done in one
        compiled call, amortizing the Python bucketing overhead over the whole
        batch.

        Parameters
        ----------
        candidates : np.ndarray (M, 3)
            Trial points to test, in priority order.

        Returns
        -------
        int
            Index into ``candidates`` of the first point overlapping nothing,
            or ``-1`` if all candidates overlap an existing point.
        """
        candidates = np.ascontiguousarray(candidates, dtype=np.float32)
        if candidates.shape[0] == 0:
            return -1
        # Integer cell coords (pre-wrap) of every candidate; expand by one cell
        # on each side so each candidate's full 27-neighborhood is covered.
        cc = np.floor((candidates - self.mins) / self.cell_size).astype(np.int64)
        lo = cc.min(axis=0) - 1
        hi = cc.max(axis=0) + 1
        gathered = self._gather_cells(lo, hi)
        if not gathered:
            return 0  # nothing nearby: first candidate is trivially valid
        neighbor_positions = np.asarray(gathered, dtype=np.float32)
        return int(
            first_valid_candidate(
                neighbor_positions,
                candidates,
                self.radius,
                self.tolerance,
                self.pbc,
                self.box_lengths,
            )
        )

    # -- reuse / consistency ------------------------------------------------

    def is_compatible(self, radius, mins, box_lengths, pbc):
        """Whether this grid can be reused for a walk with the given settings.

        Reuse is safe when the box geometry matches and the new overlap radius
        is no larger than the cell size the grid was built for (a smaller radius
        just over-fetches candidates; a larger one could miss overlaps).
        """
        if radius > self.cutoff + 1e-12:
            return False
        if not np.array_equal(np.asarray(pbc, dtype=np.bool_), self.pbc):
            return False
        mins = np.asarray(mins, dtype=np.float64)
        box_lengths = np.asarray(box_lengths, dtype=np.float32)
        # Only the periodic axes' box geometry affects cell layout.
        for a in range(3):
            if self.pbc[a]:
                if not np.isclose(box_lengths[a], self.box_lengths[a]):
                    return False
                if not np.isclose(mins[a], self.mins[a]):
                    return False
        return True

    def __len__(self):
        return self.n_binned
