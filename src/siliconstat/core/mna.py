"""Modified Nodal Analysis (MNA) system assembly.

Unknown vector layout::

    x = [ v_1 ... v_(N-1) | i_b0 ... i_b(M-1) ]

where node ``0`` is ground (eliminated) so circuit node ``k`` maps to matrix
row ``k-1``, and each branch unknown (voltage sources, inductors) occupies one
row after the node block.

The KCL sign convention is ``G @ x = b`` with ``b`` the current *injected into*
each node.  A device current that flows **out of** a node into the device
appears with a ``+`` sign on the left-hand side.
"""

from __future__ import annotations

import numpy as np

from .exceptions import NumericalError

__all__ = ["MnaSystem"]


class MnaSystem:
    """Dense MNA matrix builder.

    Dense storage is deliberate: SiliconStat targets transistor-level analog
    blocks of tens of nodes, where dense LU is faster than sparse bookkeeping
    and vastly simpler to keep correct.  ``docs/circuit_solver.md`` records the
    crossover point measured on this code base.
    """

    __slots__ = ("size", "n_nodes", "G", "b", "dtype")

    def __init__(self, n_nodes: int, n_branches: int, dtype: type = float) -> None:
        # n_nodes counts ground; ground row/col is eliminated.
        self.n_nodes = n_nodes
        self.size = (n_nodes - 1) + n_branches
        self.dtype = dtype
        self.G = np.zeros((self.size, self.size), dtype=dtype)
        self.b = np.zeros(self.size, dtype=dtype)

    # -- index helpers -----------------------------------------------------
    @staticmethod
    def node_row(node: int) -> int:
        """Matrix row for circuit node index *node* (``-1`` for ground)."""
        return node - 1

    def branch_row(self, branch: int) -> int:
        return (self.n_nodes - 1) + branch

    # -- stamping ----------------------------------------------------------
    def add(self, row: int, col: int, value: complex | float) -> None:
        if row >= 0 and col >= 0:
            self.G[row, col] += value

    def add_rhs(self, row: int, value: complex | float) -> None:
        if row >= 0:
            self.b[row] += value

    def add_conductance(self, node_a: int, node_b: int, g: complex | float) -> None:
        """Stamp a two-terminal conductance *g* between two circuit nodes."""
        ra, rb = node_a - 1, node_b - 1
        if ra >= 0:
            self.G[ra, ra] += g
        if rb >= 0:
            self.G[rb, rb] += g
        if ra >= 0 and rb >= 0:
            self.G[ra, rb] -= g
            self.G[rb, ra] -= g

    def add_vccs(self, n_out_p: int, n_out_n: int, n_ctrl_p: int, n_ctrl_n: int,
                 gain: complex | float) -> None:
        """Stamp a voltage-controlled current source.

        Current ``gain * (v[n_ctrl_p] - v[n_ctrl_n])`` flows out of
        ``n_out_p`` into the device and back into ``n_out_n``.
        """
        rp, rn = n_out_p - 1, n_out_n - 1
        cp, cn = n_ctrl_p - 1, n_ctrl_n - 1
        if rp >= 0 and cp >= 0:
            self.G[rp, cp] += gain
        if rp >= 0 and cn >= 0:
            self.G[rp, cn] -= gain
        if rn >= 0 and cp >= 0:
            self.G[rn, cp] -= gain
        if rn >= 0 and cn >= 0:
            self.G[rn, cn] += gain

    def add_current(self, node_p: int, node_n: int, current: complex | float) -> None:
        """Inject *current* into ``node_n`` and out of ``node_p``.

        i.e. a current source whose arrow points from ``node_p`` through the
        device to ``node_n`` (SPICE convention).
        """
        self.add_rhs(node_p - 1, -current)
        self.add_rhs(node_n - 1, +current)

    # -- solving -----------------------------------------------------------
    def solve(self, *, check_condition: bool = True) -> np.ndarray:
        """Solve ``G x = b`` with explicit singularity / conditioning checks."""
        if self.size == 0:
            return np.zeros(0, dtype=self.dtype)
        if not np.all(np.isfinite(self.G)):
            raise NumericalError("MNA matrix contains NaN or Inf entries")
        if not np.all(np.isfinite(self.b)):
            raise NumericalError("MNA right-hand side contains NaN or Inf entries")
        try:
            x = np.linalg.solve(self.G, self.b)
        except np.linalg.LinAlgError as exc:
            cond = float("nan")
            try:
                cond = float(np.linalg.cond(self.G))
            except Exception:  # pragma: no cover - cond can itself fail
                pass
            raise NumericalError(
                "MNA matrix is singular -- the circuit likely has a floating "
                "node, a shorted voltage-source loop, or a missing DC path to "
                "ground",
                condition=cond, detail=str(exc),
            ) from exc
        if not np.all(np.isfinite(x)):
            raise NumericalError("MNA solution contains NaN or Inf")
        if check_condition:
            # Cheap 1-norm condition estimate via LU is not exposed by numpy;
            # use the (small) dense estimate only when the system is tiny.
            if self.size <= 64:
                cond = float(np.linalg.cond(self.G))
                if cond > 1e14:
                    raise NumericalError(
                        "MNA matrix is numerically ill-conditioned; results "
                        "would not be trustworthy",
                        condition=cond,
                    )
        return x
