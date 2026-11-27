import numpy as np

from mbuild.utils.io import import_


class AnglesSampler:
    """
    TODO:Allow for passing a specific 2D weighted pre-defined sample, as opposed to a numpy distribution.
    This would use np.random.choice instead as the sample method.
    NOTES
    -----
    "uniform" distribution should use 'low' and 'high' as kwargs.
    "normal" distribution should use 'loc' and 'scale' as kwargs.
    """

    def __init__(self, distributionStr, kwargs, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        distribution = distributionStr.lower()
        if distribution == "uniform":
            assert "low" in kwargs
            assert "high" in kwargs
        elif distribution == "normal":
            assert "loc" in kwargs
            assert "scale" in kwargs
        elif distribution == "choice":
            assert "a" in kwargs  # p is not required
        else:
            raise NotImplementedError(
                f"Sample Distribution {distributionStr} not supported."
            )
        self.distribution = distribution
        self.kwargs = kwargs

    def sample(self, size=None):
        # Resolve against self.rng at call time so the rng can be swapped in.
        return getattr(self.rng, self.distribution)(size=size, **self.kwargs)


class BoltzmannAngle(AnglesSampler):
    """Sample bond angles from the Boltzmann distribution of a bending potential.

    Angles are drawn from P(theta) ~ sin(theta) * exp(-U(theta) / kT) on
    [0, pi]. The sin(theta) factor is the Jacobian of the 3D bond-vector
    distribution. The distribution is tabulated once on a grid and sampled
    with the "choice" distribution.

    Parameters
    ----------
    expression : str or sympy.Expr
        U as a function of the bond angle, e.g.
        ``"0.5 * k * (theta - theta_eq)**2"``. The angle is in radians and
        theta = pi is a straight chain.
    parameters : dict, optional
        Values for every symbol in ``expression`` other than the angle.
    kT : float, default 1.0
        Thermal energy, in the same units as U.
    angle_symbol : str, default "theta"
        Name of the angle variable in ``expression``.
    n_grid : int, default 4096
        Number of tabulated angles (0.044 degree spacing at the default).
    rng : numpy.random.Generator, optional
        Random number generator used for sampling.

    Raises
    ------
    ValueError
        If ``expression`` contains a symbol with no value in ``parameters``.
    """

    def __init__(
        self,
        expression,
        parameters=None,
        kT=1.0,
        angle_symbol="theta",
        n_grid=4096,
        rng=None,
    ):
        sympy = import_("sympy")
        parameters = dict(parameters or {})
        symbols = {name: sympy.Symbol(name) for name in (angle_symbol, *parameters)}
        # locals= keeps names like "gamma" or "beta" as plain symbols, not sympy functions
        self.expression = sympy.sympify(expression, locals=symbols)
        self.parameters = parameters
        self.kT = kT

        missing = {str(s) for s in self.expression.free_symbols} - set(symbols)
        if missing:
            raise ValueError(
                f"No values given for {sorted(missing)} in U = {self.expression}."
            )

        theta = np.linspace(0.0, np.pi, n_grid)
        energy = sympy.lambdify(
            symbols[angle_symbol], self.expression.subs(parameters), "numpy"
        )
        u = np.broadcast_to(energy(theta), theta.shape) / kT
        p = np.sin(theta) * np.exp(-(u - u.min()))
        super().__init__("choice", {"a": theta, "p": p / p.sum()}, rng=rng)


class HarmonicAngle(BoltzmannAngle):
    """Boltzmann-distributed angles for a harmonic bending potential.

    U(theta) = 0.5 * k * (theta - theta_eq)**2, the HOOMD-blue/GROMACS
    convention; for LAMMPS ``angle_style harmonic`` pass k = 2K.

    Parameters
    ----------
    theta_eq : float
        Equilibrium angle in radians.
    k : float
        Spring constant, in units of energy / rad**2.
    kT : float, default 1.0
        Thermal energy, in the same units as k.
    **kwargs
        Passed to :class:`BoltzmannAngle`.
    """

    expression = "0.5 * k * (theta - theta_eq)**2"

    def __init__(self, theta_eq, k, kT=1.0, **kwargs):
        super().__init__(
            self.expression, {"theta_eq": theta_eq, "k": k}, kT=kT, **kwargs
        )


class CosineAngle(BoltzmannAngle):
    """Boltzmann-distributed angles for a worm-like chain bending potential.

    U(theta) = k * (1 + cos(theta)), the LAMMPS ``angle_style cosine`` and
    Kratky-Porod convention. U is zero for a straight chain (theta = pi).

    Parameters
    ----------
    k : float
        Bending stiffness, in units of energy.
    kT : float, default 1.0
        Thermal energy, in the same units as k.
    **kwargs
        Passed to :class:`BoltzmannAngle`.
    """

    expression = "k * (1 + cos(theta))"

    def __init__(self, k, kT=1.0, **kwargs):
        super().__init__(self.expression, {"k": k}, kT=kT, **kwargs)


class CosineSquaredAngle(BoltzmannAngle):
    """Boltzmann-distributed angles for a cosine-squared bending potential.

    U(theta) = 0.5 * k * (cos(theta) - cos(theta_eq))**2, the HOOMD-blue
    ``md.angle.CosineSquared`` and GROMACS angle type 2 convention; for
    LAMMPS ``angle_style cosine/squared`` pass k = 2K.

    Parameters
    ----------
    theta_eq : float
        Equilibrium angle in radians.
    k : float
        Spring constant, in units of energy.
    kT : float, default 1.0
        Thermal energy, in the same units as k.
    **kwargs
        Passed to :class:`BoltzmannAngle`.
    """

    expression = "0.5 * k * (cos(theta) - cos(theta_eq))**2"

    def __init__(self, theta_eq, k, kT=1.0, **kwargs):
        super().__init__(
            self.expression, {"theta_eq": theta_eq, "k": k}, kT=kT, **kwargs
        )
