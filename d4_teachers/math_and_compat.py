"""Fix latex2sympy2_extended 1.11.0 And construction inside the isolated worker.

SymPy may simplify a conjunction to a relational expression or a boolean. Only
an actual And instance can carry the parser's original argument-order metadata.
The underlying simplification and mathematical comparison remain unchanged.
"""


def install():
    from importlib.metadata import version

    from latex2sympy2_extended.logic import And
    from sympy import And as SympyAnd
    from sympy.core.sympify import sympify

    if (
        version("latex2sympy2_extended") != "1.11.0"
        or version("sympy") != "1.14.0"
        or And.__bases__ != (SympyAnd,)
    ):
        raise RuntimeError("Unexpected isolated And implementation")
    if getattr(And.__new__, "_d4_typeguard", False):
        return

    def construct(cls, *args, **kwargs):
        args = [sympify(arg) for arg in args]
        obj = super(And, cls).__new__(cls, *args, **kwargs)
        if isinstance(obj, cls):
            obj._unsorted_args = args
        return obj

    construct._d4_typeguard = True
    And.__new__ = staticmethod(construct)
