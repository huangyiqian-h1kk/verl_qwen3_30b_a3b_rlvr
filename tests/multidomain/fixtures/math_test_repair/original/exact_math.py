"""Bounded exact checks for closed finite products and conjugate Gamma pairs.

No numerical closeness, sample-ID shortcuts, or global i substitution. Lowercase
i is interpreted as the imaginary unit ONLY in a closed conjugate Gamma pair
Gamma(N+i) Gamma(N-i), with positive integer N and no other free i. Ordinary
symbolic i and product-bound indices are untouched. Unsupported expressions go
back to Math-Verify. DLMF 5.5.1 and 5.5.3 give the exact pair identity.
"""
import math

MAX_TERMS = 10000
MAX_BITS = 300000


class Unsupported(ValueError):
    pass


def assignment(expr):
    import sympy as s
    if isinstance(expr, s.Equality) and isinstance(expr.lhs, s.Symbol) and expr.lhs not in expr.rhs.free_symbols:
        return expr.rhs
    return expr


def bounded(value):
    import sympy as s
    if not isinstance(value, s.Rational) or max(int(value.p).bit_length(), int(value.q).bit_length()) > MAX_BITS:
        raise Unsupported('Not a bounded exact rational')
    return value


def rational(expr):
    """Evaluate a small rational AST; never call Product.doit or expand."""
    import sympy as s
    if isinstance(expr, s.Rational):
        return bounded(expr)
    if isinstance(expr, s.Add):
        return bounded(sum(rational(x) for x in expr.args))
    if isinstance(expr, s.Mul):
        value = s.S.One
        for x in expr.args:
            value = bounded(value * rational(x))
        return value
    if isinstance(expr, s.Pow) and expr.exp.is_Integer and abs(int(expr.exp)) <= MAX_TERMS:
        base = rational(expr.base)
        exponent = int(expr.exp)
        if max(int(base.p).bit_length(), int(base.q).bit_length()) * abs(exponent) > MAX_BITS:
            raise Unsupported('Power budget exceeded')
        return bounded(base ** exponent)
    if isinstance(expr, s.Product) and len(expr.limits) == 1:
        index, lo, hi = expr.limits[0]
        if not (lo.is_Integer and hi.is_Integer and 0 <= int(hi-lo+1) <= MAX_TERMS):
            raise Unsupported('Product bounds outside budget')
        if expr.function.free_symbols - {index}:
            raise Unsupported('Product is not closed')
        poly = s.Poly(expr.function, index)
        if poly.degree() > 4 or any(not c.is_Rational for c in poly.all_coeffs()):
            raise Unsupported('Unsupported product polynomial')
        coeffs = poly.all_coeffs()
        value = s.S.One
        for k in range(int(lo), int(hi)+1):
            term = s.S.Zero
            for c in coeffs:
                term = term*k+c
            value = bounded(value*term)
        return value
    raise Unsupported('Unsupported exact expression')


def conjugates(expr):
    import sympy as s
    replacements = {}
    imaginary_symbols = {x for x in expr.free_symbols if x.name == 'i'}
    imaginary = next(iter(imaginary_symbols), s.I)
    # An i outside Gamma arguments is ambiguous; do not reinterpret it.
    gammas = expr.atoms(s.gamma)
    if imaginary != s.I and imaginary in expr.xreplace({g:s.S.One for g in gammas}).free_symbols:
        raise Unsupported('Free i outside Gamma')

    def transform(node):
        if not isinstance(node, s.Mul):
            return node
        factors = list(node.args)
        for g in list(factors):
            if g.func != s.gamma:
                continue
            real = s.expand(g.args[0]-imaginary)
            if not real.is_Integer or not 1 <= int(real) <= MAX_TERMS:
                continue
            other = s.gamma(real-imaginary, evaluate=False)
            if other not in factors:
                continue
            factors.remove(g)
            factors.remove(other)
            n = int(real)
            product = math.prod(k*k+1 for k in range(1,n))
            if product.bit_length() > MAX_BITS:
                raise Unsupported('Gamma product budget exceeded')
            placeholder = s.Dummy('gamma_product')
            replacements[placeholder] = s.Integer(product)
            factors.extend([placeholder, s.pi/s.sinh(s.pi)])
        return s.Mul(*factors)

    result = expr.replace(lambda x:isinstance(x,s.Mul), transform)
    if not replacements:
        return expr, False
    # Cancel only the small transcendental coefficient, with the large integers
    # kept as placeholders. This avoids expansion of a 2,013-factor Product.
    result = s.cancel(result.rewrite(s.exp))
    if result.free_symbols - set(replacements):
        raise Unsupported('Gamma expression is not a closed constant')
    result = result.xreplace(replacements)
    return result, True


def compare(gold, pred):
    """Return (True/False, route) when exact, otherwise (None, None)."""
    import sympy as s
    if not isinstance(gold,s.Basic) or not isinstance(pred,s.Basic):
        return None, None
    gold, pred = assignment(gold), assignment(pred)
    if not any(x.has(s.Product,s.gamma) for x in (gold,pred)):
        return None,None
    try:
        g, gr = conjugates(gold)
        p, pr = conjugates(pred)
        a, b = rational(g), rational(p)
        return bool(a == b), 'exact_gamma_product' if gr or pr else 'exact_finite_product'
    except (Unsupported, ValueError, TypeError, s.PolynomialError, ZeroDivisionError):
        return None,None
