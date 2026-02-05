import sympy as sp
from src.config import DOT_MAP, A, A_dot, Sigma, r, t, z


def apply_dot_operator(expr):
    """Applies the coordinate transformation rules."""
    expr = expr.doit()
    for name, (func, func_dot, func_ddot) in DOT_MAP.items():
        dt_X = func_dot - A * sp.diff(func, r)

        dt_dt_X = (
            func_ddot
            - 2 * A * sp.diff(func_dot, r)
            - A_dot * sp.diff(func, r)
            + A * sp.diff(A, r) * sp.diff(func, r)
            + (A**2) * sp.diff(func, r, r)
        )

        dt_dr_X = sp.diff(func_dot, r) - sp.diff(A, r) * sp.diff(func, r) - A * sp.diff(func, r, r)
        dt_dz_X = sp.diff(func_dot, z) - sp.diff(A, z) * sp.diff(func, r) - A * sp.diff(func, r, z)

        expr = expr.replace(sp.Derivative(func, t, t), dt_dt_X)
        expr = expr.replace(sp.Derivative(func, t, r), dt_dr_X)
        expr = expr.replace(sp.Derivative(func, r, t), dt_dr_X)
        expr = expr.replace(sp.Derivative(func, t, z), dt_dz_X)
        expr = expr.replace(sp.Derivative(func, z, t), dt_dz_X)
        expr = expr.replace(sp.Derivative(func, t), dt_X)

    return sp.simplify(expr)


def clean_equation(expr):
    """Simplifies, removes denominators, and fixes sign."""
    # 1. Float to Rational & Together
    expr = sp.together(sp.nsimplify(expr, tolerance=1e-10, rational=True))

    # 2. Extract Numerator
    expr, _ = sp.fraction(expr)

    # 3. Primitive (remove integer GCD)
    _, expr = sp.primitive(expr)

    # 4. Factor and remove non-zero symbolic terms
    factored = sp.factor(expr)
    if factored.is_Mul:
        new_args = []
        for term in factored.args:
            if term.is_Number:
                continue
            if isinstance(term, sp.exp) or (term.is_Pow and term.base == sp.E):
                continue
            if term == Sigma or (term.is_Pow and term.base == Sigma):
                continue
            new_args.append(term)
        expr = sp.Mul(*new_args) if new_args else sp.Integer(1)

    # 5. Flip sign if leading term is negative
    terms = sp.expand(expr).as_ordered_terms()
    if terms:
        coeff, _ = terms[0].as_coeff_Mul()
        if coeff < 0:
            expr = -expr

    return expr
